"""Wan2.2-R distillation components.

Wan22R follows a reconstruction/generation input convention where the raw
transformer receives latent tensors in ``[B, C, F, H, W]`` and an extended
per-token timestep.  Reference frames ``R`` and visible condition frames ``V``
are kept clean in the noisy input; only the remaining target frames are noised
and trained/distilled.  The repository wrapper still uses its public
``[B, F, C, H, W]`` layout, so this module carries the Wan22R mask/timestep
policy while reusing the common DMD loss machinery.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from model.wan22fun import Wan22FunDMD


class Wan22RDMD(Wan22FunDMD):
    """DMD-style distillation for Wan22R-style masked frame prediction."""

    def _build_wan22r_condition(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        condition_latent: Optional[torch.Tensor],
        condition_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Keep condition frames clean and build Wan-style per-token timesteps.

        ``condition_mask`` uses repo layout ``[B, F, C, H, W]`` with ``0`` for
        clean reference/visible frames and ``1`` for generated target frames.
        """
        if condition_mask is None:
            return noisy_latent, None, None

        condition_mask = condition_mask.to(device=noisy_latent.device, dtype=noisy_latent.dtype)
        if condition_latent is None:
            condition_latent = noisy_latent
        condition_latent = condition_latent.to(device=noisy_latent.device, dtype=noisy_latent.dtype)
        model_input = condition_mask * noisy_latent + (1.0 - condition_mask) * condition_latent

        sample_timestep = timestep[:, 0] if timestep.ndim == 2 else timestep
        seq_len = self.generator.get_seq_len(noisy_latent)
        token_timestep = (condition_mask[:, :, 0, ::2, ::2] * sample_timestep[:, None, None, None].float()).flatten(1)
        if token_timestep.size(1) > seq_len:
            raise ValueError(
                f"Wan22R condition timestep length {token_timestep.size(1)} exceeds model seq_len {seq_len}."
            )
        pad_timestep = token_timestep.new_ones(token_timestep.size(0), seq_len - token_timestep.size(1)) * sample_timestep[:, None].float()
        wan22_input_timestep = torch.cat([token_timestep, pad_timestep], dim=1).to(
            device=noisy_latent.device, dtype=torch.long
        )
        return model_input, condition_mask, wan22_input_timestep

    def _predict_wan22r_x0(
        self,
        score_model,
        noisy_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        condition_latent: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        model_input, mask, wan22_input_timestep = self._build_wan22r_condition(
            noisy_latent=noisy_latent,
            timestep=timestep,
            condition_latent=condition_latent,
            condition_mask=condition_mask,
        )
        flow_pred, pred_x0 = score_model(
            noisy_image_or_video=model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            wan22_input_timestep=wan22_input_timestep,
            mask2=mask,
            wan22_image_latent=condition_latent,
        )
        if mask is not None and condition_latent is not None:
            pred_x0 = mask * pred_x0 + (1.0 - mask) * condition_latent
        return flow_pred, pred_x0, mask

    def _cfg_wan22r_x0(
        self,
        score_model,
        noisy_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        guidance_scale: float,
        condition_latent: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, pred_cond, mask = self._predict_wan22r_x0(
            score_model, noisy_latent, conditional_dict, timestep, condition_latent, condition_mask
        )
        if guidance_scale == 0.0:
            return pred_cond, mask
        _, pred_uncond, _ = self._predict_wan22r_x0(
            score_model, noisy_latent, unconditional_dict, timestep, condition_latent, condition_mask
        )
        pred_cfg = pred_cond + (pred_cond - pred_uncond) * guidance_scale
        if mask is not None and condition_latent is not None:
            pred_cfg = mask * pred_cfg + (1.0 - mask) * condition_latent
        return pred_cfg, mask

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        condition_latent: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        **_,
    ) -> Tuple[torch.Tensor, dict]:
        pred_fake_image, mask = self._cfg_wan22r_x0(
            self.fake_score,
            noisy_image_or_video,
            conditional_dict,
            unconditional_dict,
            timestep,
            self.fake_guidance_scale,
            condition_latent,
            condition_mask,
        )
        pred_real_image, _ = self._cfg_wan22r_x0(
            self.real_score,
            noisy_image_or_video,
            conditional_dict,
            unconditional_dict,
            timestep,
            self.real_guidance_scale,
            condition_latent,
            condition_mask,
        )
        grad = pred_fake_image - pred_real_image
        if normalization:
            normalizer = torch.abs(estimated_clean_image_or_video - pred_real_image).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer.clamp(min=1e-8)
        if mask is not None:
            grad = grad * mask
        grad = torch.nan_to_num(grad)
        return grad, {
            "wan22r_dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
        condition_latent: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        **_,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        conditional_dict = dict(conditional_dict)
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=noise,
            clip_fea=None,
            condition_latent=condition_latent,
            condition_mask=condition_mask,
            **conditional_dict,
        )
        return pred_image_or_video.to(self.dtype), condition_mask.bool() if condition_mask is not None else None, denoised_timestep_from, denoised_timestep_to

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        condition_latent: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        **_,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frame = image_or_video.shape[:2]
        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(min_timestep, max_timestep, batch_size, num_frame, self.num_frame_per_block, uniform_timestep=True)
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * (timestep / 1000) / (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_latent,
                image_or_video,
                timestep,
                conditional_dict,
                unconditional_dict,
                condition_latent=condition_latent,
                condition_mask=condition_mask,
            )
        target = (image_or_video.double() - grad.double()).detach()
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(image_or_video.double()[gradient_mask], target[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(image_or_video.double(), target, reduction="mean")
        dmd_log_dict["wan22r_distribution_matching_loss"] = dmd_loss.detach()
        return dmd_loss, dmd_log_dict

    def generator_loss(self, image_or_video_shape, conditional_dict, unconditional_dict, clean_latent, condition_latent=None, condition_mask=None, **kwargs):
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            condition_latent=condition_latent,
            condition_mask=condition_mask,
            **kwargs,
        )
        return self.compute_distribution_matching_loss(
            pred_image,
            conditional_dict,
            unconditional_dict,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
            condition_latent=condition_latent,
            condition_mask=condition_mask,
        )

    def critic_loss(self, image_or_video_shape, conditional_dict, unconditional_dict, clean_latent, condition_latent=None, condition_mask=None, **kwargs):
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                condition_latent=condition_latent,
                condition_mask=condition_mask,
                **kwargs,
            )
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(min_timestep, max_timestep, image_or_video_shape[0], image_or_video_shape[1], self.num_frame_per_block, uniform_timestep=True)
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)
        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1), critic_noise.flatten(0, 1), critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])
        flow_pred, pred_fake_image, mask = self._predict_wan22r_x0(
            self.fake_score, noisy_generated_image, conditional_dict, critic_timestep, condition_latent, condition_mask
        )
        if self.args.denoising_loss_type == "flow":
            target_flow = critic_noise - generated_image
            loss_tensor = (flow_pred.float() - target_flow.float()).square()
        else:
            loss_tensor = (pred_fake_image.float() - generated_image.float()).square()
        if mask is not None:
            loss = (loss_tensor * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss_tensor.mean()
        return loss, {"critic_timestep": critic_timestep.detach(), "wan22r_critic_loss": loss.detach()}
