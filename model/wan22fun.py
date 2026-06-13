"""Wan2.2Fun distillation method integrated with the repository framework.

This module intentionally implements a real ``SelfForcingModel``/DMD variant
instead of an external training-loop wrapper.  The trainer can select it with
``distribution_loss: wan22fun`` and continue to use the existing FSDP,
optimizer, EMA, dataloader and checkpointing code.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from model.dmd import DMD
from utils.dataset import masks_like


class Wan22FunDMD(DMD):
    """DMD-style distillation for Wan2.2Fun/Wan2.2-TI2V-5B models.

    Wan2.2Fun uses the Wan2.2 TI2V transformer interface where the first-frame
    condition is injected into latent tokens and the model receives an extended
    per-token timestep.  This class keeps that model-specific logic inside the
    distribution-matching model while preserving the repository's standard
    ``generator_loss``/``critic_loss`` API.
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        self.wan22fun_zero_first_frame_timestep = getattr(
            args, "wan22fun_zero_first_frame_timestep", True
        )
        self.wan22fun_mask_first_frame_loss = getattr(
            args, "wan22fun_mask_first_frame_loss", True
        )

    def _flow_sigmas(self, timestep: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Return flow-match sigma values broadcast to ``like`` dimensions."""
        if timestep.ndim == 2:
            timestep = timestep[:, 0]
        self.scheduler.timesteps = self.scheduler.timesteps.to(like.device)
        self.scheduler.sigmas = self.scheduler.sigmas.to(like.device)
        timestep = timestep.to(device=like.device, dtype=self.scheduler.timesteps.dtype)
        timestep_id = torch.argmin(
            (self.scheduler.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma = self.scheduler.sigmas[timestep_id].to(device=like.device, dtype=like.dtype)
        while sigma.ndim < like.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _build_wan22fun_condition(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        wan22_image_latent: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Inject TI2V image tokens and build Wan2.2Fun's per-token timestep.

        Args:
            noisy_latent: Noisy video latent in repo layout ``[B, F, C, H, W]``.
            timestep: Per-frame timestep in shape ``[B, F]`` or per-sample
                timestep in shape ``[B]``.
            wan22_image_latent: Encoded first-frame latent ``[B, 1, C, H, W]``.

        Returns:
            ``(model_input, mask2, wan22_input_timestep)``.  ``mask2`` is one on
            denoised/generated video tokens and zero on injected image tokens.
        """
        if wan22_image_latent is None:
            return noisy_latent, None, None

        if timestep.ndim == 2:
            sample_timestep = timestep[:, 0]
        else:
            sample_timestep = timestep

        _, mask2_list = masks_like(noisy_latent, zero=True)
        mask2 = torch.stack(mask2_list, dim=0).to(
            device=noisy_latent.device, dtype=noisy_latent.dtype
        )
        model_input = (1.0 - mask2) * wan22_image_latent + mask2 * noisy_latent
        model_input = model_input.to(device=self.device, dtype=self.dtype)

        seq_len = self.generator.get_seq_len(noisy_latent)
        image_token_timestep = sample_timestep[:, None, None, None].to(
            device=noisy_latent.device, dtype=torch.float32
        )
        if self.wan22fun_zero_first_frame_timestep:
            image_token_timestep = mask2[:, :, 0, ::2, ::2].float() * image_token_timestep
        else:
            image_token_timestep = torch.ones_like(mask2[:, :, 0, ::2, ::2].float()) * image_token_timestep
        image_token_timestep = image_token_timestep.flatten(1)

        if image_token_timestep.size(1) > seq_len:
            raise ValueError(
                f"Wan2.2Fun image-token timestep length {image_token_timestep.size(1)} "
                f"exceeds model seq_len {seq_len}."
            )

        video_token_timestep = image_token_timestep.new_ones(
            image_token_timestep.size(0), seq_len - image_token_timestep.size(1)
        ) * sample_timestep[:, None].float()
        wan22_input_timestep = torch.cat(
            [image_token_timestep, video_token_timestep], dim=1
        ).to(device=noisy_latent.device, dtype=torch.long)
        return model_input, mask2, wan22_input_timestep

    def _predict_wan22fun_x0(
        self,
        score_model,
        noisy_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_camera: Optional[torch.Tensor] = None,
        full_ref: Optional[torch.Tensor] = None,
        wan22_image_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run a Wan2.2Fun score model and return ``flow_pred, pred_x0, mask2``."""
        model_input, mask2, wan22_input_timestep = self._build_wan22fun_condition(
            noisy_latent=noisy_latent,
            timestep=timestep,
            wan22_image_latent=wan22_image_latent,
        )
        flow_pred, pred_x0 = score_model(
            noisy_image_or_video=model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_input_timestep=wan22_input_timestep,
            mask2=mask2,
            wan22_image_latent=wan22_image_latent,
        )
        if mask2 is not None and wan22_image_latent is not None:
            pred_x0 = (1.0 - mask2) * wan22_image_latent + mask2 * pred_x0
        return flow_pred, pred_x0, mask2

    def _cfg_wan22fun_x0(
        self,
        score_model,
        noisy_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        unconditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        guidance_scale: float,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        y_camera: Optional[torch.Tensor] = None,
        full_ref: Optional[torch.Tensor] = None,
        wan22_image_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return classifier-free guided x0 and the Wan2.2Fun loss mask."""
        _, pred_cond, mask2 = self._predict_wan22fun_x0(
            score_model=score_model,
            noisy_latent=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )
        if guidance_scale == 0.0:
            return pred_cond, mask2

        _, pred_uncond, _ = self._predict_wan22fun_x0(
            score_model=score_model,
            noisy_latent=noisy_latent,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )
        pred_cfg = pred_cond + (pred_cond - pred_uncond) * guidance_scale
        if mask2 is not None and wan22_image_latent is not None:
            pred_cfg = (1.0 - mask2) * wan22_image_latent + mask2 * pred_cfg
        return pred_cfg, mask2

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        wan22_image_latent=None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute DMD gradient with Wan2.2Fun conditioning."""
        pred_fake_image, mask2 = self._cfg_wan22fun_x0(
            score_model=self.fake_score,
            noisy_latent=noisy_image_or_video,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            timestep=timestep,
            guidance_scale=self.fake_guidance_scale,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )
        pred_real_image, _ = self._cfg_wan22fun_x0(
            score_model=self.real_score,
            noisy_latent=noisy_image_or_video,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            timestep=timestep,
            guidance_scale=self.real_guidance_scale,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )

        grad = pred_fake_image - pred_real_image
        if normalization:
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer.clamp(min=1e-8)
        if self.wan22fun_mask_first_frame_loss and mask2 is not None:
            grad = grad * mask2
        grad = torch.nan_to_num(grad)
        return grad, {
            "wan22fun_dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        clip_fea: torch.Tensor = None,
        y: torch.Tensor = None,
        y_camera: torch.Tensor = None,
        full_ref: torch.Tensor = None,
        wan22_image_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        original_latent = image_or_video
        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True,
            )
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * (timestep / 1000) / (
                    1 + (self.timestep_shift - 1) * (timestep / 1000)
                ) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clip_fea=clip_fea,
                y=y,
                y_camera=y_camera,
                full_ref=full_ref,
                wan22_image_latent=wan22_image_latent,
            )

        target = (original_latent.double() - grad.double()).detach()
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double()[gradient_mask], target[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(), target, reduction="mean")
        dmd_log_dict["wan22fun_distribution_matching_loss"] = dmd_loss.detach()
        return dmd_loss, dmd_log_dict


    @staticmethod
    def _repo_shape_from_condition_latent(y: Optional[torch.Tensor], fallback_shape) -> list:
        """Infer repo-layout latent shape from Wan2.2Fun control latents.

        Wan2.2Fun control latents are the same spatio-temporal latent grid as
        the noisy sample.  Prefer them over static config/default frame counts
        so changing dataset frames (for example 81 input frames -> 21 latent
        frames) also changes the generated noise shape.
        """
        shape = list(fallback_shape)
        if y is None:
            return shape

        if isinstance(y, torch.Tensor) and y.ndim == 5:
            # Repo layout: [B, F, C, H, W].
            if y.shape[1] == shape[1] and y.shape[2] == shape[2]:
                return list(y.shape)
            # Model layout: [B, C, F, H, W].
            if y.shape[1] == shape[2]:
                return [y.shape[0], y.shape[2], y.shape[1], y.shape[3], y.shape[4]]
            # If the caller supplied a stale frame count in fallback_shape, the
            # channel dimension still identifies repo layout unambiguously.
            if y.shape[2] == shape[2]:
                return list(y.shape)
            return shape

        if isinstance(y, (list, tuple)) and len(y) > 0:
            first = y[0]
            if isinstance(first, torch.Tensor) and first.ndim == 4:
                # Per-sample repo layout: [F, C, H, W].
                if first.shape[1] == shape[2]:
                    return [len(y), first.shape[0], first.shape[1], first.shape[2], first.shape[3]]
                # Per-sample model layout: [C, F, H, W].
                if first.shape[0] == shape[2]:
                    return [len(y), first.shape[1], first.shape[0], first.shape[2], first.shape[3]]
        return shape

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
        clip_fea: torch.Tensor = None,
        y: torch.Tensor = None,
        y_camera: torch.Tensor = None,
        full_ref: torch.Tensor = None,
        wan22_image_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Wan2.2Fun-local generator runner with control/ref arguments.

        Keep this method in the Wan2.2Fun implementation instead of depending on
        ``DMD``/``SelfForcingModel`` signatures, so normal DMD training is not
        affected by Wan2.2Fun-only arguments such as ``y_camera`` and
        ``full_ref``.  Wan2.2Fun also keeps the requested latent shape exactly
        as supplied by the trainer/batch; it does not hard-code a model-specific
        frame count or crop the generated trajectory to a fixed clip length.
        """
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        conditional_dict = dict(conditional_dict)
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        noise_shape = self._repo_shape_from_condition_latent(y, image_or_video_shape)
        if wan22_image_latent is not None:
            noise_shape[0] = wan22_image_latent.shape[0]
            noise_shape[2:] = list(wan22_image_latent.shape[2:])

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
            **conditional_dict,
        )

        pred_image_or_video = pred_image_or_video.to(self.dtype)
        return pred_image_or_video, None, denoised_timestep_from, denoised_timestep_to

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        clip_fea: torch.Tensor = None,
        y: torch.Tensor = None,
        y_camera: torch.Tensor = None,
        full_ref: torch.Tensor = None,
        wan22_image_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Run Wan2.2Fun generator and compute DMD loss with control inputs.

        This override is intentionally explicit so the Wan2.2Fun-specific
        control latents (``y``), camera controls and ``full_ref`` are forwarded
        through the self-forcing trajectory instead of relying on the base DMD
        signature.
        """
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        clip_fea: torch.Tensor = None,
        y: torch.Tensor = None,
        y_camera: torch.Tensor = None,
        full_ref: torch.Tensor = None,
        wan22_image_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Train fake score on generated Wan2.2Fun samples using repo DMD API."""
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                clip_fea=clip_fea,
                y=y,
                y_camera=y_camera,
                full_ref=full_ref,
                wan22_image_latent=wan22_image_latent,
            )

        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (critic_timestep / 1000) / (
                1 + (self.timestep_shift - 1) * (critic_timestep / 1000)
            ) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, image_or_video_shape[:2])

        flow_pred, pred_fake_image, mask2 = self._predict_wan22fun_x0(
            score_model=self.fake_score,
            noisy_latent=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            clip_fea=clip_fea,
            y=y,
            y_camera=y_camera,
            full_ref=full_ref,
            wan22_image_latent=wan22_image_latent,
        )

        if self.args.denoising_loss_type == "flow":
            pred_fake_noise = None
        elif self.args.denoising_loss_type == "noise":
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, image_or_video_shape[:2])
        else:
            flow_pred = None
            pred_fake_noise = None

        if self.wan22fun_mask_first_frame_loss and mask2 is not None:
            mask = mask2.float()
            denom = mask.sum().clamp(min=1.0)
            if self.args.denoising_loss_type == "flow":
                target_flow = critic_noise - generated_image
                denoising_loss = ((flow_pred.float() - target_flow.float()).square() * mask).sum() / denom
            elif self.args.denoising_loss_type == "noise":
                denoising_loss = ((pred_fake_noise.float() - critic_noise.float()).square() * mask).sum() / denom
            else:
                denoising_loss = ((pred_fake_image.float() - generated_image.float()).square() * mask).sum() / denom
        else:
            denoising_loss = self.denoising_loss_func(
                x=generated_image.flatten(0, 1),
                x_pred=pred_fake_image.flatten(0, 1),
                noise=critic_noise.flatten(0, 1),
                noise_pred=pred_fake_noise,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=critic_timestep.flatten(0, 1),
                flow_pred=flow_pred,
            )

        return denoising_loss, {
            "critic_timestep": critic_timestep.detach(),
            "wan22fun_critic_loss": denoising_loss.detach(),
        }
