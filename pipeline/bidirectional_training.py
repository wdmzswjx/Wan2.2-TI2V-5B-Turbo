from typing import List
import torch

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist
from utils.dataset import masks_like

class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def generate_and_sync_list(self, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()


    def _sigma_for_timestep(self, timestep: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        scheduler_timesteps = self.scheduler.timesteps.to(device=like.device)
        scheduler_sigmas = self.scheduler.sigmas.to(device=like.device)
        if timestep.ndim == 2:
            timestep = timestep[:, 0]
        timestep = timestep.to(device=like.device, dtype=scheduler_timesteps.dtype)
        step_indices = torch.argmin((scheduler_timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = scheduler_sigmas[step_indices].to(device=like.device, dtype=like.dtype)
        while sigma.ndim < like.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    @staticmethod
    def _image_latent_from_control_y(y, reference: torch.Tensor, vae_latent_channels: int = 48):
        """Extract Wan2.2Fun first-frame injection latent from the tail of y.

        Wan2.2Fun's control tensor can contain more than a plain reference
        latent.  The reliable injection signal is the last VAE-latent block in
        y, not a separately encoded first frame.  Return repo layout
        ``[B, 1, C, H, W]`` so it broadcasts over the noisy sample frames.
        """
        if y is None:
            return None
        if isinstance(y, torch.Tensor) and y.ndim == 5:
            if y.shape[1] == reference.shape[1] and y.shape[-2:] == reference.shape[-2:]:
                return y[:, :1, -vae_latent_channels:].to(device=reference.device, dtype=reference.dtype)
            if y.shape[2] == reference.shape[1] and y.shape[-2:] == reference.shape[-2:]:
                return y[:, -vae_latent_channels:, :1].permute(0, 2, 1, 3, 4).contiguous().to(
                    device=reference.device, dtype=reference.dtype
                )
        if isinstance(y, (list, tuple)) and len(y) > 0 and isinstance(y[0], torch.Tensor):
            first = y[0]
            if first.ndim == 4 and first.shape[0] == reference.shape[1]:
                stacked = torch.stack([item[:1, -vae_latent_channels:] for item in y], dim=0)
                return stacked.to(device=reference.device, dtype=reference.dtype)
            if first.ndim == 4 and first.shape[1] == reference.shape[1]:
                stacked = torch.stack([item[-vae_latent_channels:, :1].permute(1, 0, 2, 3) for item in y], dim=0)
                return stacked.to(device=reference.device, dtype=reference.dtype)
        return None

    def inference_with_trajectory(self, noise: torch.Tensor, clip_fea, y, y_camera=None, full_ref=None, wan22_image_latent=None, **conditional_dict) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        # initial point
        noisy_image_or_video = noise
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=noise.device)

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            timestep = torch.ones(
                noise.shape[:2],
                device=noise.device,
                dtype=torch.int64) * current_timestep
            
            if "2.2" in self.generator.model_name:
                mask1, mask2 = masks_like(noisy_image_or_video, zero=True)
                mask2 = torch.stack(mask2, dim=0) # torch.Size([1, 31, 48, 44, 80])
                control_image_latent = self._image_latent_from_control_y(y, noisy_image_or_video)
                if control_image_latent is None:
                    control_image_latent = wan22_image_latent
                noisy_image_or_video = (1. - mask2) * control_image_latent + mask2 * noisy_image_or_video
                noisy_image_or_video = noisy_image_or_video.to(noise.device, dtype=noise.dtype)
                seq_len = self.generator.get_seq_len(noisy_image_or_video)

                wan22_input_timestep = torch.tensor([timestep[0][0].item()], device=noise.device, dtype=noise.dtype)
                temp_ts = (mask2[:, :, 0, ::2, ::2] * wan22_input_timestep)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(temp_ts.size(0), seq_len - temp_ts.size(1)) * wan22_input_timestep,
                ], dim=1)
                wan22_input_timestep = temp_ts.to(noise.device, dtype=torch.long)
            else:
                mask1, mask2 = None, None
                control_image_latent = wan22_image_latent
                wan22_input_timestep = None

            if not exit_flag:
                with torch.no_grad():
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        clip_fea=clip_fea,
                        y=y,
                        y_camera=y_camera,
                        full_ref=full_ref,
                        wan22_input_timestep=wan22_input_timestep,
                        mask2=mask2,
                        wan22_image_latent=control_image_latent,
                    )  # [B, F, C, H, W]

                    next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device)
                    if getattr(self.generator, "use_wan22fun_model", False):
                        sigma_next = self._sigma_for_timestep(next_timestep, denoised_pred)
                        sigma_current = self._sigma_for_timestep(timestep, noisy_image_or_video)
                        pred_epsilon = noisy_image_or_video + (1 - sigma_current) * flow_pred
                        noisy_image_or_video = (1 - sigma_next) * denoised_pred + sigma_next * pred_epsilon
                        if mask2 is not None and control_image_latent is not None:
                            noisy_image_or_video = (1. - mask2) * control_image_latent + mask2 * noisy_image_or_video
                    else:
                        noisy_image_or_video = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep.flatten(0, 1)
                        ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    clip_fea=clip_fea,
                    y=y,
                    y_camera=y_camera,
                    full_ref=full_ref,
                    wan22_input_timestep=wan22_input_timestep,
                    mask2=mask2,
                    wan22_image_latent=control_image_latent,
                )  # [B, F, C, H, W]
                break

        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        return denoised_pred, denoised_timestep_from, denoised_timestep_to
