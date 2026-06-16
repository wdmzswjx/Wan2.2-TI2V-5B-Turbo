"""Wan22R score-distillation trainer.

This trainer intentionally does not define a dataset.  It expects the caller to
provide an existing dataloader factory through config, and adapts batches with
Wan22R keys (``video`` / ``ref_images`` / ``text``) into the repository DMD
model API.
"""

import importlib
import os
import random
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from model import Wan22RDMD
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import merge_dict_list, set_seed


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.is_main_process = self.global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.output_path = config.logdir

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + self.global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

        if config.distribution_loss not in ("wan22r", "wan22r_dmd"):
            raise ValueError("Wan22R trainer requires distribution_loss='wan22r'")
        self.model = Wan22RDMD(config, device=self.device)

        pretrained_ckpt_path, self.step = self.load(self.output_path)
        if pretrained_ckpt_path is not None:
            if self.is_main_process:
                print(f"Loading checkpoint from {pretrained_ckpt_path} at step {self.step}")
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.model.generator.load_state_dict(state_dict["generator"], strict=True)
            self.model.fake_score.load_state_dict(state_dict["critic"], strict=True)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters() if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
        )

        self.dataloader = self._build_dataloader(config)
        self.dataloader_iter = iter(self.dataloader)
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None
        if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

    @staticmethod
    def _import_target(target):
        module_name, attr_name = target.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), attr_name)

    def _build_dataloader(self, config):
        target = getattr(config, "wan22r_dataloader_target", None)
        if target is None:
            raise ValueError("Set wan22r_dataloader_target to your existing dataloader factory")
        factory = self._import_target(target)
        kwargs = OmegaConf.to_container(getattr(config, "wan22r_dataloader_kwargs", {}), resolve=True)
        return factory(config=config, rank=self.global_rank, world_size=self.world_size, **kwargs)

    @staticmethod
    def _checkpoint_step(folder_name):
        prefix = "checkpoint_model_"
        return int(folder_name[len(prefix):]) if folder_name.startswith(prefix) and folder_name[len(prefix):].isdigit() else -1

    def load(self, out_path):
        if not os.path.exists(out_path):
            return None, 0
        ckpt_folders = [f for f in os.listdir(out_path) if self._checkpoint_step(f) >= 0]
        if not ckpt_folders:
            return None, 0
        ckpt_folders.sort(key=self._checkpoint_step)
        latest = ckpt_folders[-1]
        model_path = os.path.join(out_path, latest, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")
        return model_path, self._checkpoint_step(latest)

    def save(self):
        state_dict = {
            "generator": fsdp_state_dict(self.model.generator),
            "critic": fsdp_state_dict(self.model.fake_score),
        }
        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step) and self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.state_dict()
        if self.is_main_process:
            ckpt_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
            print("Model saved to", os.path.join(ckpt_dir, "model.pt"))

    def _next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)

    def _prepare_wan22r_batch(self, batch):
        video = batch["video"].to(device=self.device, dtype=self.dtype)  # [B,C,F,H,W]
        ref_images = batch.get("ref_images")
        ref_image_num = 0
        if ref_images is not None and random.random() < getattr(self.config, "wan22r_ref_prob", 0.5):
            ref_images = ref_images.to(device=self.device, dtype=self.dtype)
            max_refs = min(getattr(self.config, "wan22r_max_ref_images", 4), ref_images.shape[2])
            if max_refs > 0:
                ref_image_num = random.randint(1, max_refs)
                video = torch.cat([ref_images[:, :, :ref_image_num], video], dim=2)

        latent = video.permute(0, 2, 1, 3, 4).contiguous()  # repo layout [B,F,C,H,W]
        visible_frames = getattr(self.config, "wan22r_visible_frames", 1)
        condition_frames = min(ref_image_num + visible_frames, latent.shape[1])
        condition_mask = torch.ones_like(latent, device=latent.device, dtype=latent.dtype)
        condition_mask[:, :condition_frames] = 0
        condition_latent = latent.detach()

        prompt_embeds = batch["text"].to(device=self.device, dtype=self.dtype)
        conditional_dict = {"prompt_embeds": prompt_embeds}
        unconditional_dict = {"prompt_embeds": torch.zeros_like(prompt_embeds)}
        return latent, condition_latent, condition_mask, conditional_dict, unconditional_dict

    def fwdbwd_one_step(self, batch, train_generator):
        latent, condition_latent, condition_mask, conditional_dict, unconditional_dict = self._prepare_wan22r_batch(batch)
        if train_generator:
            loss, log_dict = self.model.generator_loss(
                image_or_video_shape=list(latent.shape),
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=latent,
                condition_latent=condition_latent,
                condition_mask=condition_mask,
            )
            loss.backward()
            grad_norm = self.model.generator.clip_grad_norm_(getattr(self.config, "max_grad_norm_generator", 10.0))
            log_dict.update({"generator_loss": loss.detach(), "generator_grad_norm": grad_norm.detach()})
        else:
            loss, log_dict = self.model.critic_loss(
                image_or_video_shape=list(latent.shape),
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=latent,
                condition_latent=condition_latent,
                condition_mask=condition_mask,
            )
            loss.backward()
            grad_norm = self.model.fake_score.clip_grad_norm_(getattr(self.config, "max_grad_norm_critic", 10.0))
            log_dict.update({"critic_loss": loss.detach(), "critic_grad_norm": grad_norm.detach()})
        return log_dict

    def train(self):
        start_step = self.step
        while True:
            if self.is_main_process:
                print(f"training step {self.step} ...")
            train_generator = self.step % self.config.dfake_gen_update_ratio == 0
            if train_generator:
                self.generator_optimizer.zero_grad(set_to_none=True)
                generator_log_dict = merge_dict_list([self.fwdbwd_one_step(self._next_batch(), True)])
                if not self.config.debug:
                    self.generator_optimizer.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dict = merge_dict_list([self.fwdbwd_one_step(self._next_batch(), False)])
            if not self.config.debug:
                self.critic_optimizer.step()
            self.step += 1
            if (self.step >= self.ema_start_step) and self.generator_ema is None and self.ema_weight > 0:
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
            if self.is_main_process and not self.disable_wandb:
                logs = {"critic_loss": critic_log_dict["critic_loss"].mean().item()}
                if train_generator:
                    logs["generator_loss"] = generator_log_dict["generator_loss"].mean().item()
                wandb.log(logs, step=self.step)
            if self.step % self.config.gc_interval == 0:
                torch.cuda.empty_cache()
