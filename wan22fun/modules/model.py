"""Wan2.2Fun transformer loader.

This module keeps the Wan2.2Fun loading path separate from the repo's plain
``Wan22Model`` path.  The implementation inherits the Wan2.2 TI2V architecture
but adds the checkpoint loading convention used by Wan2.2Fun training scripts:
load the base transformer with ``from_pretrained`` and optionally merge sharded
``diffusion_pytorch_model-*.safetensors`` weights from ``transformer_path``.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors.torch import load_file

from wan22.modules.model import Wan22Model


class Wan22FunModel(Wan22Model):
    """Wan2.2Fun model class with a Wan2.2Fun-specific checkpoint loader."""

    @staticmethod
    def _checkpoint_files(transformer_path: str) -> Iterable[Path]:
        path = Path(transformer_path)
        if path.is_dir():
            sharded = [
                path / "diffusion_pytorch_model-00001-of-00002.safetensors",
                path / "diffusion_pytorch_model-00002-of-00002.safetensors",
            ]
            if all(p.exists() for p in sharded):
                return sharded
            safetensors = sorted(path.glob("*.safetensors"))
            if safetensors:
                return safetensors
            return sorted(path.glob("*.pt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.bin"))
        return [path]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        transformer_additional_kwargs: Optional[Dict] = None,
        transformer_path: Optional[str] = None,
        **kwargs,
    ):
        # ``transformer_additional_kwargs`` is accepted to mirror the upstream
        # Wan2.2Fun call site.  The architecture config is still loaded through
        # the base Wan2.2 ``ModelMixin.from_pretrained`` machinery.
        _ = transformer_additional_kwargs
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if transformer_path is None:
            return model

        checkpoint_files = list(cls._checkpoint_files(transformer_path))
        if not checkpoint_files:
            raise FileNotFoundError(f"No transformer checkpoint found under {transformer_path}")

        all_state_dict = {}
        for checkpoint_file in checkpoint_files:
            checkpoint_file = Path(checkpoint_file)
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Transformer checkpoint not found: {checkpoint_file}")
            if checkpoint_file.suffix == ".safetensors":
                state_dict = load_file(str(checkpoint_file))
            else:
                state_dict = torch.load(str(checkpoint_file), map_location="cpu")
            state_dict = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict
            all_state_dict.update(state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(all_state_dict, strict=False)
        print(
            f"Loaded Wan2.2Fun transformer checkpoint from {transformer_path}; "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )
        return model
