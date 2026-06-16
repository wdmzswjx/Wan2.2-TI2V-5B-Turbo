#!/usr/bin/env python3
"""Export generator weights from a Wan2.2Fun training checkpoint.

The training checkpoint saved by ``trainer/wan22fun_distillation.py`` contains a
wrapper-level ``generator`` state dict.  This utility strips FSDP/checkpoint
prefixes and, by default, removes the wrapper's leading ``model.`` prefix so the
result can be loaded directly by the raw transformer model, e.g.::

    from safetensors.torch import load_file
    state_dict = load_file("generator.safetensors")
    transformer.load_state_dict(state_dict, strict=False)
"""

import argparse
import os
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def strip_checkpoint_wrapper_prefix(name):
    """Remove wrapper prefixes added by FSDP/checkpointing/torch.compile."""
    for prefix in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
        name = name.replace(prefix, "")
    return name


def extract_generator_state_dict(checkpoint, use_ema=False, transformer_only=True):
    """Return generator weights extracted from a training checkpoint.

    Args:
        checkpoint: Path to ``model.pt`` or an already-loaded checkpoint dict.
        use_ema: Prefer ``generator_ema`` when available.
        transformer_only: Strip the wrapper ``model.`` prefix and drop non-model
            wrapper entries so the output can be loaded by the raw transformer.
    """
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu")

    key = "generator_ema" if use_ema and "generator_ema" in checkpoint else "generator"
    if key in checkpoint:
        state_dict = checkpoint[key]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    extracted = OrderedDict()
    for name, tensor in state_dict.items():
        clean_name = strip_checkpoint_wrapper_prefix(name)
        if transformer_only:
            if not clean_name.startswith("model."):
                continue
            clean_name = clean_name[len("model."):]
        extracted[clean_name] = tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
    return extracted


def save_generator_for_transformer(checkpoint_path, output_path, use_ema=False, transformer_only=True):
    """Extract and save generator weights for direct transformer loading."""
    state_dict = extract_generator_state_dict(
        checkpoint_path,
        use_ema=use_ema,
        transformer_only=transformer_only,
    )
    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.endswith(".safetensors"):
        save_file(state_dict, output_path)
    else:
        torch.save(state_dict, output_path)
    return output_path, len(state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to training checkpoint model.pt")
    parser.add_argument("output", help="Output .safetensors/.pt path")
    parser.add_argument("--ema", action="store_true", help="Export generator_ema when present")
    parser.add_argument(
        "--keep-wrapper-prefix",
        action="store_true",
        help="Keep wrapper-level keys instead of stripping leading model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, num_tensors = save_generator_for_transformer(
        args.checkpoint,
        args.output,
        use_ema=args.ema,
        transformer_only=not args.keep_wrapper_prefix,
    )
    print(f"Saved {num_tensors} generator tensors to {output_path}")


if __name__ == "__main__":
    main()
