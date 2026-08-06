"""Fold the MiniMax-H3 Turbo LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`) into the diffusers transformer.

The LoRA ships against the *reference* (ComfyUI) module tree — `blocks.N.attn.qkv_proj`, `blocks.N.mlp.fc1`,
`token_refiner.blocks.N`, `final_layer.adaln_proj.linear` — with `alpha == rank`, so the update is exactly
`W + lora_B @ lora_A`. The diffusers checkpoint is the same weights under different names and two layout transforms
(see `scripts/convert_minimax_h3_to_diffusers.py` in huggingface/diffusers#14371), so each delta gets the same
transform the base weight got:

* fused `attn.qkv_proj` rows are `[q_all; k_all; v_all]` in both in-memory layouts -> split into contiguous thirds
  onto `attn.to_q` / `to_k` / `to_v`;
* fused `mlp.fc1` is `[gate; value]` while diffusers' `SwiGLU` fuses `[value; gate]` -> swap the halves onto
  `ff.net.0.proj`;
* `mlp.fc2` -> `ff.net.2`, `attn.out_proj` -> `attn.to_out.0`, `blocks.` -> `transformer_blocks.`,
  `token_refiner.blocks.` -> `token_refiner.refiner_blocks.`, `final_layer.adaln_proj.linear` -> `norm_out.linear`;
* the `adaln_proj.linear` modulation tables share the `[timestep][modality][param]` row layout in both trees, so
  they map name-for-name with no reordering.

The delta is folded into the bf16 weights rather than applied as a runtime wrapper for one reason: the AoTI block
package (`h3_aoti`) reads each block's live weights, and a wrapper module would be invisible to it. The fold computes
`lora_B @ lora_A` in float32 and rounds once on the way back into bf16.
"""

from __future__ import annotations

import os

import torch

LORA_REPO = os.environ.get("H3_LORA_REPO", "larryvrh/MiniMax-H3-Turbo-Lora")
# The trained weights; the `_ema` variant is an immature time-averaged snapshot at this checkpoint. `off` disables.
LORA_FILE = os.environ.get("H3_LORA", "minimax_h3_turbo_4step.safetensors")


def _delta_targets(name: str, delta: torch.Tensor, inner_dim: int) -> list[tuple[str, torch.Tensor]]:
    """Map one reference-tree LoRA base name and its `lora_B @ lora_A` delta onto diffusers parameter key(s)."""
    if name.startswith("token_refiner.blocks."):
        target = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif name.startswith("blocks."):
        target = name.replace("blocks.", "transformer_blocks.", 1)
    else:
        target = name
    target = target.replace("final_layer.adaln_proj.linear", "norm_out.linear")

    if target.endswith(".attn.qkv_proj"):
        prefix = target.removesuffix("qkv_proj")
        return [
            (f"{prefix}to_{kind}.weight", part.contiguous())
            for kind, part in zip(("q", "k", "v"), delta.split(inner_dim, dim=0))
        ]
    if target.endswith(".mlp.fc1"):
        gate, value = delta.chunk(2, dim=0)
        return [(target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", torch.cat([value, gate]).contiguous())]
    if target.endswith(".mlp.fc2"):
        return [(target.replace(".mlp.fc2", ".ff.net.2") + ".weight", delta)]
    if target.endswith(".attn.out_proj"):
        return [(target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", delta)]
    # `adaln_proj.linear` (block-level and the final `norm_out.linear`): identical row layout on both sides.
    return [(target + ".weight", delta)]


def apply_lora(transformer) -> str | None:
    """Fold the configured Turbo LoRA into `transformer` in place. Returns a status line, or `None` when disabled."""
    if LORA_FILE.lower() in ("", "off", "none"):
        return None

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(LORA_REPO, LORA_FILE)
    lora = load_file(path)
    bases = sorted({key.rsplit(".lora_", 1)[0] for key in lora})

    config = transformer.config
    inner_dim = config.num_attention_heads * config.attention_head_dim
    params = dict(transformer.named_parameters())

    folded = 0
    for name in bases:
        a = lora[f"{name}.lora_A.weight"].float()
        b = lora[f"{name}.lora_B.weight"].float()
        delta = b @ a  # alpha == rank, so the scale is 1
        for key, converted in _delta_targets(name, delta, inner_dim):
            param = params.get(key)
            if param is None:
                raise KeyError(f"LoRA target `{key}` (from `{name}`) not found in the transformer")
            param.data = (param.data.float() + converted).to(param.dtype)
            folded += 1
    return f"LoRA `{LORA_REPO}/{LORA_FILE}` folded into {folded} weights ({len(bases)} modules)"
