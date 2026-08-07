"""Turbo LoRA support for the diffusers MiniMax-H3 transformer: two 4-step LoRAs, one fold mechanism.

Both LoRAs are applied by folding `scale * (lora_B @ lora_A)` into the bf16 weights rather than as runtime wrappers,
because the AoTI block package (`h3_aoti`) reads each block's live weights and a wrapper module would be invisible
to it. Deltas are computed in float32 and round once on the way back into bf16. The low-rank factors of every loaded
LoRA stay resident, so the active one can be switched per request (`set_active`) — unfold the old, fold the new, in
place, through one bf16 rounding.

The two supported LoRAs ship in different layouts:

* `larry` (`larryvrh/MiniMax-H3-Turbo-Lora`) targets the *reference* (ComfyUI) module tree — `blocks.N.attn.qkv_proj`,
  `blocks.N.mlp.fc1`, `token_refiner.blocks.N`, `final_layer.adaln_proj.linear` — with `alpha == rank` (scale 1).
  Each delta gets the same transform the base weights got in the diffusers conversion
  (`scripts/convert_minimax_h3_to_diffusers.py`, huggingface/diffusers#14371): fused-QKV row thirds onto
  `attn.to_q/k/v`, the `SwiGLU` gate/value swap onto `ff.net.0.proj`, `fc2` -> `ff.net.2`,
  `blocks.` -> `transformer_blocks.`, `token_refiner.blocks.` -> `token_refiner.refiner_blocks.`,
  `final_layer.adaln_proj.linear` -> `norm_out.linear`. The row transforms are applied to `lora_B` directly
  (rows of `B @ A` are rows of `B`), so no full delta is ever materialized at load.

* `lightx` (`lightx2v/Minimax-h3-Turbo`) is a PEFT checkpoint against the diffusers tree itself —
  `transformer_blocks.N.attn.to_q.lora_A.default.weight` and friends — rank 128, `alpha == 8`, so the fold scale is
  `8 / 128 = 0.0625` (matching `set_adapters(weights=1.0)` in their inference script). Keys map name-for-name.

`H3_LORA` selects the larry file (`off` skips loading it), `H3_LIGHTX=off` skips lightx, `H3_LORA_DEFAULT` picks
which set starts folded, and `H3_LORA_STRENGTH` is the larry card's sharpness/artifact dial.
"""

from __future__ import annotations

import os

import torch

LARRY_REPO = os.environ.get("H3_LORA_REPO", "larryvrh/MiniMax-H3-Turbo-Lora")
# The recommended default per the model card: ckpt850 EMA (final checkpoint of the round, sharp at 4 steps).
LARRY_FILE = os.environ.get("H3_LORA", "minimax_h3_turbo_4step_ema_ckpt850.safetensors")
LIGHTX_REPO = os.environ.get("H3_LIGHTX_REPO", "lightx2v/Minimax-h3-Turbo")
LIGHTX_FILE = os.environ.get("H3_LIGHTX_FILE", "minimax_h3_fl2v_turbo_4step_v0.1.safetensors")
LIGHTX_ALPHA = 8
# The card's sharpness/artifact dial for the larry LoRA: >1 against blurry ghosting/smear, <1 against grain.
LARRY_STRENGTH = float(os.environ.get("H3_LORA_STRENGTH", "1.0"))
DEFAULT_LORA = os.environ.get("H3_LORA_DEFAULT", "larry")


def _larry_targets(name: str, b: torch.Tensor, inner_dim: int) -> list[tuple[str, torch.Tensor]]:
    """Map one reference-tree base name and its `lora_B` onto diffusers parameter key + row-transformed B."""
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
            for kind, part in zip(("q", "k", "v"), b.split(inner_dim, dim=0))
        ]
    if target.endswith(".mlp.fc1"):
        gate, value = b.chunk(2, dim=0)
        return [(target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", torch.cat([value, gate]).contiguous())]
    if target.endswith(".mlp.fc2"):
        return [(target.replace(".mlp.fc2", ".ff.net.2") + ".weight", b)]
    if target.endswith(".attn.out_proj"):
        return [(target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", b)]
    # `adaln_proj.linear` (block-level and the final `norm_out.linear`): identical row layout on both sides.
    return [(target + ".weight", b)]


def _load_larry(inner_dim: int) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(LARRY_REPO, LARRY_FILE))
    bases = sorted({key.rsplit(".lora_", 1)[0] for key in lora})
    entries = []
    for name in bases:
        a = lora[f"{name}.lora_A.weight"]
        b = lora[f"{name}.lora_B.weight"]
        entries.extend((key, a, b_part) for key, b_part in _larry_targets(name, b, inner_dim))
    return {
        "label": f"{LARRY_REPO}/{LARRY_FILE}",
        "scale": LARRY_STRENGTH,  # alpha == rank, so the base scale is 1
        "entries": entries,
    }


def _load_lightx() -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = load_file(hf_hub_download(LIGHTX_REPO, LIGHTX_FILE))
    suffix_a, suffix_b = ".lora_A.default.weight", ".lora_B.default.weight"
    bases = sorted({key[: -len(suffix_a)] for key in lora if key.endswith(suffix_a)})
    ranks = {lora[f"{name}{suffix_a}"].shape[0] for name in bases}
    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks in {LIGHTX_FILE}: {sorted(ranks)}")
    entries = [(f"{name}.weight", lora[f"{name}{suffix_a}"], lora[f"{name}{suffix_b}"]) for name in bases]
    return {
        "label": f"{LIGHTX_REPO}/{LIGHTX_FILE}",
        "scale": LIGHTX_ALPHA / ranks.pop(),
        "entries": entries,
    }


def _apply(entries, params, sign: float) -> None:
    for key, a, b in entries:
        param = params.get(key)
        if param is None:
            raise KeyError(f"LoRA target `{key}` not found in the transformer")
        delta = sign * (b.to(torch.float32) @ a.to(torch.float32))
        param.data = (param.data.float() + delta.to(param.device)).to(param.dtype)


def available() -> list[str]:
    """The LoRA sets that were loaded at startup, plus `off`."""
    state = getattr(_PIPE_TRANSFORMER, "_lora_state", None) if _PIPE_TRANSFORMER is not None else None
    return sorted(state["sets"]) + ["off"] if state else ["off"]


_PIPE_TRANSFORMER = None


def apply_lora(transformer) -> str | None:
    """Load every enabled LoRA set, fold the default one into `transformer`, and stash the factors for per-request
    switching. Returns a status line, or `None` when everything is disabled."""
    global _PIPE_TRANSFORMER
    _PIPE_TRANSFORMER = transformer

    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    sets = {}
    if LARRY_FILE.lower() not in ("", "off", "none"):
        sets["larry"] = _load_larry(inner_dim)
    if os.environ.get("H3_LIGHTX", "on").lower() not in ("", "off", "none"):
        sets["lightx"] = _load_lightx()
    if not sets:
        return None

    active = DEFAULT_LORA if DEFAULT_LORA in sets else sorted(sets)[0]
    params = dict(transformer.named_parameters())
    _apply(sets[active]["entries"], params, sets[active]["scale"])
    transformer._lora_state = {"active": active, "sets": sets}
    return (
        f"LoRAs loaded: "
        + ", ".join(f"`{name}` ({spec['label']}, {len(spec['entries'])} weights)" for name, spec in sets.items())
        + f" · active `{active}`"
    )


def set_active(transformer, name: str) -> str:
    """Switch the folded LoRA in place. No-op when the state already matches. Returns the active set."""
    state = getattr(transformer, "_lora_state", None)
    if state is None:
        return "off"
    name = name if name in state["sets"] else "off"
    if state["active"] == name:
        return name
    params = dict(transformer.named_parameters())
    if state["active"] != "off":
        old = state["sets"][state["active"]]
        _apply(old["entries"], params, -old["scale"])
    if name != "off":
        _apply(state["sets"][name]["entries"], params, state["sets"][name]["scale"])
    state["active"] = name
    return name


def set_enabled(transformer, enabled: bool) -> bool:
    """Backwards-compatible boolean toggle over the default set."""
    return set_active(transformer, DEFAULT_LORA if enabled else "off") != "off"
