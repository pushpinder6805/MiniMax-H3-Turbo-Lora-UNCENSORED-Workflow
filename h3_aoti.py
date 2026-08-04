"""ZeroGPU AoTI for MiniMax-H3: one compiled `MiniMaxH3TransformerBlock` package, reused by all 50 blocks.

Shared byte-identically by every MiniMax-H3 Space. A Space only calls `maybe_load()`; the rest is the build path.
"""

from __future__ import annotations

import os
from pathlib import Path

AOTI = os.environ.get("H3_AOTI", "0") == "1"
AOTI_REPO = os.environ.get("H3_AOTI_REPO", "multimodalart/minimax-h3-aoti")
AOTI_REPO_TYPE = os.environ.get("H3_AOTI_REPO_TYPE", "model")
# A package is valid for exactly one `<width>/torch<X.Y>/sm<cc>/<shape>`, and a mismatched one segfaults rather than
# raising, so `maybe_load` refuses anything but this key.
AOTI_KEY = os.environ.get("H3_AOTI_KEY", "bf16/torch2.11/sm120/dynamic")
# `dynamic` is the sequence dimension: `build_packed_sequence` pads nothing, so `S` moves with the prompt as well as
# the canvas and a static package would serve one prompt length.
AOTI_SHAPE = os.environ.get("H3_AOTI_SHAPE", "dynamic")
AOTI_DURATION = int(os.environ.get("H3_AOTI_DURATION", "1500"))

# Where a step spends its time. `MiniMaxH3TokenRefinerBlock` is also repeated but runs a handful of text rows.
BLOCK_CONTAINER = "transformer_blocks"

# Height of the AdaLN table baked into the package. `temb` grows from 1 row (step 0, both streams at one noise level)
# to 2 (from step 1, sigmas diverged), and the block gathers from `3 * rows`, so the row count is part of the compiled
# shape and is pinned by padding on both sides of the compile. Must match the package's `H3_AOTI_TEMB_ROWS`.
TEMB_ROWS = int(os.environ.get("H3_AOTI_TEMB_ROWS", "4"))

_LOADED: set[int] = set()


def pad_temb(temb, rows: int = TEMB_ROWS):
    """Grow `temb` to exactly `rows` timestep rows by repeating its last one."""
    present = temb.shape[0]
    if present == rows:
        return temb
    if present > rows:
        raise RuntimeError(
            f"{present} distinct timesteps, but this AoTI package holds at most {rows}. "
            f"Recompile with H3_AOTI_TEMB_ROWS>={present}."
        )
    import torch

    return torch.cat([temb, temb[-1:].expand(rows - present, *temb.shape[1:])], dim=0)


def width() -> str:
    """Which transformer these artifacts belong to: `bf16`, `fp8`, `nvfp4`, ..."""
    if explicit := os.environ.get("H3_WIDTH"):
        return explicit.lower()
    try:
        import h3_core

        return h3_core.WIDTH
    except Exception:
        return "bf16"


def artifact_key() -> str | None:
    """`<width>/torch<X.Y>/sm<cc>/<shape>` of the card this process is on, or `None` when there is no CUDA."""
    try:
        import torch

        torch_version = ".".join(torch.__version__.split(".")[:2])
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"{width()}/torch{torch_version}/sm{major}{minor}/{AOTI_SHAPE}"


def status() -> str:
    return (
        f"AoTI **on** · `{AOTI_REPO}` ({AOTI_REPO_TYPE}) · shape `{AOTI_SHAPE}`"
        if AOTI
        else "AoTI **off** (`H3_AOTI=1` to load compiled blocks)"
    )


def patch_blocks(transformer, package_dir) -> None:
    """Point all 50 blocks at the one compiled package, binding each block's own weights on its first call.

    `spaces.aoti_load_from_package_dir` with two changes. Weights are read on the first forward rather than at patch
    time, because this runs at startup and `Module.to` later rebinds `param.data` to fresh CUDA tensors. And `temb` is
    padded to the height the package was exported with — see `TEMB_ROWS`.
    """
    from spaces.zero.torch.aoti import LazyAOTIModel, _shallow_clone_module
    from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters

    # `LazyAOTIModel` binds constants by name and silently keeps what it cannot match, which is a SIGSEGV rather than
    # an error. The patch resolves anonymous names through the compile side's sidecar and raises if it still cannot.
    try:
        from spaces_constant_binding_patch import apply_spaces_constant_binding_patch

        apply_spaces_constant_binding_patch()
    except ImportError:
        print("[h3-aoti] spaces_constant_binding_patch.py is missing; an unbindable constant would segfault", flush=True)

    model = LazyAOTIModel(Path(package_dir) / "submodules" / BLOCK_CONTAINER / "package.pt2")

    for block in getattr(transformer, BLOCK_CONTAINER):
        bound: dict = {}

        def forward(hidden_states, temb, *rest, _block=block, _bound=bound):
            first = not _bound
            if first:
                clone = _shallow_clone_module(_block)
                unwrap_tensor_subclass_parameters(clone)
                _bound["weights"] = clone.state_dict()
            return model(_bound["weights"], first, hidden_states, pad_temb(temb), *rest)

        block.forward = forward
    print(f"[h3-aoti] {len(getattr(transformer, BLOCK_CONTAINER))} blocks patched (temb padded to {TEMB_ROWS})", flush=True)


def maybe_load(transformer) -> None:
    """Patch the block stack with its compiled package, or leave it eager. Safe to call at **startup**.

    Off unless `H3_AOTI=1`, and anything that does not line up — another card, another torch, no `spaces` AoTI
    helpers, no published package — falls back to eager with one line rather than raising or segfaulting. Nothing here
    touches a GPU: the download is CPU work and the `.pt2` is not opened until the first forward.
    """
    if not AOTI or id(transformer) in _LOADED:
        return

    key = artifact_key()
    if key is None:
        print("[h3-aoti] no CUDA device visible; running eager", flush=True)
        return
    if key != AOTI_KEY:
        print(f"[h3-aoti] this card wants `{key}`, only `{AOTI_KEY}` is published; running eager", flush=True)
        return

    try:
        from huggingface_hub import snapshot_download
        from spaces.zero.torch.aoti import LazyAOTIModel  # noqa: F401
    except Exception as error:
        print(f"[h3-aoti] no AoTI loader here ({type(error).__name__}: {error}); running eager", flush=True)
        return

    print(f"[h3-aoti] loading {AOTI_REPO}:{key} ...", flush=True)
    try:
        local = snapshot_download(repo_id=AOTI_REPO, repo_type=AOTI_REPO_TYPE, allow_patterns=f"{key}/package/*")
    except Exception as error:
        print(f"[h3-aoti] {AOTI_REPO}:{key} unreachable ({type(error).__name__}: {error}); running eager", flush=True)
        return
    package_dir = Path(local) / key / "package"
    if not package_dir.is_dir():
        print(f"[h3-aoti] no package at `{AOTI_REPO}:{key}/package`; running eager", flush=True)
        return

    patch_blocks(transformer, package_dir)
    _LOADED.add(id(transformer))
    print(f"[h3-aoti] compiled blocks in place (temb padded to {TEMB_ROWS} rows)", flush=True)


def export_block(pipe, height: int, width: int, num_frames: int, prompt: str):
    """Capture one block call out of a real request and export it with a dynamic sequence dimension.

    Runs on the GPU, after the transformer has been quantized and moved there: a package compiled for one
    quantization mode is meaningless for another.
    """
    import torch
    import spaces

    import h3_core as h3

    transformer = h3.transformer_of(pipe)
    blocks = getattr(transformer, BLOCK_CONTAINER)

    # Keep the widest `temb` over a short real run rather than `spaces.aoti_capture`'s first call, which is the
    # 1-row one — see `TEMB_ROWS`.
    original_forward = blocks[0].forward
    widest = {"args": (), "kwargs": {}, "rows": -1}
    seen = []

    def recording(*args, **kwargs):
        rows = int(args[1].shape[0]) if len(args) > 1 and hasattr(args[1], "shape") else -1
        seen.append(rows)
        if rows > widest["rows"]:
            widest.update(args=args, kwargs=kwargs, rows=rows)
        return original_forward(*args, **kwargs)

    blocks[0].forward = recording
    try:
        pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=int(os.environ.get("H3_AOTI_CAPTURE_STEPS", "4")),
            generator=torch.Generator("cpu").manual_seed(42),
        )
    finally:
        blocks[0].forward = original_forward
    call = type("Captured", (), widest)
    if not call.args:
        raise RuntimeError("Nothing was captured — the transformer block was never called.")
    print(f"[h3-aoti] temb rows seen: {sorted(set(seen))}; exporting with {TEMB_ROWS} (padded)", flush=True)

    # `block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)`, `attention_mask` being `None` for the
    # padless sequences these pipelines build. Only the sequence is dynamic: `torch.export` specializes size-1
    # dimensions unconditionally, so a `Dim` on `temb`'s rows cannot be expressed at all.
    if AOTI_SHAPE == "dynamic":
        sequence = torch.export.Dim("sequence", min=2048, max=262144)
        dynamic_shapes = ({1: sequence}, None, {0: sequence}, ({0: sequence}, {0: sequence}), None)
        dynamic_shapes = dynamic_shapes[: len(call.args)]
    else:
        dynamic_shapes = None

    args = (call.args[0], pad_temb(call.args[1]), *call.args[2:])

    # Export the **live** block, non-strict. A shallow clone under non-strict tracing lifts every weight twice — once
    # named, once as an anonymous `CONSTANT_TENSOR` aliasing it — and the loader binds by name, so the compiled block
    # dereferences constants nobody set. The clone is only for flattening tensor-subclass parameters, which inductor's
    # constant handling cannot wrap back into a `Parameter`, and it needs `strict=True`.
    from spaces.zero.torch.aoti import _shallow_clone_module
    from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters

    subclassed = sorted({type(p).__name__ for p in blocks[0].parameters()} - {"Parameter"})
    if subclassed:
        block = _shallow_clone_module(blocks[0])
        unwrap_tensor_subclass_parameters(block)
        strict = True
        print(f"[h3-aoti] tensor-subclass parameters {subclassed}: exporting a flattened clone, strict=True", flush=True)
    else:
        block = blocks[0]
        strict = False
        print("[h3-aoti] plain parameters: exporting the live block, non-strict", flush=True)

    # `torch.export` only gives a lifted tensor a real FQN when it is a registered parameter or buffer; a plain
    # attribute becomes an anonymous constant the loader can never match. Only ever on the clone, since this
    # re-registers attributes and the live block is what the eager path runs.
    if block is not blocks[0]:
        try:
            from spaces_constant_binding_patch import register_loose_tensors

            if loose := register_loose_tensors(block):
                print(f"[h3-aoti] re-registered {len(loose)} loose tensors as buffers: {loose[:6]}", flush=True)
        except ImportError:
            pass

    print(f"[h3-aoti] exporting {type(blocks[0]).__name__}, shapes={AOTI_SHAPE}, strict={strict} ...", flush=True)
    try:
        exported = torch.export.export(block, args, call.kwargs or None, dynamic_shapes=dynamic_shapes, strict=strict)
    except Exception as error:
        if not strict:
            raise
        print(f"[h3-aoti] strict export failed ({type(error).__name__}: {error}); retrying non-strict", flush=True)
        exported = torch.export.export(block, args, call.kwargs or None, dynamic_shapes=dynamic_shapes)

    anonymous = [
        spec.target for spec in exported.graph_signature.input_specs if spec.kind.name == "CONSTANT_TENSOR"
    ]
    if anonymous:
        print(
            f"[h3-aoti] WARNING {len(anonymous)} constants lifted anonymously: {anonymous[:6]}. The loader binds by "
            f"name, so `compile_and_save` writes the alias sidecar and `patch_blocks` raises rather than segfaulting.",
            flush=True,
        )
    return exported


def compile_and_save(exported_program, destination: str | os.PathLike[str]) -> Path:
    """Inductor-compile the exported block into `<destination>/package/submodules/transformer_blocks/package.pt2`.

    That layout is what `aoti_load_from_package_dir` walks, resolving the submodule name to the transformer's
    `transformer_blocks` `ModuleList` and patching every block in it with this one package.
    """
    import spaces

    package_dir = Path(destination) / "package"
    print("[h3-aoti] inductor compile (minutes) ...", flush=True)
    spaces.aoti_compile_and_save(package_dir, exported_program, submodule=BLOCK_CONTAINER)

    # The compiled artifact drops a constant's FQN when the export lifted it anonymously; the `ExportedProgram` still
    # has the real names, so record the mapping for the loader while it is available.
    try:
        from spaces_constant_binding_patch import write_constant_aliases

        if sidecar := write_constant_aliases(package_dir, exported_program, submodule=BLOCK_CONTAINER):
            print(f"[h3-aoti] constant alias sidecar written: {sidecar.name}", flush=True)
    except ImportError:
        pass

    files = sorted(str(path.relative_to(package_dir)) for path in package_dir.rglob("*") if path.is_file())
    print(f"[h3-aoti] package written: {files}", flush=True)
    return package_dir


def upload(package_dir: str | os.PathLike[str], key: str) -> str:
    """Push the package under its `<width>/torch<X.Y>/sm<cc>/<shape>` key. CPU work — never inside GPU time."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("`HF_TOKEN` is needed to push the AoTI package.")
    api = HfApi(token=token)
    api.create_repo(repo_id=AOTI_REPO, repo_type=AOTI_REPO_TYPE, private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(package_dir),
        path_in_repo=f"{key}/package",
        repo_id=AOTI_REPO,
        repo_type=AOTI_REPO_TYPE,
        commit_message=f"AoTI package for {key}",
    )
    return f"https://huggingface.co/{'datasets/' if AOTI_REPO_TYPE == 'dataset' else ''}{AOTI_REPO}/tree/main/{key}"
