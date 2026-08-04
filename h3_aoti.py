"""ZeroGPU AoTI for MiniMax-H3: compile the repeated transformer block once, reuse the package forever.

Shared byte-identically by every MiniMax-H3 Space. A Space only ever calls `maybe_load()`; the compile path runs from
the debug Space's "Compile (AoTI)" tab, or off-Space from `job_bf16_aoti.py` on an `rtx-pro-6000` Job, and pushes its
artifacts to `multimodalart/minimax-h3-aoti` under `<width>/torch<X.Y>/sm<cc>/<shape>`.

What is measured, so nobody has to guess whether this is worth turning on. Unquantized bfloat16, 124 frames,
everything resident, one dynamic-sequence package serving every row — on an RTX PRO 6000 Blackwell, torch 2.11,
cuDNN attention:

    canvas (HxW)   eager s/step   AoTI s/step   saved     faster
    768x1344          10.20          9.73       0.47 s     +4.6%
    704x1280           8.59          7.87       0.72 s     +8.4%
    640x1152           6.46          5.88       0.59 s     +9.1%
    576x1024           4.74          4.24       0.50 s    +10.5%
    544x960            4.02          3.58       0.44 s    +11.0%

Read the *absolute* column: AoTI removes a near-constant ~0.5 s/step no matter how big the canvas is. That is exactly
the shape of what it can remove — 50 blocks' worth of kernel-launch overhead and the norm / rotary / AdaLN-gather
epilogues around the matmuls. It cannot touch the matmuls themselves, and at S = 37726 one block is ~70 TFLOP of GEMM
and attention, so the released 768x1344 canvas is compute bound and only 4.6% comes back. The smaller the default
canvas gets, the better this pays.

The trap that cost a day, recorded here because the symptom is a segfault with no message: a **shallow clone exported
in torch.export's default non-strict mode duplicates every weight** — once as a named `PARAMETER` and once as an
anonymous `lifted_tensor_<N>` `CONSTANT_TENSOR` aliasing the same storage — and `LazyAOTIModel` binds constants by
name, so the anonymous half binds to nothing and the compiled kernel reads pointers nobody set. It is neither
accelerate's offload hooks nor torchao's tensor subclasses, which were both blamed first; it reproduces in plain
bfloat16 with no subclass anywhere, and it goes away with `strict=True`. See `export_block`.

Why block level rather than the whole transformer: `MiniMaxH3Transformer3DModel.forward` decides whether the packed
sequence needs a padding attention mask with `bool(is_pad.any())`, a data-dependent branch `torch.export` cannot
trace. One `MiniMaxH3TransformerBlock` is where all the time goes anyway (50 of them per step), and the sequence
length is the only thing that changes between requests, which a single dynamic dimension covers. The block's fifth
argument, `attention_mask`, is always `None` in practice — `packing.py` never emits a padding row, so `token_tags` is
never negative — which is what makes one static signature enough.

What `spaces` 0.51.1 actually provides (checked against the installed package, not the klein-era blog post):

    spaces.aoti_capture(module)                       context manager, grabs the args of the next call and aborts it
    spaces.aoti_compile(exported_program, configs)    in-process compile, returns a ZeroGPUCompiledModel
    spaces.aoti_compile_and_save(dir, ep, configs, submodule=...)
                                                      compile and write `<dir>/submodules/<submodule>/package.pt2`
    spaces.aoti_apply(compiled, module)               in-process apply
    spaces.aoti_patch(module, LazyAOTIModel)          apply a package to one module, weights stay live
    spaces.aoti_load_from_package_dir(module, dir)    walk `<dir>/{root,submodules/*}` and patch, ModuleList aware
    spaces.aoti_load(module, repo_id, ...)            the convenience wrapper — NOT usable here: it hardcodes
                                                      `snapshot_download(allow_patterns="package/*")` on a *model*
                                                      repo, and these artifacts are keyed by quant/torch/arch under a
                                                      dataset, so the download is done here and only the loader
                                                      (`aoti_load_from_package_dir`) is reused.
    spaces.aoti_blocks_load(module, repo_id, variant) the `_repeated_blocks` convenience — same repo-layout mismatch.

Weights are *not* baked into the package: `aoti_patch` binds the block's live `state_dict()`, so one package serves all
50 blocks, and the quantized weights it reads are whatever the block holds. Which is also why quantization has to
happen *before* the export — the same ordering constraint as fusing a LoRA before AoTI.
"""

from __future__ import annotations

import os
from pathlib import Path

AOTI = os.environ.get("H3_AOTI", "0") == "1"
# A public **model** repo. It used to be a private dataset, which is why the repo type is still a variable: the
# artifacts are keyed by quant/torch/arch under `<width>/torch<X.Y>/sm<cc>/<shape>` rather than laid out the way
# `spaces.aoti_load` expects, so the download is done by hand either way (see `maybe_load`).
AOTI_REPO = os.environ.get("H3_AOTI_REPO", "multimodalart/minimax-h3-aoti")
AOTI_REPO_TYPE = os.environ.get("H3_AOTI_REPO_TYPE", "model")
# `dynamic` is the one package that serves every canvas, duration *and prompt*, and for bfloat16 it is what gets built:
# a dynamic sequence dimension exports and compiles cleanly (measured on an rtx-pro-6000 Job, torch 2.11). It has to be
# dynamic to be useful at all — `build_packed_sequence` pads nothing, so
# `S = num_text_tokens + condition_rows + audio_rows + video_rows` moves with the *prompt* as well as the canvas, and a
# static package would only ever serve the one prompt length it was captured from.
#
# A `HxWxF` value instead pins the artifact to one static shape. That is the fallback for a width whose dynamic export
# is refused, which is what the NVFP4 attempt hit: export rejected the dimension and offered only the affine
# `S = 128 * k - 34` it had derived from that one capture — an offset that is a property of one canvas *and* one prompt
# rather than of the model, so a package built that way serves almost nothing. The 128 is the alignment torchao's
# scaled matmuls want, though that has not been re-verified since the bfloat16 path was proven, and a static package is
# only worth building for a width that has been shown to need one.
AOTI_SHAPE = os.environ.get("H3_AOTI_SHAPE", "dynamic")
AOTI_DURATION = int(os.environ.get("H3_AOTI_DURATION", "1500"))

# The 50-deep stack that is the whole cost of a step. `MiniMaxH3TokenRefinerBlock` is also in `_repeated_blocks` but
# runs a handful of text rows a couple of times per step, so it is left eager.
BLOCK_CONTAINER = "transformer_blocks"

# Height of the AdaLN table baked into the package. `temb` is `(num_distinct_timesteps, time_embed_dim)` and the block
# gathers from a `3 * num_distinct_timesteps` table with `adaln_indices = timestep_indices * 3 + token_tag`, so the
# table's height is part of the compiled shape. It is not constant at runtime: at step 0 the video and audio streams
# share a noise level and `temb` has a single row, and from step 1 their sigma schedules diverge and it grows one.
# Exporting whatever the first call happened to show bakes in a 3-row table and the later steps then walk off it:
#
#     Assertion `index out of bounds: 0 <= tmp22 < 3` failed
#
# A dynamic dimension is the wrong tool — `torch.export` specializes size-1 dimensions unconditionally, so a `Dim`
# taken from a 2-row capture carries a `>= 2` guard that step 0 violates. Instead `temb` is padded to a fixed height
# on both sides of the compile. Rows past the live ones are never gathered, so the output is unchanged, and the shape
# becomes a constant. Must match the `H3_AOTI_TEMB_ROWS` the package was compiled with.
#
# 4 is what the published bfloat16 packages were built with, and the padding is *validated* rather than assumed: the
# build job replays a real 1-row (step 0) call and a real 2-row (step 1) call through the compiled block and diffs both
# against eager. Two streams at two noise levels is the realistic maximum, so 4 is loose on purpose, and the cost is
# one slightly taller AdaLN projection per block against the block's own 70 TFLOP.
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
    """Which transformer these artifacts belong to: `bf16`, `fp8`, `nvfp4`, ...

    `H3_WIDTH` wins, so a Space that has no `h3_core` — the unquantized split deployment is two standalone Spaces —
    can use this module by setting one variable. Otherwise it comes from `h3_core`, which derives it from `H3_QUANT`
    or from the pre-quantized repository's suffix.
    """
    if explicit := os.environ.get("H3_WIDTH"):
        return explicit.lower()
    try:
        import h3_core

        return h3_core.WIDTH
    except Exception:
        return "bf16"


def artifact_key() -> str:
    """`<width>/torch<X.Y>/sm<cc>/<shape>` — an AoTI package is valid for exactly one of each."""
    import torch

    torch_version = ".".join(torch.__version__.split(".")[:2])
    major, minor = torch.cuda.get_device_capability()
    return f"{width()}/torch{torch_version}/sm{major}{minor}/{AOTI_SHAPE}"


def status() -> str:
    return (
        f"AoTI **on** · `{AOTI_REPO}` ({AOTI_REPO_TYPE}) · shape `{AOTI_SHAPE}`"
        if AOTI
        else "AoTI **off** (`H3_AOTI=1` to load compiled blocks)"
    )


def patch_blocks(transformer, package_dir) -> None:
    """Point all 50 blocks at the one compiled package, binding each block's own weights on its first call.

    This is `spaces.aoti_load_from_package_dir` with two changes, both forced by how this Space runs.

    *Weights are read on the first forward, not at patch time.* `spaces.aoti_patch` snapshots `state_dict()` when it
    patches, and `maybe_load` runs at startup, while the components are still on the host — `place()` only moves them
    on the first request, because ZeroGPU cannot pack a startup-resident `Float8Tensor` (it packs CUDA tensors with
    `aten.empty_like(..., pin_memory=True)`, which the subclass does not implement). `Module.to` rebinds `param.data`
    to a fresh CUDA tensor, so a snapshot taken at startup keeps pointing at the host copies and the compiled block
    would run against host memory. Reading the state dict on the first call instead picks it up wherever it now is.

    *`temb` is padded on the way in*, to the fixed height the package was exported with — see `TEMB_ROWS`.

    The clone-and-flatten here is `spaces.aoti_patch`'s own preparation, kept because a quantized width needs it: the
    names it produces are the FQNs the package's constants were derived from. For an unquantized block it is a no-op
    and the resulting names are exactly `blocks[0].state_dict()`, which is what `export_block` exported — the two
    sides agree either way. What must *not* be mirrored is the clone on the export side under non-strict tracing; see
    `export_block` for why that is the difference between a working package and a SIGSEGV.
    """
    from spaces.zero.torch.aoti import LazyAOTIModel, _shallow_clone_module
    from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters

    # `LazyAOTIModel` binds constants by intersecting `state_dict()` with `get_constant_fqns()` and
    # silently keeps whatever it does not match, so a package whose constants were lifted anonymously
    # binds nothing and the compiled block then dereferences constants nobody set — a SIGSEGV. This patch
    # resolves those names through the `constant_aliases.json` the compile side writes, and **raises** a
    # readable error if it still cannot. Purely protective: with a well-formed package it changes nothing,
    # which is why a missing sidecar module is a warning rather than a failure.
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
    """Patch the block stack with its compiled package. Once, and safe to call at **startup**.

    Nothing here touches a GPU: the download is CPU work and the `.pt2` archive is not opened until the first forward,
    which happens inside the `@spaces.GPU` call. Proven on the pool: `bf16/torch2.11/sm120/dynamic` loads at startup,
    patches all 50 blocks, and generates. The repo is public, so no token is passed for it.
    """
    if not AOTI or id(transformer) in _LOADED:
        return

    import spaces
    from huggingface_hub import snapshot_download

    key = artifact_key()
    print(f"[h3-aoti] loading {AOTI_REPO}:{key} ...", flush=True)
    local = snapshot_download(
        repo_id=AOTI_REPO,
        repo_type=AOTI_REPO_TYPE,
        allow_patterns=f"{key}/package/*",
    )
    package_dir = Path(local) / key / "package"
    if not package_dir.is_dir():
        raise RuntimeError(
            f"No AoTI package at `{AOTI_REPO}:{key}/package`. Run the debug Space's Compile (AoTI) tab on this card "
            f"with this `H3_QUANT`, or set `H3_AOTI=0`."
        )
    patch_blocks(transformer, package_dir)
    _LOADED.add(id(transformer))
    print(f"[h3-aoti] compiled blocks in place (temb padded to {TEMB_ROWS} rows)", flush=True)


def export_block(pipe, height: int, width: int, num_frames: int, prompt: str):
    """Capture one block call out of a real request and export it with a dynamic sequence dimension.

    Runs on the GPU, after the transformer has been quantized and moved there: the export traces the quantized module,
    and a package compiled for one quantization mode is meaningless for another.
    """
    import torch
    import spaces

    import h3_core as h3

    transformer = h3.transformer_of(pipe)
    blocks = getattr(transformer, BLOCK_CONTAINER)

    # Record every call the block receives over a short real run and keep the widest `temb`, rather than
    # `spaces.aoti_capture`'s first-call-then-abort. The first call is the unrepresentative one: see `TEMB_ROWS`.
    # Text encoding and packing run either way, which is the point — these are the real inputs.
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

    # `block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)`:
    #   hidden_states   (1, S, hidden)
    #   temb            (num_distinct_timesteps, time_embed_dim)
    #   adaln_indices   (S,)
    #   rotary_emb      ((S, dim), (S, dim))
    #   attention_mask  None for a padless sequence, which is what these pipelines build
    #
    # Only the sequence is asked for. `temb`'s row count is held constant by padding instead (see `TEMB_ROWS`), which
    # is both cheaper to reason about and the only thing that works: `torch.export` specializes size-1 dimensions
    # unconditionally, so a `Dim` on a dimension that is 1 at step 0 cannot be expressed at all.
    if AOTI_SHAPE == "dynamic":
        sequence = torch.export.Dim("sequence", min=2048, max=262144)
        dynamic_shapes = ({1: sequence}, None, {0: sequence}, ({0: sequence}, {0: sequence}), None)
        dynamic_shapes = dynamic_shapes[: len(call.args)]
    else:
        dynamic_shapes = None

    # `temb` to its fixed height, so the AdaLN table the package bakes in is the one `maybe_load` will feed it.
    args = (call.args[0], pad_temb(call.args[1]), *call.args[2:])

    # WHICH MODULE, AND WHICH EXPORT MODE. This is the whole difference between a working package and a SIGSEGV.
    #
    # `spaces.aoti_patch` prepares the load side by shallow-cloning the module and flattening any tensor subclass, and
    # the received wisdom is to do the identical thing before exporting so both sides derive the same constant FQNs.
    # For a subclass that is genuinely necessary: inductor's constant handling wraps a constant back into
    # `torch.nn.Parameter`, which rejects a non-floating dtype ("Only Tensors of floating point and complex dtype can
    # require gradients"), so `Float8Tensor` / `NVFP4Tensor` parameters have to be flattened first.
    #
    # But a shallow clone exported in `torch.export`'s **default non-strict** mode duplicates every weight: the same
    # tensor comes out once as a named `PARAMETER` and again as an anonymous `lifted_tensor_<N>` `CONSTANT_TENSOR`,
    # 12 of each for this block, 1.2 GiB of them, `data_ptr()` proving the two sets alias. `LazyAOTIModel` binds by
    # name, so the anonymous half binds to nothing and the compiled block dereferences constants nobody set. That is
    # the crash that stalled this work, blamed first on accelerate's offload hooks and then on torchao's subclasses;
    # it is neither. Measured on an rtx-pro-6000 Job at full size, torch 2.11, plain bfloat16, no subclass anywhere:
    #
    #     live block,    non-strict   12 PARAMETER,  0 CONSTANT_TENSOR      <- what the shipped bf16 package used
    #     live block,    strict       12 PARAMETER,  0 CONSTANT_TENSOR
    #     shallow clone, non-strict   12 PARAMETER, 12 CONSTANT_TENSOR      <- the bug
    #     shallow clone, strict       12 PARAMETER,  0 CONSTANT_TENSOR
    #
    # So: export the **live block** whenever it has no subclass parameters to flatten, which is every unquantized
    # width and, per the fp8 investigation, quantized ones whose weights are still registered parameters. Only fall
    # back to the clone when flattening is actually needed, and then in `strict` mode, which is also clean. The clone
    # is safe to skip for the live-block path precisely because there is nothing to unwrap: `state_dict()` names are
    # then identical on both sides by construction.
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

    # `torch.export` only gives a lifted tensor a real FQN when it is a registered parameter or buffer; a tensor
    # reached through a plain attribute becomes an anonymous constant the loader can never match against
    # `state_dict()`. Re-registering such tensors as buffers is numerics-preserving — it changes how a tensor is
    # registered, never the tensor and never the forward. A well-formed block has none, and this returns empty.
    # Only ever on the clone: `register_loose_tensors` *re-registers* attributes, so running it on the live block would
    # mutate the model the eager path uses. A `MiniMaxH3TransformerBlock` has no loose tensor attributes, so this is
    # empty in practice and the live-block export needs nothing; if that ever changes, the warning below catches it.
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
        # Dynamo refuses some modules it cannot trace. Non-strict is still worth attempting, with the duplication
        # reported loudly below rather than left to segfault at load time.
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
            f"name, so this package will not bind them; `compile_and_save` writes the alias sidecar and "
            f"`patch_blocks` raises rather than letting it segfault.",
            flush=True,
        )
    return exported


def compile_and_save(exported_program, destination: str | os.PathLike[str]) -> Path:
    """Inductor-compile the exported block into `<destination>/package/submodules/transformer_blocks/package.pt2`.

    That layout is what `aoti_load_from_package_dir` walks: it resolves the submodule name to the transformer's
    `transformer_blocks` `ModuleList` and, because a `ModuleList` is iterable, patches every one of the 50 blocks with
    this single package.
    """
    import spaces

    package_dir = Path(destination) / "package"
    print("[h3-aoti] inductor compile (minutes) ...", flush=True)
    spaces.aoti_compile_and_save(package_dir, exported_program, submodule=BLOCK_CONTAINER)

    # The compiled artifact keeps a constant's dtype, shape and slot index but drops its FQN when the
    # export lifted it anonymously. The `ExportedProgram` still has the real names, so record the
    # mapping next to the package while it is still available; the loader reads it back.
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
    """Push the package under its `<quant>/torch<X.Y>/sm<cc>/<shape>` key. CPU work — never inside GPU time."""
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
