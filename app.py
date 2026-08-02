"""MiniMax-H3, split deployment — **the denoising half**.

This Space holds the transformer and the two autoencoders, **unquantized bfloat16**, and nothing else. The 62.14 GiB
Qwen3-VL conditioner lives in its own Space,
[`minimax-h3-conditioner`](https://huggingface.co/spaces/diffusers-internal-dev/minimax-h3-conditioner), which this
one calls over the gradio API for every request; what comes back is a safetensors file holding the two tensors the
denoiser needs, `prompt_embeds` and `text_token_tags`.

Why split at all: MiniMax-H3 is 195.9 GiB in bfloat16 and a ZeroGPU Space is evicted at 150 GB of storage, so an
unquantized single Space is impossible — the existing demos run NVFP4 or float8 weights for that reason alone. Cut at
the text-encoder step, this half pulls 77.3 GB (`transformer/` 61.73 GiB + `vae/` 9.70 + `audio_vae/` 0.56) and the
other 66.7 GB, and neither is quantized.

The blockset is `MiniMaxH3Blocks` with its `text_encoder` step removed — see `h3_split_blocks.py`. Dropping the step
drops the three components it declares, so `load_components` never fetches the conditioner, and `prompt_embeds` /
`text_token_tags` become ordinary required inputs of the pipeline call.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback

# First, and at module level. `import spaces` patches `torch.cuda` before any GPU is attached, which is what lets the
# 82 GiB load happen at **startup** rather than on GPU time; it also has to precede anything that initializes CUDA.
import spaces
import gradio as gr

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "diffusers-internal-dev/MiniMax-H3")
CONDITIONER_SPACE = os.environ.get("H3_CONDITIONER", "diffusers-internal-dev/minimax-h3-conditioner")
# `preplace` moves the 61.73 GiB transformer onto the card at startup and leaves the 10.43 GiB of float32
# autoencoders for the first GPU call; `offload` hands the whole thing to
# `ComponentsManager.enable_auto_cpu_offload` instead. See `_drop_cached_weight_blobs` for why the *transformer
# alone* rather than everything: the 150 GB storage quota, not the 95 GiB card, is the binding constraint at startup.
PLACEMENT = os.environ.get("H3_PLACEMENT", "preplace").lower()
# cuDNN's fused attention is 10-20% faster than the SDPA default on this pool and needs nothing installed.
# flash-attention 3 is sm90-only and this card is sm120 (the `zero-a10g` flavour name is legacy).
ATTENTION = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
GPU_DURATION = int(os.environ.get("H3_GPU_DURATION", "900"))
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")
ON_SPACES = bool(os.environ.get("SPACE_ID"))

CANVASES = {
    "16:9 (768x1344)": (768, 1344),
    "9:16 (1344x768)": (1344, 768),
    "1:1 (768x768)": (768, 768),
    "4:3 (768x1024)": (768, 1024),
    "3:4 (1024x768)": (1024, 768),
    "21:9 (672x1536)": (672, 1536),
}
DEFAULT_CANVAS = "16:9 (768x1344)"
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5
MAX_UI_DURATION = 14


def snap_frames(seconds: float) -> int:
    """The frame count MiniMax-H3's video VAE can decode: the next `17 * n + 5` at 24 fps."""
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


PIPE = None
MANAGER = None
LOAD_ERROR: str | None = None
LOADED_IN: float | None = None
CLIENT = None


def status() -> str:
    if LOAD_ERROR:
        return LOAD_ERROR
    if PIPE is None:
        return f"Loading `{MODEL_REPO}` (transformer + VAEs, 77.3 GB). Watch the Space logs."
    return (
        f"Ready · transformer + VAEs **bfloat16, unquantized** · placement `{PLACEMENT}` · attention `{ATTENTION}` · "
        f"loaded in {LOADED_IN:.0f}s · conditioner `{CONDITIONER_SPACE}`"
    )


def load_models() -> str | None:
    """Load the denoising half. At **startup**.

    `MiniMaxH3GeneratorBlocks` declares `transformer`, `vae`, `audio_vae`, `scheduler`, `audio_scheduler` and
    `video_processor`, so `load_components` fetches exactly those subfolders out of the shared
    `modular_model_index.json` — `text_encoder/` and `transformer_ref/` are never touched.

    Both autoencoders carry `_keep_in_fp32_modules` over every module, so the `dtype` below is refused for them and
    they load float32 (~20.5 GiB rather than 10.26): a bfloat16 audio VAE decodes the soundtrack ~20 dB too quiet.
    """
    global PIPE, MANAGER, LOAD_ERROR, LOADED_IN

    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    token = os.environ.get("HF_TOKEN")
    if not token:
        LOAD_ERROR = f"**`HF_TOKEN` secret is missing** and `{MODEL_REPO}` is private. Add it and restart."
        return LOAD_ERROR

    started = time.time()
    try:
        import torch
        from diffusers import ComponentsManager

        from h3_split_blocks import MiniMaxH3GeneratorBlocks

        manager = ComponentsManager()
        blocks = MiniMaxH3GeneratorBlocks()
        print(f"[gen] loading {[c.name for c in blocks.expected_components]} from {MODEL_REPO} ...", flush=True)
        pipe = blocks.init_pipeline(MODEL_REPO, components_manager=manager, collection="h3")
        pipe.load_components(dtype=torch.bfloat16, token=token)
        pipe.transformer.set_attention_backend(ATTENTION)

        if PLACEMENT == "preplace":
            # Plain bfloat16 tensors, so ZeroGPU's startup packing handles them — the thing that cannot be moved at
            # startup is a torchao `Float8Tensor`, whose `aten.empty_like(pin_memory=True)` is unimplemented. The
            # autoencoders deliberately stay behind; `_generate` moves them on the first request.
            pipe.transformer.to("cuda")
            _log_disk(f"freed {_drop_cached_weight_blobs():.1f} GiB of downloaded weights")
        else:
            manager.enable_auto_cpu_offload(device="cuda")
            _arm_decode_hooks(pipe)

        PIPE, MANAGER = pipe, manager
        LOADED_IN = time.time() - started
        print(f"[gen] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = f"**Loading `{MODEL_REPO}` failed** after {time.time() - started:.0f}s: `{type(error).__name__}: {error}`"
    return LOAD_ERROR


def _log_disk(label: str) -> None:
    import shutil

    from huggingface_hub.constants import HF_HUB_CACHE

    probe = HF_HUB_CACHE if os.path.isdir(HF_HUB_CACHE) else os.path.expanduser("~")
    _total, used, free = shutil.disk_usage(probe)
    print(f"[gen] {label} · disk {used / 2**30:.1f} used, {free / 2**30:.1f} free", flush=True)


def _drop_cached_weight_blobs() -> float:
    """Delete the downloaded weight files once their tensors are copies in (fake-)CUDA memory.

    A ZeroGPU Space is evicted at **150 GB of storage**, and `spaces`' startup `torch.pack()` writes every
    startup-resident CUDA tensor to a *second* copy on disk before deleting the originals itself. This half
    downloads 77.3 GB, so packing all of it needs 154.6 GB and the Space dies with `OSError: [Errno 28] No space
    left on device` in `os.posix_fallocate` — which is exactly how its first bring-up died.

    Two things keep it under the cap. Only the transformer is moved onto the card, so the pack is 66.3 GB rather
    than 77.3 GB; and its shards are dropped here, before `launch()`, rather than after the pack. `.to("cuda")`
    copies each parameter out of the memory-mapped shard, so those blobs are unreferenced by then and unlinking
    them frees the space for real. The autoencoders are still mapped, so their ~11 GB survives the unlink — which
    is fine, 66.3 + 11 is nowhere near the cap, and it is the arithmetic this Space is designed around:

        worst case, nothing frees   77.3 (download) + 66.3 (pack) = 143.6 GB
        expected                    11 (still-mapped VAEs) + 66.3 (pack) = 77.3 GB
    """
    import glob

    from huggingface_hub.constants import HF_HUB_CACHE

    blobs = os.path.join(HF_HUB_CACHE, "models--" + MODEL_REPO.replace("/", "--"), "blobs", "*")
    freed = 0
    for blob in glob.glob(blobs):
        size = os.path.getsize(blob)
        os.unlink(blob)
        freed += size
    return freed / 2**30


def _arm_decode_hooks(pipe):
    """Make the offload hooks fire for the two VAEs.

    `enable_auto_cpu_offload` installs accelerate hooks, which wrap `forward`. The decode blocks call
    `components.vae.decode(...)` and `components.audio_vae.decode(...)` directly, so the hook never runs and the VAE
    is still on the host when the latents arrive on the card.
    """
    for name in ("vae", "audio_vae"):
        module = getattr(pipe, name)
        inner = module.decode

        def armed(*args, _module=module, _decode=inner, **kwargs):
            hook = getattr(_module, "_hf_hook", None)
            if hook is not None:
                hook.pre_forward(_module)
            return _decode(*args, **kwargs)

        module.decode = armed


def conditioner():
    """The other half, over the gradio API. Cached — building a `Client` costs a round trip to the Space config."""
    global CLIENT
    if CLIENT is None:
        from gradio_client import Client

        CLIENT = Client(CONDITIONER_SPACE, token=os.environ.get("HF_TOKEN"))
    return CLIENT


def encode_remote(prompt, image_path, last_image_path, canvas, num_frames):
    """Ask the conditioner Space for `prompt_embeds` + `text_token_tags`. Off this Space's GPU time entirely."""
    from gradio_client import handle_file
    from safetensors import safe_open

    path, plan = conditioner().predict(
        prompt=prompt,
        image_path=handle_file(image_path) if image_path else None,
        last_image_path=handle_file(last_image_path) if last_image_path else None,
        canvas=canvas,
        num_frames=num_frames,
        api_name="/encode",
    )
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        return handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags"), metadata, plan


@spaces.GPU(duration=GPU_DURATION, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed):
    """The only thing on GPU time: the packed-sequence denoise loop and the two decoders.

    Only the three generated outputs come back. A `@spaces.GPU` return crosses a process boundary by pickling, and
    the full `PipelineState` still holds the packed latents, the rotary grid and the row indices on the card.
    """
    import torch

    if PLACEMENT == "preplace":
        # The two autoencoders were left on the host so that ZeroGPU's startup pack stayed inside the storage quota;
        # 10.43 GiB across PCIe on the first request, and a no-op walk on every one after it.
        PIPE.vae.to("cuda")
        PIPE.audio_vae.to("cuda")

    state = PIPE(
        prompt_embeds=prompt_embeds.to("cuda"),
        text_token_tags=text_token_tags,
        image=image,
        last_image=last_image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=int(steps),
        generator=torch.Generator("cpu").manual_seed(int(seed)),
    )
    return state.get("videos")[0], state.get("audio")[0].cpu(), state.get("sampling_rate")


def generate(prompt, image_path, last_image_path, canvas, duration, steps, seed, progress=gr.Progress()):
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR)
    if PIPE is None:
        raise gr.Error("The denoiser is still loading.")
    if not prompt or not prompt.strip():
        raise gr.Error("MiniMax-H3 always takes a prompt, keyframes or not.")

    from PIL import Image

    from diffusers.utils import encode_video

    num_frames = snap_frames(duration)

    progress(0.0, desc=f"Conditioning on {CONDITIONER_SPACE} ...")
    conditioned = time.time()
    prompt_embeds, text_token_tags, metadata, plan = encode_remote(
        prompt, image_path, last_image_path, canvas, num_frames
    )
    condition_seconds = time.time() - conditioned
    height, width, num_frames = (int(metadata[key]) for key in ("height", "width", "num_frames"))

    progress(0.1, desc=f"Denoising {steps} steps at {width}x{height}, {num_frames} frames ...")
    started = time.time()
    frames, audio, sampling_rate = _generate(
        prompt_embeds,
        text_token_tags,
        Image.open(image_path) if image_path else None,
        Image.open(last_image_path) if last_image_path else None,
        height,
        width,
        num_frames,
        steps,
        seed,
    )
    generate_seconds = time.time() - started

    directory = os.path.join(tempfile.gettempdir(), "h3-outputs")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"h3-{int(time.time() * 1000)}.mp4")
    encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)

    report = (
        f"`{width}x{height}`, {num_frames} frames ({num_frames / FPS:.3f} s), {int(steps)} steps · "
        f"conditioner {condition_seconds:.0f}s ({plan['num_text_tokens']} tokens) · "
        f"denoise + decode {generate_seconds:.0f}s ({generate_seconds / int(steps):.1f} s/step) · seed {int(seed)}"
    )
    print(f"[gen] {report}", flush=True)
    return path, report


load_models()

INTRO = """# MiniMax-H3 — unquantized, split across two Spaces

Joint video **and** soundtrack out of one denoising pass, at **bfloat16, no quantization anywhere**.

MiniMax-H3 is 195.9 GiB in bfloat16 and a ZeroGPU Space is evicted at 150 GB of storage, so the unquantized
checkpoint does not fit in one Space. It does fit in two: the 62.14 GiB Qwen3-VL conditioner runs in
[`minimax-h3-conditioner`](https://huggingface.co/spaces/diffusers-internal-dev/minimax-h3-conditioner) and this
Space holds the 61.73 GiB transformer plus the two autoencoders. Every request calls the conditioner over the gradio
API and gets back `prompt_embeds` `(1, num_text_tokens, 5120)` and `text_token_tags` `(num_text_tokens,)` — the whole
wire format of the split.

Fixed by the checkpoint: 24 fps, a 768 pixel short edge, 5–15 s, no CFG and no negative prompt.
"""


with gr.Blocks(title="MiniMax-H3 (split, bf16)") as demo:
    gr.Markdown(INTRO)
    banner = gr.Markdown(status())

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
            )
            with gr.Row():
                image = gr.Image(label="First keyframe (optional)", type="filepath")
                last_image = gr.Image(label="Last keyframe (optional)", type="filepath")
            canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES), value=DEFAULT_CANVAS)
            duration = gr.Slider(label="Duration (s)", minimum=5, maximum=MAX_UI_DURATION, step=1, value=5)
            steps = gr.Slider(label="Steps", minimum=10, maximum=40, step=1, value=30)
            seed = gr.Number(label="Seed", value=42, precision=0)
            run = gr.Button("Generate", variant="primary")
        with gr.Column():
            video = gr.Video(label="Video + soundtrack")
            report = gr.Markdown()

    run.click(
        generate,
        [prompt, image, last_image, canvas, duration, steps, seed],
        [video, report],
        api_name="generate",
    )
    demo.load(status, None, banner, api_name="status")


if __name__ == "__main__":
    demo.queue(max_size=4).launch(show_error=True)
