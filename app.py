"""MiniMax-H3 `t2va` / `fl2va`, split deployment — the denoising half."""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from functools import cache

# Before anything that could initialize CUDA: `import spaces` patches `torch.cuda` so the 72 GiB load can happen at
# startup rather than on GPU time.
import spaces
from fastapi.responses import HTMLResponse
from gradio import Server
from gradio.data_classes import FileData

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
CONDITIONER_SPACE = os.environ.get("H3_CONDITIONER", "multimodalart/qwen3vl-conditioner")
# `pack` places the transformer at startup, `lazy` moves everything on the first GPU call, `offload` hands placement to
# `ComponentsManager.enable_auto_cpu_offload`.
PLACEMENT = os.environ.get("H3_PLACEMENT", "pack").lower()
# cuDNN's fused attention is 10-20% faster than the SDPA default on this pool and needs nothing installed.
ATTENTION = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")

# Must stay identical to the conditioner's table: the *label* goes over the wire, so a canvas that half does not know
# is rejected there and surfaces as a failure here.
CANVASES = {
    # 16:9
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    # 9:16
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    # 1:1
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    # 4:3 / 3:4
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    # 21:9
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"
FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5
# It is the *snapped* frame count the ceiling has to hold for: 15 s is 360 frames, which rounds up to 362, i.e.
# 15.083 s, and is refused.
MIN_UI_DURATION, MAX_UI_DURATION = 2, 14


def snap_frames(seconds: float) -> int:
    """The frame count MiniMax-H3's video VAE can decode: the next `17 * n + 5` at 24 fps."""
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def lower_duration_floor(seconds: float = MIN_UI_DURATION) -> None:
    """Let the pipeline generate below its 5 s floor. 56 frames (2.33 s) is fine on the released checkpoint."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    MiniMaxH3ModularPipeline.min_duration = property(lambda self: float(seconds))


PIPE = None
MANAGER = None
LOAD_ERROR: str | None = None
LOADED_IN: float | None = None
LORA_STATUS: str | None = None


def status() -> str:
    if LOAD_ERROR:
        return LOAD_ERROR
    if PIPE is None:
        return f"Loading `{MODEL_REPO}` (transformer + VAEs, 77.3 GB). Watch the Space logs."
    import h3_aoti

    return (
        f"Ready · transformer + VAEs **bfloat16, unquantized** · placement `{PLACEMENT}` · attention `{ATTENTION}` · "
        f"{h3_aoti.status()} · {LORA_STATUS or 'no LoRA'} · loaded in {LOADED_IN:.0f}s · "
        f"conditioner `{CONDITIONER_SPACE}`"
    )


def load_models() -> str | None:
    """Load the denoising half at startup.

    `MiniMaxH3GeneratorBlocks` declares `transformer`, `vae`, `audio_vae`, the two schedulers and `video_processor`,
    so `load_components` fetches exactly those subfolders — `text_encoder/` and `transformer_ref/` are never touched.
    Both autoencoders carry `_keep_in_fp32_modules` over every module and stay float32: a bfloat16 audio VAE decodes
    the soundtrack roughly 20 dB too quiet.
    """
    global PIPE, MANAGER, LOAD_ERROR, LOADED_IN, LORA_STATUS

    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    started = time.time()
    try:
        import torch
        from diffusers import ComponentsManager

        from h3_split_blocks import MiniMaxH3GeneratorBlocks

        lower_duration_floor()
        manager = ComponentsManager()
        blocks = MiniMaxH3GeneratorBlocks()
        print(f"[gen] loading {[c.name for c in blocks.expected_components]} from {MODEL_REPO} ...", flush=True)
        pipe = blocks.init_pipeline(MODEL_REPO, components_manager=manager, collection="h3")
        pipe.load_components(dtype=torch.bfloat16)

        # Fold the 4-step Turbo LoRA into the bf16 weights before AoTI packages the blocks, so the compiled forward
        # reads weights that already carry the update. `H3_LORA=off` disables.
        import h3_lora

        LORA_STATUS = h3_lora.apply_lora(pipe.transformer)
        if LORA_STATUS:
            print(f"[gen] {LORA_STATUS}", flush=True)

        pipe.transformer.set_attention_backend(ATTENTION)

        # Still startup, still free: an AoTI package carries no weights and opens its archive lazily inside the GPU
        # worker. Off unless `H3_AOTI=1`.
        import h3_aoti

        h3_aoti.maybe_load(pipe.transformer)

        if PLACEMENT == "pack":
            # Scoped to the transformer. `spaces` packs every startup-resident CUDA tensor into a second on-disk copy,
            # and packing all 77.3 GB busts the 150 GB storage quota; the 61.7 GB transformer alone fits. The ~10 GB of
            # fp32 VAEs move on the first GPU call instead.
            pipe.transformer.to("cuda")

        if PLACEMENT == "offload":
            manager.enable_auto_cpu_offload(device="cuda")
            _arm_decode_hooks(pipe)

        PIPE, MANAGER = pipe, manager
        LOADED_IN = time.time() - started
        print(f"[gen] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = f"**Loading `{MODEL_REPO}` failed** after {time.time() - started:.0f}s: `{type(error).__name__}: {error}`"
    return LOAD_ERROR


def _arm_decode_hooks(pipe):
    """Make the offload hooks fire for the two VAEs.

    `enable_auto_cpu_offload` wraps `forward`, and the decode blocks call `vae.decode(...)` directly, so the hook
    never runs and the VAE is still on the host when the latents arrive on the card.
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


@cache
def conditioner():
    """The other half, over the gradio API. `gradio_client` attaches the caller's own ZeroGPU token per call, so the
    conditioner's booking is billed to whoever asked for the video."""
    from gradio_client import Client

    return Client(CONDITIONER_SPACE)


def encode_remote(prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=False):
    """`/encode` on the conditioner Space: a safetensors file holding `prompt_embeds` + `text_token_tags`, with the
    resolved `height` / `width` / `num_frames` in its metadata, plus the plan. `canvas` is the label."""
    from gradio_client import handle_file
    from safetensors import safe_open

    path, plan = conditioner().predict(
        prompt=prompt,
        image_path=handle_file(image_path) if image_path else None,
        last_image_path=handle_file(last_image_path) if last_image_path else None,
        canvas=canvas,
        num_frames=num_frames,
        rewrite_prompt=bool(rewrite_prompt),
        api_name="/encode",
    )
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        return handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags"), metadata, plan


# Seconds of GPU one request needs, from the packed video rows it is about to denoise: linear in the rows for the
# matmuls, quadratic for the attention, against the AoTI block package this Space runs.
_DUR_B, _DUR_C = 1.1745e-4, 3.8396e-9
# The two resident decoders and the mux, which scale with the output rather than with the step count.
_DECODE_BASE, _DECODE_PER_DEFAULT_CANVAS, _DEFAULT_CANVAS_PIXELS = 15, 15, 960 * 544 * 124
# `pack` mode: only the ~10 GB of VAEs move on a cold worker.
_PLACEMENT_ALLOWANCE, _PAD = 12, 10


def get_duration(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed, *a, **k):
    height, width, num_frames, steps = int(height), int(width), int(num_frames), int(steps)
    latent_frames = (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    patches = (height // 32) * (width // 32)
    rows = latent_frames * patches + (int(image is not None) + int(last_image is not None)) * patches
    denoise = steps * (_DUR_B * rows + _DUR_C * rows**2)
    decode = _DECODE_BASE + _DECODE_PER_DEFAULT_CANVAS * (height * width * num_frames) / _DEFAULT_CANVAS_PIXELS
    return max(60, int(denoise + decode) + _PLACEMENT_ALLOWANCE + _PAD)


@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed):
    """The only thing on GPU time: the packed-sequence denoise loop and the two decoders.

    Only the three generated outputs come back — a `@spaces.GPU` return crosses a process boundary by pickling, and
    the full `PipelineState` still holds the packed latents, the rotary grid and the row indices on the card.
    """
    import torch

    if PLACEMENT == "lazy":
        PIPE.to("cuda")
    elif PLACEMENT == "pack":
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


def _fit_keyframe(image_path, current_canvas):
    """Cover-crop an uploaded keyframe to the closest supported aspect ratio and pick that ratio's smallest
    (fastest) canvas, unless the caller already picked a matching ratio. Returns `(image_path, canvas_label)`."""
    from PIL import Image as _Image

    img = _Image.open(image_path)
    aspect = img.width / img.height
    fastest = {}
    for label, (h, w) in CANVASES.items():
        r = w / h
        if r not in fastest or w * h < fastest[r][1][0] * fastest[r][1][1]:
            fastest[r] = (label, (h, w))
    ratio = min(fastest, key=lambda r: abs(r - aspect))
    label, (h, w) = fastest[ratio]

    cur_h, cur_w = CANVASES[current_canvas]
    if abs(cur_w / cur_h - aspect) <= abs(ratio - aspect):
        label = current_canvas
        h, w = cur_h, cur_w

    target = w / h
    if abs(img.width / img.height - target) > 1e-3:
        if img.width / img.height > target:
            new_w = int(img.height * target)
            left = (img.width - new_w) // 2
            img = img.crop((left, 0, left + new_w, img.height))
        else:
            new_h = int(img.width / target)
            top = (img.height - new_h) // 2
            img = img.crop((0, top, img.width, top + new_h))
        img.save(image_path)
    return image_path, label


def generate(prompt, image_path=None, last_image_path=None, canvas=DEFAULT_CANVAS, duration=5, steps=4, seed=42, upsample=False):
    """One request. `upsample` is last and defaults off, so a positional API client that predates it is unaffected."""
    if LOAD_ERROR:
        raise Exception(LOAD_ERROR)
    if PIPE is None:
        raise Exception("The denoiser is still loading.")
    if not prompt or not prompt.strip():
        raise Exception("MiniMax-H3 always takes a prompt, keyframes or not.")

    from PIL import Image, ImageOps

    from diffusers.utils import encode_video

    # Server mode: keyframes arrive as FileData dicts, and the cover-crop / canvas-fit that used to be an upload
    # event in the Blocks UI runs here instead, so API callers get the same treatment.
    first = image_path["path"] if isinstance(image_path, dict) else image_path
    last = last_image_path["path"] if isinstance(last_image_path, dict) else last_image_path
    if first:
        first, canvas = _fit_keyframe(first, canvas)
    if last:
        last, canvas = _fit_keyframe(last, canvas)

    num_frames = snap_frames(duration)

    conditioned = time.time()
    prompt_embeds, text_token_tags, metadata, plan = encode_remote(
        prompt, first, last, canvas, num_frames, rewrite_prompt=upsample
    )
    condition_seconds = time.time() - conditioned
    height, width, num_frames = (int(metadata[key]) for key in ("height", "width", "num_frames"))
    refined = plan.get("refined_prompt") or ""

    def keyframe(path):
        # The conditioning latents encoded here have to be of the image the conditioner looked at, which it prepares
        # exactly this way.
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path else None

    started = time.time()
    frames, audio, sampling_rate = _generate(
        prompt_embeds,
        text_token_tags,
        keyframe(first),
        keyframe(last),
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
        f"{width}x{height} · {num_frames} frames ({num_frames / FPS:.3f} s) · {int(steps)} steps · "
        f"conditioner {condition_seconds:.0f}s ({plan['num_text_tokens']} tokens"
        f"{', upsampled' if refined else ''}) · "
        f"denoise + decode {generate_seconds:.0f}s ({generate_seconds / int(steps):.1f} s/step) · seed {int(seed)}"
    )
    print(f"[gen] {report}", flush=True)
    return FileData(path=path), report, refined



# ======================================================================
# Server mode: Gradio's API engine (queue, SSE, concurrency, ZeroGPU,
# gradio_client) under a fully custom studio frontend (index.html).
# ======================================================================
app = Server(title="MiniMax-H3 Studio")


@app.api(name="generate")
def _generate_api(prompt: str, image_path: FileData | None = None, last_image_path: FileData | None = None,
                  canvas: str = DEFAULT_CANVAS, duration: float = 5, steps: int = 4, seed: float = 42,
                  upsample: bool = False) -> tuple[FileData, str, str]:
    """Generate a video with a synchronized soundtrack. Returns (video, report, refined prompt)."""
    return generate(prompt, image_path, last_image_path, canvas, duration, steps, seed, upsample)


@app.get("/status")
def studio_status():
    """Polled by the frontend: is the denoiser ready, and the human-readable status line."""
    return {"ready": PIPE is not None and LOAD_ERROR is None, "status": status()}


@app.get("/config")
def studio_config():
    """The canvas table and slider ranges, so the frontend never hardcodes a label the backend would reject."""
    return {
        "canvases": list(CANVASES),
        "default_canvas": DEFAULT_CANVAS,
        "min_duration": MIN_UI_DURATION,
        "max_duration": MAX_UI_DURATION,
    }


@app.get("/", response_class=HTMLResponse)
def homepage():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8") as f:
        return f.read()


load_models()

if __name__ == "__main__":
    app.launch(show_error=True)
