"""MiniMax-H3, split deployment — denoising
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback

# First, and at module level. `import spaces` patches `torch.cuda` before any GPU is attached, which is what lets the
# 72 GiB load happen at **startup** rather than on GPU time; it also has to precede anything that initializes CUDA.
import spaces
import gradio as gr

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "diffusers-internal-dev/MiniMax-H3")
CONDITIONER_SPACE = os.environ.get("H3_CONDITIONER", "multimodalart/qwen3vl-conditioner")
# `lazy` moves all 72.16 GiB onto the card on the first GPU call and leaves it there; `offload` hands placement to
# `ComponentsManager.enable_auto_cpu_offload` instead. Neither puts anything on the card at *startup*, which is
# deliberate — see `load_models`: the 150 GB storage quota, not the 95 GiB card, is what rules that out here.
PLACEMENT = os.environ.get("H3_PLACEMENT", "pack").lower()
# cuDNN's fused attention is 10-20% faster than the SDPA default on this pool and needs nothing installed.
ATTENTION = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
GPU_DURATION = int(os.environ.get("H3_GPU_DURATION", "900"))
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")
ON_SPACES = bool(os.environ.get("SPACE_ID"))

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
    import h3_aoti

    return (
        f"Ready · transformer + VAEs **bfloat16, unquantized** · placement `{PLACEMENT}` · attention `{ATTENTION}` · "
        f"{h3_aoti.status()} · loaded in {LOADED_IN:.0f}s · conditioner `{CONDITIONER_SPACE}`"
    )


def load_models() -> str | None:
    """Load the denoising half. At **startup**, but *not* onto the card.

    `MiniMaxH3GeneratorBlocks` declares `transformer`, `vae`, `audio_vae`, `scheduler`, `audio_scheduler` and
    `video_processor`, so `load_components` fetches exactly those subfolders out of the shared
    `modular_model_index.json` — `text_encoder/` and `transformer_ref/` are never touched.

    Both autoencoders carry `_keep_in_fp32_modules` over every module, so the `dtype` below is refused for them and
    they stay float32: a bfloat16 audio VAE decodes the soundtrack roughly 20 dB too quiet.

    Nothing is moved onto the card here, which is the one place this Space departs from the ZeroGPU idiom, and the
    reason is storage rather than memory. `spaces`' startup `torch.pack()` writes every startup-resident CUDA tensor
    to a **second copy on disk** and only deletes the downloaded originals afterwards; 77.3 GB of weights plus a
    77.3 GB pack is 154.6 GB against a 150 GB quota, and the Space is evicted mid-pack with `OSError: [Errno 28] No
    space left on device` out of `os.posix_fallocate`. Deleting the shards first does not help either: the pack's own
    cleanup walks the still-open mappings and `lstat`s them, so an unlinked blob turns into `FileNotFoundError:
    ... (deleted)`. Placement therefore happens on the first GPU call, where it costs about 10 s of PCIe and then
    persists across every later request in the same worker.
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

        # Still startup, still free: an AoTI package carries no weights and opens its compiled archive lazily inside
        # the GPU worker, so pointing the 50-block stack at it is CPU work. Off unless `H3_AOTI=1`.
        import h3_aoti

        h3_aoti.maybe_load(pipe.transformer)

        if PLACEMENT == "pack":
            # Idiomatic ZeroGPU startup placement, scoped to the transformer only. `spaces` packs every
            # startup-resident CUDA tensor into a second on-disk copy; packing all 77.3 GB (transformer + fp32
            # VAEs) busts the 150 GB storage quota (77.3 + 77.3 + shards), but the 61.7 GB transformer alone
            # packs to ~123 GB total and fits. The VAEs (~10 GB) take the lazy path on first GPU call, ~2 s.
            # With AoTI the packed transformer pairs with the precompiled blocks: no placement, no compile,
            # first request runs at steady state.
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

        CLIENT = Client(CONDITIONER_SPACE)  # public Space, no org token: the request runs on the caller side quota
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




# Fitted on live probes (5 configs spanning canvas, duration and steps; max residual 3.7 s):
# gpu_seconds = A + B * steps * tokens + C * steps * tokens^2, where tokens is the packed video row count.
# PLACEMENT_ALLOWANCE covers the one-time 72 GiB lazy .to("cuda") a cold worker pays inside its first call.
_DUR_A, _DUR_B, _DUR_C = -6.023, 2.0877e-4, 2.1221e-9
_PLACEMENT_ALLOWANCE, _PAD = 12, 10  # pack mode: only the ~10 GB VAEs move on a cold worker


def get_duration(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed, *a, **k):
    latent_frames = (int(num_frames) - 5) // 17 * 5 + 2
    patches = (int(height) // 32) * (int(width) // 32)
    tokens = latent_frames * patches
    tokens += (int(image is not None) + int(last_image is not None)) * patches
    st = int(steps) * tokens
    compute = _DUR_A + _DUR_B * st + _DUR_C * st * tokens
    return max(60, int(compute) + _PLACEMENT_ALLOWANCE + _PAD)


@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, steps, seed):
    """The only thing on GPU time: the packed-sequence denoise loop and the two decoders.

    Only the three generated outputs come back. A `@spaces.GPU` return crosses a process boundary by pickling, and
    the full `PipelineState` still holds the packed latents, the rotary grid and the row indices on the card.
    """
    import torch

    if PLACEMENT == "lazy":
        # 72.16 GiB across PCIe on the first request of a worker, a no-op walk on every one after it.
        PIPE.to("cuda")
    elif PLACEMENT == "pack":
        # Transformer was packed at startup; only the ~10 GB of fp32 VAEs walk across on a cold worker.
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


def generate(prompt, image_path=None, last_image_path=None, canvas=DEFAULT_CANVAS, duration=5, steps=28, seed=42, progress=gr.Progress(track_tqdm=True)):
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



def _fit_keyframe(image_path, current_canvas):
    """Cover-crop an uploaded keyframe to the closest supported aspect ratio and select that ratio's
    smallest (fastest) canvas, unless the user already picked a matching ratio."""
    if not image_path:
        return gr.update(), gr.update()
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
    if abs(img.width / img.height - target) <= 1e-3:
        return gr.update(), gr.update(value=label)
    if True:
        if img.width / img.height > target:
            new_w = int(img.height * target)
            left = (img.width - new_w) // 2
            img = img.crop((left, 0, left + new_w, img.height))
        else:
            new_h = int(img.width / target)
            top = (img.height - new_h) // 2
            img = img.crop((0, top, img.width, top + new_h))
        img.save(image_path)
    return gr.update(value=image_path), gr.update(value=label)


load_models()

INTRO = """# MiniMax-H3

<div align="center">
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3"><strong>[ model ]</strong></a> &nbsp;
  <a href="PAPER_URL_PLACEHOLDER"><strong>[ paper ]</strong></a> &nbsp;
  <a href="https://www.minimax.io"><strong>[ project ]</strong></a>
</div>

**MiniMax-H3** is a 33B parameter state of the art video generation model that produces video and a
fully synchronized soundtrack (ambience, foley, speech).
"""

CSS = """
.main.fillable {max-width: 1250px !important}
.dark .gradio-container { color: var(--body-text-color); }
"""

with gr.Blocks(title="MiniMax-H3") as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
            )
            with gr.Row():
                image = gr.Image(label="First frame (optional)", type="filepath")
                last_image = gr.Image(label="Last frame (optional)", type="filepath")
            run = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced options", open=False):
                canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES), value=DEFAULT_CANVAS)
                duration = gr.Slider(label="Duration (s)", minimum=5, maximum=MAX_UI_DURATION, step=1, value=5)
                steps = gr.Slider(label="Steps", minimum=10, maximum=40, step=1, value=28)
                seed = gr.Number(label="Seed", value=42, precision=0)
            
        with gr.Column():
            video = gr.Video(label="Video + soundtrack")
            report = gr.Markdown(visible=False)

    image.upload(_fit_keyframe, [image, canvas], [image, canvas])

    gr.Examples(
        examples=[
            ["A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot", None, None, "1344x768 · 16:9 full"],
            ["A busy night market, neon signs reflecting in puddles, sizzling street food", None, None, "768x1344 · 9:16 full"],
            ["A cellist playing a slow melody in an empty concert hall", None, None, "768x768 · 1:1 full"],
            ["The fox looks around, then trots deeper into the forest", "examples/first.png", None, "1344x768 · 16:9 full"],
            ["A slow seamless camera move from the first view to the last", "examples/first.png", "examples/last.png", "1344x768 · 16:9 full"],
        ],
        inputs=[prompt, image, last_image, canvas],
        outputs=[video, report],
        fn=generate,
        cache_examples=True,
        cache_mode="lazy",
    )

    run.click(
        generate,
        [prompt, image, last_image, canvas, duration, steps, seed],
        [video, report],
        api_name="generate",
    )


if __name__ == "__main__":
    demo.launch(show_error=True, theme=gr.themes.Citrus(), css=CSS)
