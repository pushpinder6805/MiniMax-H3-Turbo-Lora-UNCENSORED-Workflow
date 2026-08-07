---
title: MiniMax H3
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: true
short_description: Video generation with a synchronized soundtrack
suggested_hardware: zero-a10g
---

# MiniMax-H3 — unquantized, split across two Spaces

Joint video **and** soundtrack out of a single denoising pass, at **bfloat16 with no quantization anywhere**.

This Space is the denoising half: the 61.73 GiB transformer and the two autoencoders. The 62.14 GiB Qwen3-VL
conditioner runs in
[`qwen3vl-conditioner`](https://huggingface.co/spaces/multimodalart/qwen3vl-conditioner), which this
Space calls over the gradio API for every request. The weights are the public
[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) diffusers checkpoint.

## Why split

MiniMax-H3 is 195.9 GiB in bfloat16 and a ZeroGPU Space is evicted at **150 GB of storage**. An unquantized single
Space is therefore impossible, which is why quantized demos of it run NVFP4 or float8 weights. Cut the
`MiniMaxH3Blocks` sequence at its `text_encoder` step and both halves fit unquantized:

| Space | Subfolders | Download | Resident |
|---|---|---|---|
| [`qwen3vl-conditioner`](https://huggingface.co/spaces/multimodalart/qwen3vl-conditioner) | `text_encoder/` + `tokenizer/` + `processor/` | 66.7 GB | 62.15 GiB bf16 |
| this one | `transformer/` + `vae/` + `audio_vae/` | 77.3 GB | 61.73 GiB bf16 + 10.43 GiB float32 |

Besides the quality argument, unquantized weights are the ones AoTI can export; an NVFP4 checkpoint cannot be
exported at all.

## Studio frontend (gradio.Server)

The UI is a custom single-page studio (`index.html`) served by [`gradio.Server`](https://www.gradio.app/docs/gradio/server):
`@app.get("/")` serves the page, `@app.api(name="generate")` keeps the request on Gradio's queue (concurrency control,
SSE, ZeroGPU booking, `gradio_client` compatibility — the API name and signature are unchanged), and the page talks
to it with the `@gradio/client` JS package. `/status` and `/config` are plain FastAPI routes the page polls for
readiness and the canvas table. Keyframe cover-crop / canvas fitting moved server-side into `generate`, so API
callers get the same treatment the old upload event gave.

## 4-step Turbo LoRA

The transformer runs with [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
folded into its bf16 weights at startup (`h3_lora.py`), so the default is **6 sampling steps** instead of 28 
card's comfort zone at the current checkpoint; 4 is the design point but softer). The larry fold mirrors the diffusers key
conversion exactly (fused-QKV thirds, the `SwiGLU` gate/value swap, the shared AdaLN row layout); lightx keys are
already diffusers-native. Both happen before the AoTI package is patched in, so compiled blocks carry the update too,
and the low-rank factors stay resident so switching is an in-place unfold/fold through one bf16 rounding.
`H3_LORA` selects the larry file (`off` skips it), `H3_LIGHTX=off` skips lightx, `H3_LORA_DEFAULT` picks which set
starts folded, and `H3_LORA_STRENGTH` scales the larry update (the card's sharpness/artifact dial).

## AoTI-compiled blocks

With `H3_AOTI=1` the 50 repeated transformer blocks run from a compiled package,
[`multimodalart/minimax-h3-aoti`](https://huggingface.co/multimodalart/minimax-h3-aoti)`:bf16/torch2.11/sm120/dynamic` — a single dynamic-sequence artifact that serves
every canvas, duration and prompt length. It carries no weights (it reads each block's live ones), so patching it in
is startup CPU work and costs no GPU time.

It removes a near-constant ~0.5 s/step — 50 blocks' worth of kernel-launch overhead plus the norm / rotary / AdaLN
epilogues around the matmuls — and cannot touch the matmuls themselves. So it pays best where the block is *not*
compute bound, i.e. on the small canvases:

| canvas (HxW) | eager s/step | AoTI s/step | faster |
|---|---|---|---|
| 768x1344 | 10.20 | 9.73 | +4.6% |
| 640x1152 | 6.46 | 5.88 | +9.1% |
| 544x960 | 4.02 | 3.58 | +11.0% |

At 72.16 GiB of weights on a 95.0 GiB card there is no offloading in the request path at all, which is why an
*unquantized* MiniMax-H3 is also the fastest one measured here: **10.6 s/step** against the 19–21 s/step the 4 bit
Space pays, where the whole cost is the traffic auto-offload has to move.

## How the split is expressed

`MiniMaxH3Blocks` is a `SequentialPipelineBlocks` whose branches are picked per request — and per `workflow=` — from
the inputs:

```
before_encode -> text_encoder -> vae_encoder -> denoise -> after_denoise -> decode
```

where `denoise` is itself `prepare_layout -> prepare_latents -> set_timesteps -> denoise`.

`h3_split_blocks.py` subclasses it with the `text_encoder` step removed. Dropping the step drops the three
components it declares, so `load_components` resolves `transformer` / `vae` / `audio_vae` / the two schedulers out of
the shared `modular_model_index.json` and never fetches the conditioner — and `prompt_embeds` and `text_token_tags`
become ordinary required inputs of the pipeline call:

```py
pipe = MiniMaxH3GeneratorBlocks().init_pipeline("MiniMaxAI/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)
state = pipe(prompt_embeds=..., text_token_tags=..., height=768, width=1344, num_frames=124, num_inference_steps=30)
```

The wire format is exactly those two tensors — `(1, num_text_tokens, 5120)` bfloat16 and `(num_text_tokens,)` int64 —
carried as one safetensors file with the resolved `height` / `width` / `num_frames` in its metadata header. A
text-only request is 246 KB of it; one 768x1344 keyframe adds 1016 vision rows and takes it to 10.7 MB.

The keyframe `resize` step runs on **both** halves. It owns no pretrained component (PIL and arithmetic) and it puts
the keyframes onto the target canvas — which the conditioner needs to build its vision blocks and this Space needs to
encode with the video VAE. It is deterministic, and the conditioner returns the plan it resolved so this Space pins
the same canvas rather than re-deriving it. Two things that step no longer does, and that both halves therefore do
themselves: EXIF-transposing a keyframe into upright RGB, and snapping `num_frames` to `17 * n + 5` — the frame count
is resolved by the layout step, which lives on this side of the cut.

## Nothing is paid for with GPU time

The 77.3 GB download and the load happen at **startup**: `import spaces` at module top patches `torch.cuda` before
any GPU is attached, so nothing about the load needs a card. The conditioner round trip is a network call on this
Space's CPU. A `@spaces.GPU` call is therefore only the placement (once) and the denoise loop and the two decoders.

### The 150 GB quota, not the 95 GiB card, is what rules out startup placement

One thing does *not* happen at startup: the move onto the card. `spaces`' startup `torch.pack()` writes every
startup-resident CUDA tensor to a **second copy on disk** and only deletes the downloaded originals afterwards
(`Cleaned 62.13GB of tensor files ... after packing`, which is what keeps the conditioner half comfortable at
66.7 GB). Packing 77.3 GB needs 154.6 GB at once, and this Space is evicted mid-pack:

```
ZeroGPU tensors packing:   0%|          | 0.00/77.3G
OSError: [Errno 28] No space left on device      # os.posix_fallocate, spaces/zero/torch/packing.py
```

Unlinking the shards first does not rescue it. `.to("cuda")` under the startup patch does not release the
memory-mapped safetensors, so nothing is freed — and the pack's own cleanup walks those still-open mappings and
`lstat`s them, so a deleted blob becomes `FileNotFoundError: .../blobs/3d449... (deleted)`.

Placement therefore happens on the **first GPU call**, `PIPE.to("cuda")` at the top of the `@spaces.GPU` function:
about 10 s of PCIe once, then a no-op walk, and the denoise loop runs with everything resident and no offloading at
all. It is the same trick the 4 bit Space uses, for the same reason.

## Generation constraints

Fixed by the checkpoint: 24 fps, a 768 pixel short edge, 5 to 15 s, `num_frames` snapped up to the next `17 * n + 5`,
no CFG and no negative prompt (it is guidance-distilled, so every step is one forward pass).

## Measured

An `rtx-pro-6000` Job — the same silicon as the ZeroGPU pool (RTX PRO 6000 Blackwell, sm120, 95.0 GiB) — running
exactly this blockset over the wire format, 1344x768, 124 frames, 30 steps, bfloat16, cuDNN attention, everything
resident:

| | |
|---|---|
| `load_components` (77.3 GB, warm Xet) | 43 s |
| `.to("cuda")`, once | 10 s |
| resident weights | 72.16 GiB |
| denoise + decode | 317 s, **10.58 s/step** |
| peak allocated / reserved | 78.54 / 85.37 GiB |
| output | h264 1344x768 @ 24 fps, 5.167 s + stereo AAC @ 32 kHz |

And on this Space itself, driven over `gradio_client`. Startup is 93 s — the 77.3 GB download and the load, with
no placement and therefore no pack.

| Request | Conditioner | Denoise + decode | Steady | Round trip |
|---|---|---|---|---|
| text only, 18 tokens | 7 s | 339 s | 10.53 s/step | 353 s |
| one 768x1344 keyframe, 1034 tokens | 9 s | 370 s | 11.39 s/step | 386 s |

The keyframe costs about 8% per step rather than a placement penalty: it puts 1016 vision rows in front of the
prompt *and* 1016 conditioning rows in the packed sequence, and MiniMax-H3 attends over all of it every layer. The
one-time `PIPE.to("cuda")` is inside the first row's 339 s and does not reappear in the second.

## Space variables

| Variable | Default | Meaning |
|---|---|---|
| `H3_MODEL_REPO` | `MiniMaxAI/MiniMax-H3` | The diffusers-layout checkpoint. Public. |
| `H3_CONDITIONER` | `multimodalart/qwen3vl-conditioner` | The Space this one asks for embeddings. |
| `H3_PLACEMENT` | `lazy` | `lazy` moves all 72.16 GiB onto the card on the first GPU call and leaves it there; `offload` hands placement to `ComponentsManager.enable_auto_cpu_offload` instead. |
| `H3_ATTENTION` | `_native_cudnn` | cuDNN's fused kernel, 10–20% faster than the SDPA default and needs nothing installed. flash-attention 3 is sm90-only and this pool is sm120. |
| `H3_GPU_DURATION` | `900` | Seconds per request; the pool applies a 1.5 duration factor. |
| `H3_GPU_SIZE` | `xlarge` | ZeroGPU allocation size. `large` does not fit. |
| `H3_LORA` | `minimax_h3_turbo_4step_ema_ckpt850.safetensors` | Turbo LoRA file folded into the transformer at startup. `off` disables. |
| `H3_LORA_REPO` | `larryvrh/MiniMax-H3-Turbo-Lora` | Hub repo the LoRA is fetched from. |
| `H3_LORA_STRENGTH` | `1.0` | Scales the larry LoRA delta (sharpness/artifact trade-off). |
| `H3_LIGHTX` | `on` | Set to `off` to skip loading the lightx2v LoRA set. |
| `H3_LORA_DEFAULT` | `larry` | Which loaded LoRA set starts folded (`larry` / `lightx`). |

## Whose GPU quota pays

Two cards are booked per request — this Space's denoise loop and the conditioner's forward — and both are billed to the
requesting user, with nothing here arranging it: `gradio_client` attaches the caller's own `x-ip-token` to every
outgoing call, reading it off gradio's `LocalContext` inside the event listener (`Client.send_data` ->
`add_zero_gpu_headers`), and ZeroGPU charges the booking to whatever that token identifies.

A caller with no token to forward — a `gradio_client` script rather than a browser — leaves the conditioner's booking
attributed to this Space's pod IP and its small shared quota. An unattributed caller may book at most 120 credits at a
time and an `xlarge` booking costs twice its seconds, so the conditioner books the encode (45 s) and a prompt upsample
(60 s) as two separate calls, each within that ceiling.

## Secrets

None are required. Everything this Space downloads is public — the
[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) checkpoint and the
[`multimodalart/minimax-h3-aoti`](https://huggingface.co/multimodalart/minimax-h3-aoti) packages — and the conditioner
is a public Space called on the requesting user's own ZeroGPU token, never on an org token.

## Where diffusers comes from

MiniMax-H3 is modular-only and not in a released `diffusers`, so `requirements.txt` installs it from the canonical
pull request, [huggingface/diffusers#14371](https://github.com/huggingface/diffusers/pull/14371), pinned to the commit
`665f5782` (`refs/pull/14371/head`) rather than to the moving `minimax-h3-refactor` branch.

That PR is a WIP, so it needs re-pinning whenever it updates, and `h3_split_blocks.py` — which subclasses its block
classes to cut the pipeline in two — has to be re-checked against the new head at the same time.
