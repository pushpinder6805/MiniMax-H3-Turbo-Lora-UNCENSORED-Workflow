---
title: MiniMax-H3 (split, bf16)
emoji: 🎬
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
short_description: Unquantized MiniMax-H3, split across two ZeroGPU Spaces
suggested_hardware: zero-a10g
---

# MiniMax-H3 — unquantized, split across two Spaces

Joint video **and** soundtrack out of a single denoising pass, at **bfloat16 with no quantization anywhere**.

This Space is the denoising half: the 61.73 GiB transformer and the two autoencoders. The 62.14 GiB Qwen3-VL
conditioner runs in
[`minimax-h3-conditioner`](https://huggingface.co/spaces/diffusers-internal-dev/minimax-h3-conditioner), which this
Space calls over the gradio API for every request.

## Why split

MiniMax-H3 is 195.9 GiB in bfloat16 and a ZeroGPU Space is evicted at **150 GB of storage**. An unquantized single
Space is therefore impossible — the existing demos
([`minimax-h3`](https://huggingface.co/spaces/diffusers-internal-dev/minimax-h3),
[`-fp8`](https://huggingface.co/spaces/diffusers-internal-dev/minimax-h3-fp8)) run NVFP4 and float8 weights for that
reason alone. Cut the `MiniMaxH3Blocks` sequence at its `text_encoder` step and both halves fit unquantized:

| Space | Subfolders | Download | Resident |
|---|---|---|---|
| [`minimax-h3-conditioner`](https://huggingface.co/spaces/diffusers-internal-dev/minimax-h3-conditioner) | `text_encoder/` + `tokenizer/` + `processor/` | 66.7 GB | 62.15 GiB bf16 |
| this one | `transformer/` + `vae/` + `audio_vae/` | 77.3 GB | 61.73 GiB bf16 + ~20.5 GiB float32 |

Besides the quality argument, unquantized weights are the ones AoTI can export; an NVFP4 checkpoint cannot be
exported at all.

## How the split is expressed

`MiniMaxH3Blocks` is a `SequentialPipelineBlocks` of eight steps:

```
setup -> text_encoder -> vae_encoder -> prepare_layout -> prepare_latents -> set_timesteps -> denoise -> decode
```

`h3_split_blocks.py` subclasses it with the `text_encoder` step removed. Dropping the step drops the three
components it declares, so `load_components` resolves `transformer` / `vae` / `audio_vae` / the two schedulers out of
the shared `modular_model_index.json` and never fetches the conditioner — and `prompt_embeds` and `text_token_tags`
become ordinary required inputs of the pipeline call:

```py
pipe = MiniMaxH3GeneratorBlocks().init_pipeline("diffusers-internal-dev/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)
state = pipe(prompt_embeds=..., text_token_tags=..., height=768, width=1344, num_frames=124, num_inference_steps=30)
```

The wire format is exactly those two tensors — `(1, num_text_tokens, 5120)` bfloat16 and `(num_text_tokens,)` int64 —
carried as one safetensors file with the resolved `height` / `width` / `num_frames` in its metadata header. A
text-only request is 246 KB of it; one 768x1344 keyframe adds 1016 vision rows and takes it to 10.7 MB.

The `setup` step runs on **both** halves. It owns no component (PIL and arithmetic) and it resolves the canvas, the
`17 * n + 5` frame count and the keyframes placed onto that canvas — which the conditioner needs to build its vision
blocks and this Space needs to encode with the video VAE. It is deterministic, and the conditioner returns the plan
it resolved so this Space pins the same canvas rather than re-deriving it.

## Nothing is paid for with GPU time

Download, load and the move onto the card happen at **startup**: `import spaces` at module top patches `torch.cuda`
before any GPU is attached, and these are plain bfloat16 tensors, so ZeroGPU's startup packing handles them (it is a
torchao `Float8Tensor` that cannot be packed, and there is none here). The conditioner round trip is a network call
on this Space's CPU. A `@spaces.GPU` call is therefore only the denoise loop and the two decoders.

## Generation constraints

Fixed by the checkpoint: 24 fps, a 768 pixel short edge, 5 to 15 s, `num_frames` snapped up to the next `17 * n + 5`,
no CFG and no negative prompt (it is guidance-distilled, so every step is one forward pass).

## Space variables

| Variable | Default | Meaning |
|---|---|---|
| `H3_CONDITIONER` | `diffusers-internal-dev/minimax-h3-conditioner` | The Space this one asks for embeddings. |
| `H3_PLACEMENT` | `resident` | `resident` keeps all 82.3 GiB of weights on the 95.0 GiB card; `offload` hands placement to `ComponentsManager.enable_auto_cpu_offload`. |
| `H3_ATTENTION` | `_native_cudnn` | cuDNN's fused kernel, 10–20% faster than the SDPA default and needs nothing installed. flash-attention 3 is sm90-only and this pool is sm120. |
| `H3_GPU_DURATION` | `900` | Seconds per request; the pool applies a 1.5 duration factor. |
| `H3_GPU_SIZE` | `xlarge` | ZeroGPU allocation size. `large` does not fit. |

## Required secret

`HF_TOKEN` — `diffusers-internal-dev/MiniMax-H3` is private, and so is the conditioner Space this one calls.

## Where diffusers comes from

MiniMax-H3 is modular-only and not in a released `diffusers`, so the integration branch's `src/diffusers` tree is
vendored here as a top-level `diffusers/` package; the working directory comes first on `sys.path`, so there is no
install step. `requirements.txt` only carries what that tree imports.
