"""The two halves of a **split** MiniMax-H3 deployment.

MiniMax-H3 is modular-only, and `MiniMaxH3Blocks` is a `SequentialPipelineBlocks` of eight steps:

    setup -> text_encoder -> vae_encoder -> prepare_layout -> prepare_latents -> set_timesteps -> denoise -> decode

The conditioner (a 62.14 GiB Qwen3-VL) and the denoiser (a 61.73 GiB transformer plus ~20.5 GiB of float32 VAEs) do
not fit on one 95 GiB card unquantized, so this module cuts the sequence in two at the `text_encoder` step:

  * `MiniMaxH3ConditionerBlocks` = `[setup, text_encoder]`  — loads `text_encoder` / `tokenizer` / `processor` only,
    and emits `prompt_embeds` + `text_token_tags`, which is the whole wire format between the two halves.
  * `MiniMaxH3GeneratorBlocks`   = everything else          — loads `transformer` / `vae` / `audio_vae` / the two
    schedulers only, and takes `prompt_embeds` + `text_token_tags` as *inputs*.

`setup` runs on both sides on purpose. It owns no component (it is PIL and arithmetic), it resolves the canvas, the
`17 * n + 5` frame count and the latent geometry, and it puts the keyframes onto that canvas — which the conditioner
needs to build its vision blocks and the generator needs to encode with the video VAE. Running it twice over the same
inputs is deterministic; the conditioner half returns its resolved `height` / `width` / `num_frames` anyway, so the
caller can pin them explicitly on the generator half.
"""

from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLatentsStep,
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3SetupStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoKeyframeVaeEncoderStep,
    MiniMaxH3DecodeStep,
    _generation_outputs,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam


class MiniMaxH3ConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split MiniMax-H3: the request plan plus the Qwen3-VL read at its 50th layer."""

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3SetupStep, MiniMaxH3TextEncoderStep]
    block_names = ["setup", "text_encoder"]

    @property
    def description(self):
        return (
            "The conditioner half of a split MiniMax-H3 deployment: resolves the request plan (canvas, frame count, "
            "latent geometry, keyframes on the canvas) and encodes MiniMax-H3's presentation of it into the "
            "`prompt_embeds` / `text_token_tags` pair the denoising half consumes."
        )

    @property
    def outputs(self):
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam("text_token_tags", description="The per-row modality tag of every row of `prompt_embeds`."),
            OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
            OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
            OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5."),
        ]


class MiniMaxH3GeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split MiniMax-H3: `MiniMaxH3Blocks` with its `text_encoder` step removed."""

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3SetupStep,
        MiniMaxH3AutoKeyframeVaeEncoderStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3SetTimestepsStep,
        MiniMaxH3DenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = [
        "setup",
        "vae_encoder",
        "prepare_layout",
        "prepare_latents",
        "set_timesteps",
        "denoise",
        "decode",
    ]

    @property
    def description(self):
        return (
            "The denoising half of a split MiniMax-H3 deployment: `MiniMaxH3Blocks` without its text-encoder step, so "
            "`prompt_embeds` and `text_token_tags` come in as inputs and the 62.14 GiB Qwen3-VL conditioner is never "
            "loaded here."
        )

    @property
    def outputs(self):
        return _generation_outputs()
