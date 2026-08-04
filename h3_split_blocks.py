"""The halves of a **split** MiniMax-H3 deployment, for both of its checkpoint partitions.

MiniMax-H3 is modular-only. Since https://github.com/huggingface/diffusers/pull/14371 the whole model is *one*
`MiniMaxH3Blocks` sequence whose branches are picked per request (and per `workflow=`) from the inputs:

    before_encode -> text_encoder -> vae_encoder -> denoise -> after_denoise -> decode

where `before_encode`, `text_encoder`, `vae_encoder` and `denoise` are each an auto-block that switches on
`references` (the `ref2va` workflow) versus the keyframe inputs (`t2va` / `fl2va`), and `denoise` is a whole
sub-sequence — `prepare_layout -> prepare_latents -> set_timesteps -> denoise` — against `transformer` or
`transformer_ref`.

The conditioner (a 62.14 GiB Qwen3-VL) and the denoiser (a 61.73 GiB transformer plus ~20.5 GiB of float32 VAEs) do
not fit on one 95 GiB card unquantized, so this module cuts that sequence in two at the `text_encoder` step, once per
partition:

  * `MiniMaxH3ConditionerBlocks` = `[resize, text_encoder]` — loads `text_encoder` / `tokenizer` / `processor` only
    (plus the `image_processor`, which is built from config and downloads nothing), and emits `prompt_embeds` +
    `text_token_tags`, which is the whole wire format between the two halves.
  * `MiniMaxH3GeneratorBlocks` = everything else — loads `transformer` / `vae` / `audio_vae` / the two schedulers
    only, and takes `prompt_embeds` + `text_token_tags` as *inputs*.
  * `MiniMaxH3Ref2VAConditionerBlocks` / `MiniMaxH3Ref2VAGeneratorBlocks` are the same cut through the `ref2va`
    branch, so one conditioner Space serves both partitions out of the weights it already holds.

What the refactor moved, and what that means for the cut:

  * the old `MiniMaxH3SetupStep` is gone. Its keyframe half is now `MiniMaxH3ResizeStep` (wrapped in the conditional
    `MiniMaxH3AutoResizeStep`, skipped entirely for a text-only request) and its geometry half moved *into*
    `MiniMaxH3PrepareLayoutStep`, which lives on the denoising side. So the `t2va` / `fl2va` conditioner half no
    longer resolves `num_frames` at all — the caller does, with `align_num_frames`, which is the one line of
    arithmetic that used to come back in the plan.
  * `MiniMaxH3Ref2VASetupStep` survived and still resolves the canvas *and* the frame count, but `num_frames` is now
    required there: a request that leaves the duration to its single audio-bearing reference resolves it caller-side.
  * unpacking the denoised rows moved out of the decoders into `MiniMaxH3AfterDenoiseStep`, so the generating halves
    carry that step explicitly.
  * `model_name` is `"minimax-h3"` on every block now — the `"minimax-h3-ref2va"` pipeline mapping was dropped when
    the two blocksets became workflows of one pipeline.

`resize` / `setup` run on both sides on purpose. They own no pretrained component (PIL, decoded media and
arithmetic), they resolve the canvas and prepare the keyframes or normalize the references — which the conditioner
needs to build its vision blocks and the generator needs to encode with the VAEs. Running them twice over the same
inputs is deterministic; both conditioner halves return the resolved `height` / `width` / `num_frames` anyway, so the
caller pins them explicitly on the generating half.

Only *text* encoding is remote. `vae_encoder` / `reference_encoder` stay on the denoising side: they run the two
autoencoders, which the conditioner Space does not hold.
"""

from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (
    MiniMaxH3AutoKeyframeVaeEncoderStep,
    MiniMaxH3AutoResizeStep,
    MiniMaxH3CoreDenoiseStep,
    MiniMaxH3DecodeStep,
    MiniMaxH3Ref2VACoreDenoiseStep,
    _generation_outputs,
)
from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam


def _wire_outputs(num_frames: bool = True) -> list[OutputParam]:
    """The wire format of the split, plus the plan the caller pins on the generating half.

    `num_frames` is only declared by the `ref2va` half: its setup step is the one that still resolves the frame count,
    while the keyframe half's own resolution moved into the layout step, on the other side of the cut.
    """
    return [
        OutputParam.template("prompt_embeds"),
        OutputParam("text_token_tags", description="The per-row modality tag of every row of `prompt_embeds`."),
        OutputParam("height", type_hint=int, description="Resolved height of the generated video in pixels."),
        OutputParam("width", type_hint=int, description="Resolved width of the generated video in pixels."),
        *(
            [OutputParam("num_frames", type_hint=int, description="Resolved number of frames, of the form 17 * n + 5.")]
            if num_frames
            else []
        ),
    ]


class MiniMaxH3ConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split MiniMax-H3: the keyframes on the canvas plus the Qwen3-VL read at layer 50."""

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3AutoResizeStep, MiniMaxH3TextEncoderStep]
    block_names = ["resize", "text_encoder"]

    @property
    def description(self):
        return (
            "The conditioner half of a split MiniMax-H3 deployment: puts the keyframes onto the target canvas and "
            "encodes MiniMax-H3's presentation of the request into the `prompt_embeds` / `text_token_tags` pair the "
            "denoising half consumes. The frame count is the caller's to align — that arithmetic now lives in the "
            "layout step, on the denoising side."
        )

    @property
    def outputs(self):
        return _wire_outputs(num_frames=False)


class MiniMaxH3GeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split MiniMax-H3: `MiniMaxH3Blocks` with its `text_encoder` step removed."""

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3AutoResizeStep,
        MiniMaxH3AutoKeyframeVaeEncoderStep,
        MiniMaxH3CoreDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = ["resize", "vae_encoder", "denoise", "after_denoise", "decode"]

    @property
    def description(self):
        return (
            "The denoising half of a split MiniMax-H3 deployment: the `t2va` / `fl2va` branch of `MiniMaxH3Blocks` "
            "without its text-encoder step, so `prompt_embeds` and `text_token_tags` come in as inputs and the "
            "62.14 GiB Qwen3-VL conditioner is never loaded here."
        )

    @property
    def outputs(self):
        return _generation_outputs()


class MiniMaxH3Ref2VAConditionerBlocks(SequentialPipelineBlocks):
    """The conditioner half of a split `ref2va`: the resolved plan plus the Qwen3-VL read at its 50th layer.

    Component for component this is `MiniMaxH3ConditionerBlocks` — `text_encoder`, `tokenizer`, `processor` — which
    is what lets one conditioner Space serve both partitions of the checkpoint out of the weights it already holds.
    What differs is the presentation the Qwen3-VL is shown: `ref2va` prepends a label per reference, numbered per
    modality, and a vision block per image and per merged video frame pair, so the references themselves have to
    reach this half. An audio reference never does — it contributes its `"<Audio j>: "` label and nothing else — but
    it is still part of the request here, because the setup step normalizes every soundtrack and validates the mix.
    """

    model_name = "minimax-h3"
    block_classes = [MiniMaxH3Ref2VASetupStep, MiniMaxH3Ref2VATextEncoderStep]
    block_names = ["setup", "text_encoder"]

    @property
    def description(self):
        return (
            "The conditioner half of a split MiniMax-H3 `ref2va` deployment: resolves the request plan (canvas, frame "
            "count, references normalized onto MiniMax-H3's own rates and resolutions) and encodes MiniMax-H3's "
            "presentation of it into the `prompt_embeds` / `text_token_tags` pair the denoising half consumes."
        )

    @property
    def outputs(self):
        return _wire_outputs()


class MiniMaxH3Ref2VAGeneratorBlocks(SequentialPipelineBlocks):
    """The denoising half of a split `ref2va`: the `ref2va` branch with its `text_encoder` step removed.

    Only the text-encoder step is dropped. `reference_encoder` is this half's own encoder — it runs the video VAE
    over the image and video references and the audio VAE over the soundtracks, and its output shapes are where every
    reference block's geometry in the packed layout comes from — so it stays here, next to the autoencoders.
    """

    model_name = "minimax-h3"
    block_classes = [
        MiniMaxH3Ref2VASetupStep,
        MiniMaxH3Ref2VAReferenceEncoderStep,
        MiniMaxH3Ref2VACoreDenoiseStep,
        MiniMaxH3AfterDenoiseStep,
        MiniMaxH3DecodeStep,
    ]
    block_names = ["setup", "reference_encoder", "denoise", "after_denoise", "decode"]

    @property
    def description(self):
        return (
            "The denoising half of a split MiniMax-H3 `ref2va` deployment: the `ref2va` branch of `MiniMaxH3Blocks` "
            "without its text-encoder step, so `prompt_embeds` and `text_token_tags` come in as inputs and the "
            "62.14 GiB Qwen3-VL conditioner is never loaded here. The transformer is the `transformer_ref` partition."
        )

    @property
    def outputs(self):
        return _generation_outputs()
