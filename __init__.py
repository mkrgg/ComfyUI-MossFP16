"""Load a MOSS-Image diffusion model in fp16 on GPUs without bf16.

MOSS-Image declares ``supported_inference_dtypes = [bfloat16, float32]``.  On a
Turing card (no bf16) that makes the sampler cast the whole residual stream to
fp32 (``BaseModel.get_dtype_inference`` returns the manual cast dtype), even
when the block linears are int8.  The loader below adds fp16 to that list for
the duration of the load, which is what upstream #17684 does in
``supported_models.py``: weights are created in fp16 and the int8 linears run
with fp16 activations, with no per-step casts.

The optional clamp bounds each block's residual output at the fp16 maximum.  It
is not needed for the fp16-fixed or healed weights (their stream peaks around
4k); it is there for the official base weights, which peak near 62k.
"""

import logging

import torch

import comfy.sd
import comfy.supported_models
import comfy.utils
import folder_paths

FP16_MAX = 65504.0

MOSS_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _clamp_output(module, args, output):
    if isinstance(output, torch.Tensor) and output.dtype == torch.float16:
        output.clamp_(-FP16_MAX, FP16_MAX)
    return output


def _install_clamp(model):
    diffusion = getattr(model.model, "diffusion_model", None)
    if not hasattr(diffusion, "layers"):
        raise RuntimeError("clamp_fp16 is only available for MOSS-Image checkpoints.")
    for block in diffusion.layers:
        if getattr(block, "_moss_fp16_clamp", False):
            continue
        block.register_forward_hook(_clamp_output)
        block._moss_fp16_clamp = True


def load_moss_fp16(unet_path, model_options=None, clamp=False, disable_dynamic=False):
    config = comfy.supported_models.MOSSImage
    original = config.supported_inference_dtypes
    config.supported_inference_dtypes = MOSS_DTYPES
    try:
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd,
            model_options={} if model_options is None else dict(model_options),
            metadata=metadata,
            disable_dynamic=disable_dynamic,
        )
    finally:
        config.supported_inference_dtypes = original

    if model is None:
        raise RuntimeError("Could not detect a diffusion model type for {}.".format(unet_path))

    if clamp:
        _install_clamp(model)

    model.cached_patcher_init = (load_moss_fp16, (unet_path, model_options, clamp))
    return model


class MossImageLoaderFP16:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "clamp_fp16": ("BOOLEAN", {"default": False, "tooltip": "Clamp each block's residual at the fp16 maximum. Not needed for the fp16-fixed or healed weights."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MOSS"
    DESCRIPTION = "Loads a MOSS-Image diffusion model with fp16 weights and compute on GPUs without bf16, keeping int8 checkpoints quantized."

    def load(self, model_name, clamp_fp16):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        logging.info("MOSS loader: {} (fp16 dtypes{}".format(model_name, ", residual clamp" if clamp_fp16 else "") + ")")
        return (load_moss_fp16(unet_path, clamp=clamp_fp16),)


NODE_CLASS_MAPPINGS = {"MossImageLoaderFP16": MossImageLoaderFP16}
NODE_DISPLAY_NAME_MAPPINGS = {"MossImageLoaderFP16": "MOSS Image Loader (FP16)"}
