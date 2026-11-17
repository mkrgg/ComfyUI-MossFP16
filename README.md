# ComfyUI-MossFP16

A single loader node for MOSS-Image checkpoints on GPUs without bf16 (Turing:
RTX 20xx, GTX 16xx, Quadro RTX, Tesla T4).

HF repo: https://huggingface.co/mkrgg/MOSS-Image-Base-INT8-FP16-Turing

This node was tested on ComfyUI 0.45.0 with:
- triton 3.2.0 (--enable-triton-backend)
- pytorch version: 2.9.1+cu130

At 2MP on a RTX 2060 SUPER:
- moss_image_base_int8_convrot: 6.64s/it
- moss_image_base_int8_fp16: **4.78s/it**
- moss_image_base_int8_convrot + node **(*)**: 4.80s/it

**(*)** You can use the base int8 with the node and see the same speed boost, however if you are using loras at high mp your output is gonna be with artifacts or some visual bug due to the clamp. There's more information about this in the hf repo.

`MOSSImage` declares `supported_inference_dtypes = [bfloat16, float32]`. Without
bf16, ComfyUI stores the weights in bf16 and runs the whole residual stream
through an fp32 manual cast (`BaseModel.get_dtype_inference`), so the int8
linears are native but attention, norms, RoPE and the residual adds run on the
CUDA cores in fp32. `--fp16-unet` does not change the manual cast dtype of a
quantized model.

The node adds fp16 to that list for the duration of the load (the dtype part of
ComfyUI PR #17684, applied at runtime instead of editing `supported_models.py`).
The weights are created in fp16, the int8 ConvRot linears keep their codes and
run with fp16 activations, and nothing is cast per step.

## Inputs

- `model_name`: a file from `models/diffusion_models`.
- `clamp_fp16`: bounds every block's residual output at the fp16 maximum
  (±65504). Off by default. Not needed for the fp16-fixed or healed weights
  (their stream peaks around 4k); enable it for the official base weights,
  whose stream peaks near 62k at 1024² and overflows at 2560².

# Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mkrgg/ComfyUI-MossFP16.git
```

## Use

Restart ComfyUI, add **MOSS Image Loader (FP16)** (category `MOSS`) and select
the checkpoint, e.g. `moss_image_heal_int8_fp16.safetensors`.

For reloads under memory pressure the node registers its own
`cached_patcher_init`, so the fp16 dtype list is applied again instead of
falling back to the core loader.

## AI

The custom node and the README was created using Inklink 2 Small.
