"""Stable Diffusion pipeline helpers."""
from __future__ import annotations

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def load_pipeline(cfg: dict) -> StableDiffusionControlNetPipeline:
    """Load Stable Diffusion pipeline with ControlNet."""
    controlnet = ControlNetModel.from_pretrained(
        cfg["model"]["controlnet"], torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg["model"]["base"],
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        cfg["model"]["vae"], torch_dtype=torch.float16
    )
    pipe.vae = vae
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_loras(pipe: StableDiffusionControlNetPipeline, cfg: dict) -> None:
    """Load and optionally fuse LoRA adapters."""
    loras = cfg.get("loras")
    if not loras:
        return

    if isinstance(loras, dict):
        loras = [loras]

    for idx, lora_cfg in enumerate(loras):
        model = lora_cfg.get("model")
        if model is None:
            continue
        weight_name = lora_cfg.get("weight_name")
        adapter_name = lora_cfg.get("adapter_name", f"lora_{idx}")
        pipe.load_lora_weights(model, weight_name=weight_name, adapter_name=adapter_name)
        scale = lora_cfg.get("scale")
        if scale is not None:
            pipe.fuse_lora(lora_scale=scale, adapter_names=[adapter_name])


def get_free_gpu_memory_gb(device: int = 0) -> float:
    """Return available GPU memory in gigabytes."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not found.")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes / (1024 ** 3)


def sizeof_pipe(pipe: StableDiffusionControlNetPipeline) -> float:
    """Estimate VRAM usage of the pipeline in gigabytes."""
    modules = []
    for name in ["unet", "controlnet", "vae", "text_encoder", "text_encoder_2"]:
        if hasattr(pipe, name):
            modules.append(getattr(pipe, name))
    total_params = sum(p.numel() for m in modules for p in m.parameters())
    return total_params * 2 / (1024 ** 3)
