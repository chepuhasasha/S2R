"""Generate an image from a sketch using Stable Diffusion ControlNet."""

from pathlib import Path
from datetime import datetime
from shutil import copy2
import yaml

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def load_config(path: str = "config.yaml") -> dict:
    """Load generation settings from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    # более стабильный scheduler
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
    # FP16 weights occupy 2 bytes
    return total_params * 2 / (1024 ** 3)


def canny(image: Image.Image, low: int, high: int) -> Image.Image:
    """Apply Canny edge detection and return an RGB image."""
    arr = np.array(image)
    arr = cv2.Canny(arr, low, high)
    arr = arr[:, :, None]
    arr = np.concatenate([arr, arr, arr], axis=2)
    return Image.fromarray(arr)


from typing import Union


def preprocess(source: Union[str, Image.Image], size: int) -> Image.Image:
    """Resize and pad an image keeping aspect ratio."""
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        img = Image.open(source).convert("RGB")
    w, h = img.size
    if max(w, h) > size:
        s = size / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    pad_w = (size - img.width) // 2
    pad_h = (size - img.height) // 2
    img = ImageOps.expand(
        img,
        border=(pad_w, pad_h, size - img.width - pad_w, size - img.height - pad_h),
        fill="white",
    )
    return ImageOps.autocontrast(img)


def save_debug(img: Image.Image, name: str, base: str = "debug") -> None:
    """Save intermediate image if debugging is enabled."""
    out_dir = Path(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / name)


def generate(
    pipe: StableDiffusionControlNetPipeline,
    cfg: dict,
    image: Union[str, Image.Image] | None = None,
) -> Image.Image:
    """Generate an image using the given pipeline and config."""
    prompt = cfg["prompt"]
    negative_prompt = cfg["negative_prompt"]
    if image is None:
        image = cfg["input"]

    preprocessed = preprocess(image, cfg.get("preprocess_size", 1024))
    save_debug(preprocessed, "preprocess.png", base=cfg.get("debug_dir", "debug"))

    edge_cfg = cfg.get("canny", {})
    low = edge_cfg.get("low", 100)
    high = edge_cfg.get("high", 200)
    edged = canny(preprocessed, low, high)
    save_debug(edged, "canny.png", base=cfg.get("debug_dir", "debug"))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edged,
        controlnet_conditioning_scale=cfg.get("controlnet_conditioning_scale", 1.0),
        num_inference_steps=cfg.get("num_inference_steps", 20),
        guidance_scale=cfg.get("guidance_scale", 7.5),
    ).images[0]

    result.save(cfg.get("output", "output.png"))
    return result


def main() -> None:
    cfg = load_config()

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    copy2("config.yaml", run_dir / "config.yaml")

    cfg["debug_dir"] = str(run_dir / "debug")
    cfg["output"] = str(run_dir / "output.png")

    pipe = load_pipeline(cfg)
    load_loras(pipe, cfg)
    size = sizeof_pipe(pipe)
    free_mem = get_free_gpu_memory_gb()
    print(f"VRAM needed: {size} GB")
    print(f"Free GPU VRAM: {free_mem} GB")

    if free_mem > size:
        pipe.to("cuda")
    else:
        print(f"Not enough memory: {size - free_mem} GB. Enable sequential cpu offload...")
        print(f"VRAM needed: {size} GB")
        print(f"Free GPU VRAM: {free_mem} GB")
        pipe.enable_sequential_cpu_offload()

    generate(pipe, cfg)


if __name__ == "__main__":
    main()
