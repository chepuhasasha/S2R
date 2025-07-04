"""Generate an image from a sketch using Stable Diffusion ControlNet."""

from pathlib import Path

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


def load_pipeline() -> StableDiffusionControlNetPipeline:
    """Load Stable Diffusion pipeline with ControlNet."""
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16
    )
    pipe.vae = vae

    # более стабильный scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


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


def preprocess(path: str, size: int) -> Image.Image:
    """Resize and pad an image keeping aspect ratio."""
    img = Image.open(path).convert("RGB")
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


def main() -> None:
    pipe = load_pipeline()
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

    prompt = (
        "minimalistic modern house, white concrete,"
        "large glass walls, panoramic windows, elegant, photorealistic,"
        "Icelandic volcanic nature, black sand beach, winding river,"
        "dramatic mountains, mist, sunset lighting, no people, high detail, sharp focus, 8k"
    )
    negative_prompt = (
        "blurry, low quality, people, watermark, text, logo,"
        "distortion, cartoon, surreal, painting, unrealistic, overexposed,"
        "underexposed, bad anatomy, artifacts"
    )

    preprocessed = preprocess("scetch.png", 1024)
    # Uncomment to debug preprocessing
    # save_debug(preprocessed, "preprocess.png")

    edged = canny(preprocessed, 100, 200)
    # Uncomment to debug edges
    # save_debug(edged, "canny.png")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=edged,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    result.save("output.png")


if __name__ == "__main__":
    main()
