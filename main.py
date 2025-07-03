# main.py  —  проверено 02-Jul-2025, diffusers 0.34.0, torch 2.7.1+cu121

from pathlib import Path
from datetime import datetime
import argparse, shutil
import torch, cv2, numpy as np
from PIL import Image, ImageOps
import yaml
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)




def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_output_dir(base: str = "runs") -> Path:
    """Return a unique directory for all output images."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base) / ts
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir

def get_free_gpu_memory_gb(device: int = 0) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not found.")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes / (1024 ** 3)

def sizeof_pipe(pipe) -> float:
    modules = []
    for name in ['unet', 'controlnet', 'vae', 'text_encoder', 'text_encoder_2']:
        if hasattr(pipe, name):
            modules.append(getattr(pipe, name))
    total_params = sum(p.numel() for m in modules for p in m.parameters())
    return total_params * 2 / (1024 ** 3)  # FP16 = 2 bytes per parameter

def dbg(img: Image.Image, name: str, out_dir: Path):
    """Save intermediate image to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / name)

def preprocess(path: str, size: int) -> Image.Image:
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
    img = ImageOps.autocontrast(img)
    return img

def canny(image: Image.Image, low: int, high: int) -> Image.Image:
    image = np.array(image)
    image = cv2.Canny(image, low, high)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)
    
def load_pipeline(cfg, dtype=torch.float16):
    # ControlNet
    controlnet = ControlNetModel.from_pretrained(
        cfg["controlnet_path"], torch_dtype=dtype, variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(cfg["vae_path"], torch_dtype=torch.float16)
    # SDXL text-to-image pipeline
    base = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg["model_path"],
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
    )
    base.scheduler = EulerAncestralDiscreteScheduler.from_config(base.scheduler.config)
    base.enable_attention_slicing()
    base.vae.enable_tiling()

    # SDXL refiner pipeline for final quality pass
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        cfg["refiner_path"],
        torch_dtype=dtype,
    )
    refiner.scheduler = EulerAncestralDiscreteScheduler.from_config(refiner.scheduler.config)
    refiner.enable_attention_slicing()
    refiner.vae.enable_tiling()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    return base, refiner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="config.yaml",
        help="путь к YAML конфигурации",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    out_dir = create_output_dir()
    shutil.copy(args.config, out_dir / "config.yaml")

    if not torch.cuda.is_available():
        raise SystemError(
            "CUDA device not found. Install appropriate NVIDIA drivers and CUDA 12.1."
        )

    init_img  = preprocess(cfg["input"], cfg["size"])
    dbg(init_img,  "preprocessed.png", out_dir)

    edge_img  = canny(init_img, cfg["canny_low"], cfg["canny_high"])
    dbg(edge_img, "canny.png", out_dir)

    base, refiner = load_pipeline(cfg)
    size = sizeof_pipe(base) 
    free_mem = get_free_gpu_memory_gb()

    if free_mem > size:
        base.to('cuda')
        refiner.to('cuda')

    else:
        print(f"Not enough memory: {size - free_mem} GB. Enable sequential cpu offload...")
        print(f"VRAM neaded: {size} GB")
        print(f"Free GPU VRAM: {free_mem} GB")
        base.enable_sequential_cpu_offload()
        refiner.enable_sequential_cpu_offload()

    result = base(
        prompt=cfg["prompt"],
        image=init_img,
        control_image=edge_img,
        num_inference_steps=cfg["steps"],
        guidance_scale=cfg["guidance"],
        controlnet_conditioning_scale=cfg["conditioning"],
        negative_prompt=cfg["negative_prompt"]
    ).images[0]
    dbg(result, "base.png", out_dir)

    result = refiner(
        prompt=cfg["prompt"],
        image=result,
        num_inference_steps=cfg["refiner_steps"],
        denoising_start=cfg["refiner_start"],
        negative_prompt=cfg["negative_prompt"]
    ).images[0]
    result.save(out_dir / "result.png")

if __name__ == "__main__":
    main()
