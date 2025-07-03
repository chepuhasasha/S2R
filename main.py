# main.py  —  проверено 02-Jul-2025, diffusers 0.34.0, torch 2.7.1+cu121

from pathlib import Path
import argparse, torch, cv2, numpy as np
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)

MODEL_PATH      = "./models/stable-diffusion-xl-base-1.0"
CONTROLNET_PATH = "./models/controlnet-canny-sdxl-1.0"   # 2.1 GB fp16
SIZE            = 1024
CANNY_LOW       = 5
CANNY_HIGH      = 160
STEPS           = 30
STRENGTH        = 0.80
GUIDANCE        = 8.0
DEBUG_DIR       = "debug"          # "debug" чтобы сохранять промежуточные картинки

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


def dbg(img: Image.Image, name: str):
    if DEBUG_DIR:
        Path(DEBUG_DIR).mkdir(exist_ok=True)
        img.save(Path(DEBUG_DIR) / name)

def preprocess(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > SIZE:
        s = SIZE / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.NEAREST)
    pad_w = (SIZE - img.width) // 2
    pad_h = (SIZE - img.height) // 2
    return ImageOps.expand(
        img,
        border=(pad_w, pad_h, SIZE - img.width - pad_w, SIZE - img.height - pad_h),
        fill="white",
    )

def canny(img: Image.Image) -> Image.Image:
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    e = cv2.Canny(g, CANNY_LOW, CANNY_HIGH)
    return Image.fromarray(cv2.cvtColor(e, cv2.COLOR_GRAY2RGB))

def load_pipeline(dtype=torch.float16):
    # ControlNet
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH, torch_dtype=dtype, variant="fp16"
    )

    # SDXL-pipeline целиком на CPU
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        MODEL_PATH,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="скетч PNG/JPG")
    ap.add_argument("--output", required=True, help="PNG-файл рендера")
    ap.add_argument("--prompt", required=True, help="текстовый промпт")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemError(
            "CUDA device not found. Install appropriate NVIDIA drivers and CUDA 12.1."
        )

    init_img  = preprocess(args.input)
    dbg(init_img,  "preprocessed.png")

    edge_img  = canny(init_img)
    dbg(edge_img, "canny.png")

    pipe = load_pipeline()
    size = sizeof_pipe(pipe)
    free_mem = get_free_gpu_memory_gb()

    if free_mem > size:
        result = pipe(
            prompt=args.prompt,
            image=init_img,
            control_image=edge_img,
            num_inference_steps=STEPS,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
        ).images[0]

        result.save(args.output)
    else:
        print(f"Not enough memory: {size - free_mem} GB")
        print(f"VRAM neaded: {size} GB")
        print(f"Free GPU VRAM: {free_mem} GB")


if __name__ == "__main__":
    main()
