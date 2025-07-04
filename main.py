import torch, numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler

def load_pipe():
  controlnet = ControlNetModel.from_pretrained(
      "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
  )
  pipe = StableDiffusionControlNetPipeline.from_pretrained(
      "SG161222/Realistic_Vision_V6.0_B1_noVAE",
      controlnet=controlnet, 
      safety_checker=None, 
      torch_dtype=torch.float16
  )
  vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16)
  pipe.vae = vae
  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)       # более стабильный scheduler
  return pipe

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

def canny(image: Image.Image, low: int, high: int) -> Image.Image:
    image = np.array(image)
    image = cv2.Canny(image, low, high)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)
  
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

def dbg(img: Image.Image, name: str, base: str):
    out_dir = Path(base)
    out_dir.mkdir(parents=True, exist_ok=False)
    img.save(out_dir / name)
  
pipe = load_pipe()
size = sizeof_pipe(pipe) 
free_mem = get_free_gpu_memory_gb()
print(f"VRAM neaded: {size} GB")
print(f"Free GPU VRAM: {free_mem} GB")

if free_mem > size:
    pipe.to('cuda')
else:
    print(f"Not enough memory: {size - free_mem} GB. Enable sequential cpu offload...")
    print(f"VRAM neaded: {size} GB")
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
# dbg(preprocess, 'preprocess.png', 'debug')
image=canny(preprocessed, 100, 200)
# dbg(preprocess, 'canny.png', 'debug')
result = pipe(
  prompt=prompt,
  negative_prompt=negative_prompt, 
  image=image, 
  controlnet_conditioning_scale=1.0, 
  num_inference_steps=20,
  guidance_scale=7.5
).images[0]

result.save("output.png")
