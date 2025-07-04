from flask import Flask, request, render_template, send_file
from io import BytesIO
from PIL import Image
import torch

from main import (
    load_config,
    load_pipeline,
    load_loras,
    sizeof_pipe,
    get_free_gpu_memory_gb,
    generate,
)

app = Flask(__name__)

cfg = load_config()
pipe = load_pipeline(cfg)
load_loras(pipe, cfg)
size = sizeof_pipe(pipe)
free_mem = get_free_gpu_memory_gb()
print(f"VRAM needed: {size} GB")
print(f"Free GPU VRAM: {free_mem} GB")
if torch.cuda.is_available() and free_mem > size:
    pipe.to("cuda")
else:
    pipe.enable_sequential_cpu_offload()

@app.route("/")
def index():
    return render_template("index.html", cfg=cfg)

@app.route("/generate", methods=["POST"])
def generate_route():
    local_cfg = cfg.copy()
    local_cfg["prompt"] = request.form.get("prompt", cfg["prompt"])
    local_cfg["negative_prompt"] = request.form.get(
        "negative_prompt", cfg["negative_prompt"]
    )
    local_cfg["num_inference_steps"] = int(
        request.form.get("steps", cfg.get("num_inference_steps", 20))
    )
    local_cfg["guidance_scale"] = float(
        request.form.get("guidance", cfg.get("guidance_scale", 7.5))
    )
    local_cfg["controlnet_conditioning_scale"] = float(
        request.form.get(
            "scale", cfg.get("controlnet_conditioning_scale", 1.0)
        )
    )
    low = int(request.form.get("canny_low", 100))
    high = int(request.form.get("canny_high", 200))
    local_cfg["canny"] = {"low": low, "high": high}

    file = request.files.get("image")
    if file:
        image = Image.open(file.stream).convert("RGB")
    else:
        image = cfg["input"]

    result = generate(pipe, local_cfg, image=image)
    buf = BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
