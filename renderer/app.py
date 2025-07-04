"""High level generation routines."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from shutil import copy2

from .config import load_config
from .pipeline import (
    load_pipeline,
    load_loras,
    get_free_gpu_memory_gb,
    sizeof_pipe,
)
from .utils import preprocess, canny, save_debug


def generate(cfg: dict) -> None:
    """Run the full generation pipeline using a configuration dict."""
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    copy2("config.yaml", run_dir / "config.yaml")

    cfg.setdefault("output", str(run_dir / "output.png"))
    output_dir = Path(cfg["output"]).parent

    pipe = load_pipeline(cfg)
    load_loras(pipe, cfg)
    size = sizeof_pipe(pipe)
    free_mem = get_free_gpu_memory_gb()
    print(f"VRAM needed: {size} GB")
    print(f"Free GPU VRAM: {free_mem} GB")

    if free_mem > size:
        pipe.to("cuda")
    else:
        print(
            f"Not enough memory: {size - free_mem} GB. Enable sequential cpu offload..."
        )
        pipe.enable_sequential_cpu_offload()

    preprocessed = preprocess(cfg["input"], cfg.get("preprocess_size", 1024))
    save_debug(preprocessed, "preprocess.png", base=output_dir)

    edge_cfg = cfg.get("canny", {})
    edged = canny(preprocessed, edge_cfg.get("low", 100), edge_cfg.get("high", 200))
    save_debug(edged, "canny.png", base=output_dir)

    result = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        image=edged,
        controlnet_conditioning_scale=cfg.get("controlnet_conditioning_scale", 1.0),
        num_inference_steps=cfg.get("num_inference_steps", 20),
        guidance_scale=cfg.get("guidance_scale", 7.5),
    ).images[0]

    result.save(cfg.get("output", "output.png"))


def main() -> None:
    cfg = load_config()
    generate(cfg)


if __name__ == "__main__":
    main()
