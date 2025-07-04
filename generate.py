#!/usr/bin/env python3
"""Simple CLI for image generation using config defaults."""

import argparse
from pathlib import Path
from datetime import datetime
from shutil import copy2

from main import (
    load_config,
    load_pipeline,
    load_loras,
    get_free_gpu_memory_gb,
    sizeof_pipe,
    preprocess,
    canny,
    save_debug,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image from a sketch using ControlNet"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--input", help="Path to the input sketch image")
    parser.add_argument("--output", help="Where to save the result")
    parser.add_argument("--prompt", help="Prompt for generation")
    parser.add_argument("--negative-prompt", help="Negative prompt")
    parser.add_argument("--steps", type=int, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, help="Guidance scale")
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        help="ControlNet conditioning scale",
    )
    parser.add_argument("--preprocess-size", type=int, help="Preprocess image size")
    parser.add_argument("--canny-low", type=int, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, help="Canny high threshold")
    parser.add_argument("--debug-dir", help="Directory for debug images")
    return parser.parse_args()


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    if args.input:
        cfg["input"] = args.input
    if args.output:
        cfg["output"] = args.output
    if args.prompt:
        cfg["prompt"] = args.prompt
    if args.negative_prompt:
        cfg["negative_prompt"] = args.negative_prompt
    if args.steps is not None:
        cfg["num_inference_steps"] = args.steps
    if args.guidance is not None:
        cfg["guidance_scale"] = args.guidance
    if args.controlnet_scale is not None:
        cfg["controlnet_conditioning_scale"] = args.controlnet_scale
    if args.preprocess_size is not None:
        cfg["preprocess_size"] = args.preprocess_size
    if args.canny_low is not None or args.canny_high is not None:
        canny_cfg = cfg.get("canny", {})
        if args.canny_low is not None:
            canny_cfg["low"] = args.canny_low
        if args.canny_high is not None:
            canny_cfg["high"] = args.canny_high
        cfg["canny"] = canny_cfg
    if args.debug_dir:
        cfg["debug_dir"] = args.debug_dir
    return cfg


def run(cfg: dict, cfg_path: str, args: argparse.Namespace) -> None:
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    copy2(cfg_path, run_dir / "config.yaml")

    if args.debug_dir is None:
        cfg["debug_dir"] = str(run_dir / "debug")
    if args.output is None:
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
        print(
            f"Not enough memory: {size - free_mem} GB. Enable sequential cpu offload..."
        )
        pipe.enable_sequential_cpu_offload()

    preprocessed = preprocess(cfg["input"], cfg.get("preprocess_size", 1024))
    save_debug(preprocessed, "preprocess.png", base=cfg.get("debug_dir", "debug"))

    edge_cfg = cfg.get("canny", {})
    edged = canny(preprocessed, edge_cfg.get("low", 100), edge_cfg.get("high", 200))
    save_debug(edged, "canny.png", base=cfg.get("debug_dir", "debug"))

    result = pipe(
        prompt=cfg["prompt"],
        negative_prompt=cfg["negative_prompt"],
        image=edged,
        controlnet_conditioning_scale=cfg.get("controlnet_conditioning_scale", 1.0),
        num_inference_steps=cfg.get("num_inference_steps", 20),
        guidance_scale=cfg.get("guidance_scale", 7.5),
    ).images[0]

    result.save(cfg.get("output", "output.png"))


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)
    run(cfg, args.config, args)
