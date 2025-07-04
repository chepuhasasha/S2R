#!/usr/bin/env python3
"""Simple CLI for image generation using config defaults."""
import argparse

from renderer.config import load_config
from renderer.app import generate


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)
    generate(cfg)


if __name__ == "__main__":
    main()
