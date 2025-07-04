"""Utility functions for preprocessing and debugging."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2


def canny(image: Image.Image, low: int, high: int) -> Image.Image:
    """Apply Canny edge detection and return an RGB image."""
    arr = np.array(image)
    arr = cv2.Canny(arr, low, high)
    arr = arr[:, :, None]
    arr = np.concatenate([arr, arr, arr], axis=2)
    return Image.fromarray(arr)


def preprocess(path: str | Path, size: int) -> Image.Image:
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


def save_debug(img: Image.Image, name: str, base: str | Path) -> None:
    """Save intermediate image next to the output."""
    out_dir = Path(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / name)


def save_image_fast(
    img: Image.Image,
    path: str | Path,
    compress_level: int = 1,
) -> None:
    """Save final image using OpenCV for faster compression."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(out_path), arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(out_path), arr, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
