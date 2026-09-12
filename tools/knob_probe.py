"""旋钮探测 —— 拆解像素尾巴里各步骤各自的贡献。

固定一个网格，只改变处理方式，看差异到底从哪来：

    A  朴素 lanczos 缩小（绝大多数「像素滤镜」的做法）
    B  结构感知降采样 + 影调映射
    C  再加调色板量化 + 亮度自适应有序抖动   ← 推荐尾巴

结论：抖动才是把「海报化照片」变成「像素画」的那一步；
      结构感知降采样负责不糊、保住轮廓。

用法::

    python tools/knob_probe.py
    python tools/knob_probe.py --grid 480x270 --colors 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.palette import from_perceptual, palette_from_image, snap_exact  # noqa: E402
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.resample import (  # noqa: E402
    fit_to_aspect,
    structure_aware_downsample,
    tone_map,
)

DETAIL_BOX = (760, 280, 1380, 900)


def parse_grid(s: str) -> tuple[int, int]:
    w, _, h = s.partition("x")
    return int(w), int(h)


def naive_downsample(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """朴素做法：直接 lanczos 缩小。"""
    return np.asarray(img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def quantize_only(f: np.ndarray, n_colors: int) -> np.ndarray:
    """只量化、不抖动。"""
    pal = palette_from_image(f, n_colors)
    return from_perceptual(snap_exact(np.sqrt(np.clip(f, 0, 1)), pal))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--grid", default="480x270")
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--target-width", type=int, default=1920)
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    w, h = grid
    k = args.target_width // w

    src = fit_to_aspect(Image.open(args.src).convert("RGB"), 16 / 9)
    ref = src.resize((1920, 1080), Image.Resampling.LANCZOS)
    p = PixelTail(n_colors=args.colors)

    struct = tone_map(structure_aware_downsample(ref, grid, p.var_gain),
                      p.tone_black, p.tone_white, p.tone_scurve)

    variants: list[tuple[str, np.ndarray]] = [
        ("A  朴素缩放 + 量化（典型滤镜）",
         quantize_only(naive_downsample(ref, grid), args.colors)),
        ("B  结构感知 + 影调映射 + 量化",
         quantize_only(struct, args.colors)),
        ("C  再加上亮度自适应有序抖动  ← 推荐",
         from_perceptual(_apply_tail(struct, p))),
    ]

    x0, y0, x1, y1 = DETAIL_BOX
    pw, ph = x1 - x0, y1 - y0

    sheet = Image.new("RGB", (pw * len(variants), ph + 52), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    for i, (title, arr) in enumerate(variants):
        small = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        big = small.resize((w * k, h * k), Image.Resampling.NEAREST)
        sheet.paste(big.crop((x0, y0, x1, y1)), (i * pw, 52))
        d.text((i * pw + 14, 12), title, font=C.font(24), fill=C.LABEL_FG)

    dst = out("knob", f"knobs_{args.grid}.png")
    C.save(sheet, dst)
    print(f"[ok] {dst}")

    # 再存一份全图对比
    full = Image.new("RGB", (1920, 1080), (8, 9, 13))
    for i, (title, arr) in enumerate(variants):
        small = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        big = small.resize((1920, 1080), Image.Resampling.NEAREST)
        full.paste(big.resize((960, 540), Image.Resampling.NEAREST), ((i % 2) * 960, (i // 2) * 540))
    C.save(full, out("knob", f"knobs_{args.grid}_full.png"))
    return 0


def _apply_tail(f: np.ndarray, p: PixelTail) -> np.ndarray:
    """复用包里的实现，保证与正式管线完全一致。"""
    pal = palette_from_image(f, p.n_colors)
    u8, _ = pixelate(f, p, palette=pal)
    return np.asarray(u8, dtype=np.float32) / 255.0


if __name__ == "__main__":
    raise SystemExit(main())
