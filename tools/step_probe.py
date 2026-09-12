"""分级统计探针 —— 逐个阶段打印直方图分位，用来定位"画面为什么发灰/发糊"。

像素尾巴是串联的，任何一步出问题都会表现为"结果不好看"，但原因可能完全不同：
降采样糊了？色阶没拉开？抖动触发了边缘压暗？量化把色阶浪费在暗部了？

这个脚本把每一步的 mean 与分位数打出来，一眼定位。

用法::

    python tools/step_probe.py assets/input/ref04_lain_room.jpg
    python tools/step_probe.py assets/input/ref05_veil_city.jpg --grid-long 480 --scale 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixelart.palette import LUMA, palette_from_image, to_perceptual  # noqa: E402
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.resample import (  # noqa: E402
    auto_levels,
    structure_aware_downsample,
    tone_map,
    unsharp,
)


def stat(tag: str, a: np.ndarray, prev: float | None = None) -> float:
    m = float(a.mean())
    q = np.percentile(a, [1, 25, 50, 75, 99])
    drift = "" if prev is None else f"   Δ={m - prev:+.3f}"
    print(f"  {tag:24s} mean={m:.3f}{drift:>10s}   p1/25/50/75/99 = " +
          " ".join(f"{v:.3f}" for v in q))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--grid-long", type=int, default=480)
    ap.add_argument("--scale", type=int, default=4, help="工作分辨率 = 网格 × 该倍数")
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--levels", type=float, default=0.9)
    ap.add_argument("--sharpen", type=float, default=0.55)
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGB")
    gw = args.grid_long if src.width >= src.height else round(src.width * args.grid_long / src.height)
    gh = args.grid_long if src.height > src.width else round(src.height * args.grid_long / src.width)
    grid = (max(2, gw // 2 * 2), max(2, gh // 2 * 2))
    ref = src.resize((grid[0] * args.scale, grid[1] * args.scale), Image.Resampling.LANCZOS)

    print(f"{Path(args.src).name}   原图 {src.size}   网格 {grid}   x{args.scale}")
    p = PixelTail(n_colors=args.colors, edge_gain=2.0)

    m = stat("ref(工作分辨率)", np.asarray(ref, np.float32) / 255.0)

    d = structure_aware_downsample(ref, grid, p.var_gain)
    m = stat("① 结构感知降采样", d, m)

    # 诊断：降采样的边缘保留权重有没有饱和
    a = np.asarray(ref, np.float32) / 255.0
    sq = Image.fromarray(np.clip(a * a * 255, 0, 255).astype(np.uint8))
    box = np.asarray(ref.resize(grid, Image.Resampling.BOX), np.float32) / 255.0
    sqb = np.asarray(sq.resize(grid, Image.Resampling.BOX), np.float32) / 255.0
    var = np.clip(sqb - box * box, 0, None).mean(axis=2)
    w = np.clip(var * p.var_gain, 0, 1)
    print(f"     ↳ 边缘权重 wgt 均值={w.mean():.3f} 饱和比例={float((w > 0.99).mean()):.3f}"
          f"   var 分位 50/90/99 = " + " ".join(f"{v:.5f}" for v in np.percentile(var, [50, 90, 99])))

    d = auto_levels(d, (1.0, 99.0), args.levels)
    m = stat("② +auto_levels", d, m)
    d = tone_map(d, p.tone_black, p.tone_white, p.tone_scurve)
    m = stat("③ +tone_map", d, m)
    d = unsharp(d, args.sharpen)
    m = stat("④ +unsharp", d, m)

    pal = palette_from_image(d, args.colors)
    print(f"     ↳ 色板 {len(pal)} 色")
    print("     ↳ edge_gain 扫描（目标：整体漂移 < 5%）:")
    for eg in (1.5, 2.0, 4.0, 7.0):
        u8_, _ = pixelate(d, PixelTail(n_colors=args.colors, edge_gain=eg), palette=pal)
        mm = float(np.asarray(u8_, np.float32).mean()) / 255.0
        print(f"        edge_gain={eg:<4} -> mean={mm:.3f}  漂移={mm - m:+.3f}")

    u8, _ = pixelate(d, p, palette=pal)
    stat("⑤ pixelate", np.asarray(u8, np.float32) / 255.0, m)

    lum = pal @ LUMA
    print(f"     ↳ 色板感知亮度 {lum.min():.3f}~{lum.max():.3f}，"
          f"其中 <0.2 的 {int((lum < 0.2).sum())} 个（过多说明暗部浪费了色阶）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
