"""散射色上限的可视化对照：三种 sat_max 下体积光注入的差异。

只看指标不够，这里直接放大**体积光图层本身**（它才是被污染的东西）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402
SRC = INPUT / "ref10_green_cliff.jpg"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.compose import (  # noqa: E402
    brightest_center,
    depth_equalize,
    depth_fog,
    volumetric_light,
)
from pixelart.paths import out  # noqa: E402
from pixelart.pixelate import PixelTail  # noqa: E402
from pixelart.resample import auto_levels, structure_aware_downsample, tone_map, unsharp  # noqa: E402

K, DENSITY, RAYS = 4, 0.7, 0.55
CASES = [(1.0, "旧：7×7 均值、不压饱和"), (0.15, "sat_max 0.15（过狠）"),
         (0.25, "sat_max 0.25（标定值）"), (0.0, "sat_max 0（完全不着色）")]


def prep(p):
    src = Image.open(p).convert("RGB")
    gw, gh = round(src.width * 480 / src.height), 480
    grid = (max(2, gw // 2 * 2), max(2, gh // 2 * 2))
    ref = src.resize((grid[0] * K, grid[1] * K), Image.Resampling.LANCZOS)
    far = DepthEstimator().predict_far(ref)
    farg = depth_equalize(np.asarray(
        Image.fromarray(far.astype(np.float32), mode="F").resize(grid, Image.Resampling.BOX),
        np.float32), strength=0.5)
    pt = PixelTail(n_colors=32)
    b = structure_aware_downsample(ref, grid, pt.var_gain)
    b = auto_levels(b, (1.0, 99.0), 0.9)
    b = tone_map(b, pt.tone_black, pt.tone_white, pt.tone_scurve)
    return unsharp(b, 0.55), farg


base, farg = prep(SRC)
fogged = depth_fog(base, farg, density=DENSITY, power=3.0)
cx, cy = brightest_center(fogged, 3.0)

print("体积光图层统计（放大 4 倍显示，另附注入的色相）")
print(f"{'配置':24s} {'散射色':>9s} {'饱和':>6s} {'偏绿':>7s} {'注入量':>8s} {'暖度R-B':>8s}")
print("-" * 76)
tiles = []
for sm, label in CASES:
    vol = volumetric_light(fogged, farg, (cx, cy), strength=RAYS, air_sat_max=sm)
    tot = float(vol.sum())
    c = vol.sum(axis=(0, 1)) / max(tot, 1e-9)
    air = volumetric_light(fogged, farg, (cx, cy), strength=1.0, air_sat_max=sm)
    print(f"{label:24s} {'':>9s} {'':>6s} {float(c[1] - 0.5 * (c[0] + c[2])):+7.3f} "
          f"{float(vol.mean()):8.5f} {float(c[0] - c[2]):+8.3f}")

    # 可视化：按各自最大值归一化，这样看到的是**颜色**而不是强度
    t = Image.fromarray(np.rint(np.clip(vol / max(float(vol.max()), 1e-9), 0, 1) * 255).astype(np.uint8))
    tiles.append((label, t))

print("\n注：上面每格按各自最大值归一化 —— 看的是**颜色**（它会往画面里注入什么色），")
print("    不是强度。所以四格的亮度看起来接近，但色相差别很明显。")

cw = tiles[0][1].width + 20
ch = tiles[0][1].height + 46
sheet = Image.new("RGB", (cw * 2, ch * 2), (8, 9, 13))
d = ImageDraw.Draw(sheet)
for i, (label, im) in enumerate(tiles):
    bx, by = (i % 2) * cw, (i // 2) * ch
    sheet.paste(im, (bx + 10, by + 40))
    d.text((bx + 12, by + 12), label, font=C.font(24), fill=C.LABEL_FG)
C.save(sheet, out("m1", "scatter_ab.png"))
print("\n[ok] out/m1/scatter_ab.png")
