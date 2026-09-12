"""系统性检查：色板"中性色缺失"在 10 张素材上到底有多普遍。

对每张图：
1. 在**原始图**上找出"明亮 + 低饱和"像素（建筑/白墙/白雪那类），记下它们的真实颜色。
2. 走一遍预处理 → 量化，量出这批像素**量化后**的颜色。
3. 报告：饱和度增量（被染色的程度）、色相偏移方向、以及色板里最亮中性项的"中性度"。

> 判据：如果量化后这批像素的饱和度显著上升、且均值偏到某个色相，
>   就是「中和色被涂成彩色」。ref10 的白楼变绿是典型。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.palette import (  # noqa: E402
    LUMA,
    ensure_neutral_highlight,
    from_perceptual,
    palette_from_image,
    to_perceptual,
)
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.resample import (  # noqa: E402
    auto_levels,
    structure_aware_downsample,
    tone_map,
    unsharp,
)

GRID_LONG, SCALE, NCOL = 480, 4, 32


def hexs(c: np.ndarray) -> str:
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat_of(c: np.ndarray) -> float:
    m = c.reshape(-1, 3).mean(axis=0)
    mx, mn = m.max(), m.min()
    return float((mx - mn) / max(mx, 1e-6))


print(f"{'素材':28s} {'中性像素':>7s} {'原饱和度':>7s} {'量化后':>7s} {'Δsat':>7s} "
      f"{'色相偏移':>10s} {'色板最亮中性项':>14s}")
print("-" * 100)

for path in sorted(INPUT.glob("*.jpg")):
    src = Image.open(path).convert("RGB")
    full = np.asarray(src, np.float32) / 255.0

    # 在**原始全分辨率**上找明亮低饱和像素 = 真实的中性色区域
    lum = full @ LUMA
    mx, mn = full.max(axis=-1), full.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1e-6, None)
    mask = (lum >= np.percentile(lum, 92)) & (sat <= 0.16)
    if mask.sum() < 200:
        print(f"{path.stem:28s}  (无足够中性亮区，跳过)")
        continue

    true_c = full[mask].mean(axis=0)

    # 走管线
    gw = GRID_LONG if src.width >= src.height else round(src.width * GRID_LONG / src.height)
    gh = GRID_LONG if src.height > src.width else round(src.height * GRID_LONG / src.width)
    grid = (max(2, gw // 2 * 2), max(2, gh // 2 * 2))
    ref = src.resize((grid[0] * SCALE, grid[1] * SCALE), Image.Resampling.LANCZOS)

    p = PixelTail(n_colors=NCOL)
    base = structure_aware_downsample(ref, grid, p.var_gain)
    base = auto_levels(base, (1.0, 99.0), 0.9)
    base = tone_map(base, p.tone_black, p.tone_white, p.tone_scurve)
    base = unsharp(base, 0.55)

    u8, pal = pixelate(base, p, palette=None)
    got = np.asarray(u8, np.float32)[..., ::-1] / 255.0   # u8 是 RGB，np.asarray 后也是 RGB
    got = np.asarray(u8, np.float32) / 255.0

    # 把这些中性区**下采样到网格**后取同一批位置
    mimg = Image.fromarray((mask * 255).astype(np.uint8))
    msmall = np.asarray(mimg.resize(grid, Image.Resampling.BOX), np.float32) / 255.0 > 0.5
    if msmall.sum() < 20:
        print(f"{path.stem:28s}  (网格上中性区太小，跳过)")
        continue

    c_true = base[msmall].mean(axis=0)
    c_got = got[msmall].mean(axis=0)

    d_sat = sat_of(c_got) - sat_of(c_true)
    # 色相偏移方向：量化后相对预处理的偏色方向
    delta = c_got - c_true
    dom = ["R", "G", "B"][int(np.argmax(delta))]
    hue = f"{dom}{delta.max():+.3f}"

    # 色板里最亮的中性项
    palx = ensure_neutral_highlight(base, palette_from_image(base, NCOL))
    pl = palx @ LUMA
    pax = palx.max(axis=1)
    pnn = palx.min(axis=1)
    psat = (pax - pnn) / np.clip(pax, 1e-6, None)
    neutral = np.where(psat <= 0.35)[0]
    if len(neutral):
        j = neutral[np.argmax(pl[neutral])]
        cn = from_perceptual(palx[j])
        nmark = f"{hexs(cn)} sat={psat[j]:.2f}"
    else:
        nmark = "无中性项"

    print(f"{path.stem:28s} {mask.mean() * 100:6.1f}% {sat_of(c_true):7.3f} "
          f"{sat_of(c_got):7.3f} {d_sat:+7.3f} {hue:>10s} {nmark:>14s}")

print()
print("说明：Δsat > 0 表示中性色被染上颜色；数值越大越明显。")
print("      原饱和度是**预处理后**（base）的值，排除了 auto_levels 的影响。")
