"""最终验收：生产路径 vs 基线（只用 median cut）。

基线 = pixelate 显式传 median cut 色板（等价于修复前的行为）
生产 = pixelate(palette=None)（median cut → refine_palette → ensure_neutral_highlight）

三个指标：
  Δsat       中性/亮部区域饱和度增量（越小越好 = 越不被染色）
  色相背叛率  亮度前 25% 像素里主色通道被翻转的比例（越小越好 = 越不"变绿"）
  err        全图量化误差（保真度，不能变差）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.palette import LUMA, from_perceptual, palette_from_image, to_perceptual  # noqa: E402
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.resample import auto_levels, structure_aware_downsample, tone_map, unsharp  # noqa: E402

SCALE, NCOL, LONG = 4, 32, 480


def hexs(c):
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat_of(c):
    m = c.reshape(-1, 3).mean(axis=0)
    return float((m.max() - m.min()) / max(m.max(), 1e-6))


def prep(p):
    src = Image.open(p).convert("RGB")
    gw = LONG if src.width >= src.height else round(src.width * LONG / src.height)
    gh = LONG if src.height > src.width else round(src.height * LONG / src.width)
    grid = (max(2, gw // 2 * 2), max(2, gh // 2 * 2))
    ref = src.resize((grid[0] * SCALE, grid[1] * SCALE), Image.Resampling.LANCZOS)
    pt = PixelTail(n_colors=NCOL)
    b = structure_aware_downsample(ref, grid, pt.var_gain)
    b = auto_levels(b, (1.0, 99.0), 0.9)
    b = tone_map(b, pt.tone_black, pt.tone_white, pt.tone_scurve)
    return unsharp(b, 0.55), pt


def measure(b, out):
    q = to_perceptual(b)
    lum = b @ LUMA
    mx, mn = b.max(axis=-1), b.min(axis=-1)
    M = (lum >= np.percentile(lum, 88)) & ((mx - mn) / np.clip(mx, 1e-6, None) <= 0.25)
    ds = sat_of(out[M]) - sat_of(b[M]) if M.sum() >= 50 else 0.0
    err = float(np.sqrt(((q - to_perceptual(out)) ** 2).sum(axis=2)).mean())
    bright = (lum >= np.percentile(lum, 75)).reshape(-1)
    flip = float((b.reshape(-1, 3).argmax(-1)[bright] != out.reshape(-1, 3).argmax(-1)[bright]).mean())
    return ds, err, flip, float(M.mean())


print(f"{'素材':24s} │ {'Δsat 旧 → 新':>18s} {'err Δ%':>8s} │ {'色相背叛 旧 → 新':>19s}")
print("─" * 84)
agg = {"ds0": [], "ds1": [], "e": [], "f0": [], "f1": []}
for f in sorted(INPUT.glob("*.jpg")):
    b, pt = prep(f)
    pal0 = palette_from_image(b, NCOL)
    u0, _ = pixelate(b, pt, palette=pal0)
    o0 = np.asarray(u0, np.float32) / 255.0
    u1, pal1 = pixelate(b, pt, palette=None)
    o1 = np.asarray(u1, np.float32) / 255.0

    ds0, e0, f0, mm = measure(b, o0)
    ds1, e1, f1, _ = measure(b, o1)
    agg["ds0"].append(ds0); agg["ds1"].append(ds1); agg["e"].append((e1 - e0) / e0)
    agg["f0"].append(f0); agg["f1"].append(f1)
    print(f"{f.stem:24s} │ {ds0:+.3f} → {ds1:+.3f} {((e1-e0)/e0*100):+7.2f}% │ "
          f"{f0*100:6.1f}% → {f1*100:5.1f}%   (色板 {len(pal0)}→{len(pal1)})")

print("─" * 84)
print(f"{'平均':24s} │ {np.mean(agg['ds0']):+.3f} → {np.mean(agg['ds1']):+.3f} "
      f"{np.mean(agg['e'])*100:+7.2f}% │ {np.mean(agg['f0'])*100:6.1f}% → {np.mean(agg['f1'])*100:5.1f}%")
print(f"\nΔsat 平均改善        {np.mean(agg['ds1']) - np.mean(agg['ds0']):+.3f}")
print(f"色相背叛率平均改善   {(np.mean(agg['f1']) - np.mean(agg['f0'])) * 100:+.2f} pp")
print(f"量化误差平均变化     {np.mean(agg['e'])*100:+.2f}%")
