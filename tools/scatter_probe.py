"""标定体积散射色的饱和度上限 `air_sat_max`。

背景：旧代码直接取光源中心 7×7 的**均值**，在 ref10 上取到草丛，
得到饱和黄绿 (0.618, 0.793, 0.392)，叠上去整幅图变绿。

新代码取"窗口内较亮一半像素的中位数"，再压饱和度。
但 `sat_max` 取多少不能拍脑袋：

  - 取太小 → **暖色光场景（ref02 REPLACE 的暖灯）会被抽成灰白**，光晕失去颜色。
  - 取太大 → ref10 那种"光源落在有颜色物体上"的情况救不回来。

所以这里逐素材打印：
  旧散射色 / 新散射色（不压饱和）/ 各 sat_max 下的散射色，
  以及**扣掉散射色后画面的偏色方向**（这才是用户看得见的东西）。

另外单独看"暖色场景有没有被抽灰"：报告散射色的色相是否还保得住。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.compose import (  # noqa: E402
    auto_scatter_color,
    brightest_center,
    depth_equalize,
    depth_fog,
    limit_saturation,
    volumetric_light,
)
from pixelart.pixelate import PixelTail  # noqa: E402
from pixelart.resample import auto_levels, structure_aware_downsample, tone_map, unsharp  # noqa: E402

LONG, SCALE, DENSITY = 480, 4, 0.7


def hexs(c):
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat_of(c):
    c = np.asarray(c, np.float32).ravel()
    mx, mn = float(c.max()), float(c.min())
    return (mx - mn) / max(mx, 1e-6)


def prep(p):
    src = Image.open(p).convert("RGB")
    gw = LONG if src.width >= src.height else round(src.width * LONG / src.height)
    gh = LONG if src.height > src.width else round(src.height * LONG / src.width)
    grid = (max(2, gw // 2 * 2), max(2, gh // 2 * 2))
    ref = src.resize((grid[0] * SCALE, grid[1] * SCALE), Image.Resampling.LANCZOS)
    far = DepthEstimator().predict_far(ref)
    farg = depth_equalize(np.asarray(
        Image.fromarray(far.astype(np.float32), mode="F").resize(grid, Image.Resampling.BOX),
        np.float32), strength=0.5)
    pt = PixelTail(n_colors=32)
    b = structure_aware_downsample(ref, grid, pt.var_gain)
    b = auto_levels(b, (1.0, 99.0), 0.9)
    b = tone_map(b, pt.tone_black, pt.tone_white, pt.tone_scurve)
    return unsharp(b, 0.55), farg


def old_scatter(rgb01, px, py, radius=3):
    patch = rgb01[max(0, py - radius):py + radius + 1, max(0, px - radius):px + radius + 1]
    return np.clip(patch.reshape(-1, 3).mean(axis=0), 0, 1).astype(np.float32)


CAPS = (0.15, 0.25, 0.35, 0.50)

print("旧散射色 = 光源中心 7×7 均值；新 = 亮半中位数 + 压饱和\n")
hdr = f"{'素材':24s} {'光源位置':>11s} {'旧散射色':>22s}"
for c in CAPS:
    hdr += f"  {'新@'+str(c):>22s}"
print(hdr)
print("─" * 150)

rows = []
for f in sorted(INPUT.glob("*.jpg")):
    b, farg = prep(f)
    fogged = depth_fog(b, farg, density=DENSITY, power=3.0)
    cx, cy = brightest_center(fogged, 3.0)
    px, py = int(cx * (b.shape[1] - 1)), int(cy * (b.shape[0] - 1))

    o = old_scatter(fogged, px, py)
    line = f"{f.stem:24s} {cx:5.2f},{cy:4.2f}  {hexs(o)} sat={sat_of(o):.2f}"
    news = {}
    for c in CAPS:
        n = auto_scatter_color(fogged, px, py, radius=3, sat_max=c)
        news[c] = n
        line += f"  {hexs(n)} sat={sat_of(n):.2f}"
    rows.append((f.stem, fogged, farg, (px, py), o, news))
    print(line)

print("\n" + "=" * 150)
print("扣掉散射色后，全图平均色的**偏色方向**（旧 vs 各 sat_max）")
print("（散射色是加性图层，这里用「散射色本身」的色相来代表它会往画面里注入什么颜色）")
print("=" * 150)
print(f"{'素材':24s} {'旧注入色':>10s} {'旧偏绿':>8s}   " + "".join(f"{'@'+str(c):>10s}" for c in CAPS))
print("─" * 100)
for name, fogged, farg, (px, py), o, news in rows:
    gi_old = float(o[1] - 0.5 * (o[0] + o[2]))
    line = f"{name:24s} {hexs(o):>10s} {gi_old:+8.3f}   "
    for c in CAPS:
        n = news[c]
        gi = float(n[1] - 0.5 * (n[0] + n[2]))
        line += f"{gi:+10.3f}"
    print(line)

# 暖色场景是否被抽灰：看色相还在不在
print("\n" + "=" * 150)
print("暖色场景检验：散射色的「暖度」(R-B)，负数=变冷。收得太狠会把暖灯抽成灰白。")
print("=" * 150)
print(f"{'素材':24s} {'旧 R-B':>8s}   " + "".join(f"{'@'+str(c):>10s}" for c in CAPS))
print("─" * 100)
for name, fogged, farg, (px, py), o, news in rows:
    line = f"{name:24s} {(o[0] - o[2]):+8.3f}   "
    for c in CAPS:
        n = news[c]
        line += f"{(n[0] - n[2]):+10.3f}"
    print(line)

print("\n=== limit_saturation 自身行为核对 ===")
for c in ([0.618, 0.793, 0.392], [0.90, 0.65, 0.30], [0.5, 0.5, 0.5], [0.02, 0.02, 0.02]):
    a = np.array(c, np.float32)
    out = {s: limit_saturation(a, s) for s in CAPS}
    print(f"  输入 {hexs(a)} sat={sat_of(a):.2f} -> " +
          "  ".join(f"@{s}: {hexs(v)} sat={sat_of(v):.2f} lum={v.mean():.3f}" for s, v in out.items()))
