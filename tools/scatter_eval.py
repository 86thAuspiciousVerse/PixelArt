"""体积散射色修复的**最终验收** —— 看对最终画面的实际影响。

只看加性层的颜色是不够的（注入量还受 strength / falloff 影响），
这里直接量**合成后的画面**：

  1. 散射色本身：饱和度、偏绿指数、暖度（R−B）
  2. 合成后画面的**色偏变化**（相对不加体积光）：Δ偏绿 / Δ暖度
     这是用户真正看得见的东西 —— 加体积光会不会把画面染绿。
  3. 体积光的注入量：不能因为压饱和把"光感"也压没了。

对照：旧行为（7×7 均值，不压饱和）vs 新行为（全窗口中位数 + 压饱和）。
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
    LUMA,
)
from pixelart.pixelate import PixelTail  # noqa: E402
from pixelart.resample import auto_levels, structure_aware_downsample, tone_map, unsharp  # noqa: E402

LONG, SCALE, DENSITY, RAYS = 480, 4, 0.7, 0.55
SAT_MAX = 0.25


def hexs(c):
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat_of(c):
    c = np.asarray(c, np.float32).ravel()
    return (float(c.max()) - float(c.min())) / max(float(c.max()), 1e-6)


def green_of(c):
    c = np.asarray(c, np.float32).ravel()
    return float(c[1] - 0.5 * (c[0] + c[2]))


def warm_of(c):
    c = np.asarray(c, np.float32).ravel()
    return float(c[0] - c[2])


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


def old_scatter(rgb, px, py, radius=3):
    patch = rgb[max(0, py - radius):py + radius + 1, max(0, px - radius):px + radius + 1]
    return np.clip(patch.reshape(-1, 3).mean(axis=0), 0, 1).astype(np.float32)


def vol_with(rgb, far, xy, color, amount=0.55):
    """用指定散射色算体积光图层（复用生产函数，显式传入 air_color）。"""
    return volumetric_light(rgb, far, xy, strength=amount, air_color=color)


print(f"sat_max = {SAT_MAX}   （旧行为 = 7×7 均值、不压饱和）\n")
print(f"{'素材':24s} │ {'散射色 旧 → 新':>22s} {'饱和度':>10s} │ "
      f"{'Δ偏绿(合成后)':>16s} {'Δ暖度(合成后)':>16s} {'注入量比':>9s}")
print("─" * 122)

agg = {"dg": [], "dw": [], "amt": [], "dsat": []}
for f in sorted(INPUT.glob("*.jpg")):
    b, farg = prep(f)
    fogged = depth_fog(b, farg, density=DENSITY, power=3.0)
    cx, cy = brightest_center(fogged, 3.0)
    px, py = int(cx * (b.shape[1] - 1)), int(cy * (b.shape[0] - 1))

    c_old = old_scatter(fogged, px, py)
    c_new = auto_scatter_color(fogged, px, py, radius=3, sat_max=SAT_MAX)

    v_old = vol_with(fogged, farg, (cx, cy), c_old, RAYS)
    v_new = vol_with(fogged, farg, (cx, cy), c_new, RAYS)

    gi_o, gi_n = green_of((fogged + v_old).sum(axis=(0, 1)) / v_old.size), green_of((fogged + v_new).sum(axis=(0, 1)) / v_new.size)
    # 用整体均值色的偏移衡量"是否把画面染绿/变冷"
    base_mean = fogged.mean(axis=(0, 1))
    d_old_g = green_of((fogged + v_old).mean(axis=(0, 1))) - green_of(base_mean)
    d_new_g = green_of((fogged + v_new).mean(axis=(0, 1))) - green_of(base_mean)
    d_old_w = warm_of((fogged + v_old).mean(axis=(0, 1))) - warm_of(base_mean)
    d_new_w = warm_of((fogged + v_new).mean(axis=(0, 1))) - warm_of(base_mean)

    ratio = float(v_new.mean()) / max(float(v_old.mean()), 1e-9)
    agg["dg"].append(d_new_g - d_old_g)
    agg["dw"].append(d_new_w - d_old_w)
    agg["amt"].append(ratio)
    agg["dsat"].append(sat_of(c_new) - sat_of(c_old))

    print(f"{f.stem:24s} │ {hexs(c_old)} → {hexs(c_new)} {sat_of(c_old):5.2f}→{sat_of(c_new):4.2f} │ "
          f"{d_old_g:+.4f}→{d_new_g:+.4f}  {d_old_w:+.4f}→{d_new_w:+.4f} {ratio:8.2f}")

print("─" * 122)
print(f"{'平均':24s} │ {'':22s} {np.mean(agg['dsat']):+9.3f} │ "
      f"{'':16s} {'':16s} {np.mean(agg['amt']):8.2f}")
print(f"\n偏绿注入平均变化      {np.mean(agg['dg']):+.5f}   （负数 = 修复后更不绿）")
print(f"暖度注入平均变化      {np.mean(agg['dw']):+.5f}   （≈0 = 暖色场景未被误伤）")
print(f"体积光注入量保持      {np.mean(agg['amt']) * 100:.1f}%  （≈100% = 光感没被压掉）")
