"""两个候选修复的量化评测。

A. 「量化把建筑区染色」—— 是色板槽位不够，还是 refine 的赎回额度不够？
   ref10 的建筑只占约 4% 像素，median cut 按像素数量分配 → 它拿不到自己的色阶。
   两个可调旋钮：色板颜色数、refine 的赎回上限。

B. 「像蒙了一层全局滤镜」—— 嫌疑是**雾色饱和度**。
   实测 ref10 的自动雾色是 #2381a6（饱和度 0.79），非常饱和。
   真实的空气透视是**降饱和**的（散射把颜色洗掉），所以拿一张饱和蓝天当雾色，
   叠到远景上就是"蒙了一层蓝滤镜"。
   另外体积光的散射色也是全局按深度叠加的 —— 一起量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.compose import (  # noqa: E402
    auto_scatter_color,
    auto_fog_color,
    bloom,
    depth_fog,
    limit_saturation,
    volumetric_light,
)
from pixelart.palette import LUMA, from_perceptual  # noqa: E402
from pixelart.pixelate import pixelate  # noqa: E402
from pixelart.tune import TuneParams, build_scene  # noqa: E402



def hexs(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(axis=0)
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(axis=0)
    return float((c.max() - c.min()) / max(c.max(), 1e-6))


def grn(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(axis=0)
    return float(c[1] - 0.5 * (c[0] + c[2]))


def region(sc, arr, q=88.0, satmax=0.20):
    lum = sc.base @ LUMA
    mx, mn = sc.base.max(-1), sc.base.min(-1)
    return (lum >= np.percentile(lum, q)) & ((mx - mn) / np.clip(mx, 1e-6, None) <= satmax)


# ══════════════════════════════════════════════════════════════
print("=" * 78)
print("A. 量化染色 vs 色板容量")
print("=" * 78)
print("全部 10 张素材的自动雾色饱和度（看全局染色的普遍性）")
print(f"{'素材':30s} {'雾色':>9s} {'饱和度':>7s}")
print("-" * 52)
for f in sorted(INPUT.glob("*.jpg")):
    p = TuneParams(grid_long=160, work_long=640)
    sc = build_scene(Image.open(f), p)
    fc = auto_fog_color(sc.base, sc.far)
    print(f"{f.stem:30s} {hexs(fc):>9s} {sat(fc):7.3f}")

print()
print("=" * 78)
print("B. ref10 建筑区染色 vs 色板颜色数")
print("=" * 78)
img10 = Image.open(INPUT / "ref10_green_cliff.jpg")
p0 = TuneParams()
sc = build_scene(img10, p0)
M = region(sc, sc.base)
print(f"  建筑区真实色（预处理后）：{hexs(sc.base[M])}  饱和度 {sat(sc.base[M]):.3f}  "
      f"偏绿 {grn(sc.base[M]):+.3f}")
print()
print(f"{'色板颜色数':>10s} {'建筑区平均色':>14s} {'饱和度':>8s} {'偏绿':>8s} {'色板中性项':>10s}")
print("-" * 58)
for ncol in (16, 24, 32, 48, 64):
    p = TuneParams(colors=ncol)
    p.grid_long = p0.grid_long
    p.work_long = p0.work_long
    fogged = depth_fog(sc.base, sc.far, color=sc.fog_color, density=p.density, power=p.power)
    vol = volumetric_light(fogged, sc.far, sc.light_xy, strength=p.rays, air_sat_max=p.rays_sat)
    lit = np.clip(fogged + vol, 0, 1)
    bl = bloom(lit, threshold=0.55, strength=p.bloom * 0.5, radii=(2, 5, 11))
    u8, pal = pixelate(bl, p.to_tail(), palette=None)
    out = np.asarray(u8, np.float32) / 255.0
    cp = from_perceptual(pal)
    psat = (cp.max(1) - cp.min(1)) / np.clip(cp.max(1), 1e-6, None)
    print(f"{ncol:10d} {hexs(out[M]):>14s} {sat(out[M]):8.3f} {grn(out[M]):+8.3f} "
          f"{int((psat <= 0.12).sum()):10d}")

print()
print("=" * 78)
print("C. 雾色降饱和 × 全图染色")
print("=" * 78)
print("auto_fog_color 现在只做 25% 降饱和。真实的空气透视应当更接近灰。")
print()
fog_raw = auto_fog_color(sc.base, sc.far)
print(f"{'降饱和':>8s} {'雾色':>9s} {'饱和':>6s} | {'全图偏绿':>8s} {'远景偏绿':>8s} {'远景饱和':>8s}")
print("-" * 62)
far_m = sc.far >= np.quantile(sc.far, 0.85)
for extra in (0.0, 0.25, 0.5, 0.65, 0.8):
    c = fog_raw.astype(np.float32)
    if extra > 0:
        c = c * (1 - extra) + float(c.mean()) * extra
    fogged = depth_fog(sc.base, sc.far, color=c, density=p0.density, power=p0.power)
    print(f"{extra:8.2f} {hexs(c):>9s} {sat(c):6.3f} | {grn(fogged):+8.3f} "
          f"{grn(fogged[far_m]):+8.3f} {sat(fogged[far_m]):8.3f}")

print()
print("=" * 78)
print("D. 体积光：屏幕空间受限 vs 只按深度")
print("=" * 78)
p = TuneParams()
fogged = depth_fog(sc.base, sc.far, color=sc.fog_color, density=p.density, power=p.power)
gh, gw = sc.grid[1], sc.grid[0]
yy, xx = np.mgrid[0:gh, 0:gw]
dist = np.hypot(xx / gw - sc.light_xy[0], yy / gh - sc.light_xy[1])

vol = volumetric_light(fogged, sc.far, sc.light_xy, strength=p.rays, air_sat_max=p.rays_sat)
print(f"  现状：全图平均注入 {vol.mean():.5f}")
for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.5)):
    m = (dist >= lo) & (dist < hi)
    if m.sum():
        print(f"    屏幕距离 {lo:.2f}~{hi:.2f}：占画面 {m.mean() * 100:5.1f}%  "
              f"注入 {vol[m].mean():.5f}  （占总量 {vol[m].sum() / vol.sum() * 100:4.1f}%）")

print()
print("  如果再加一层**屏幕空间**衰减（只让靠近光源的像素吃到光）：")
for power in (0.0, 1.0, 2.0):
    if power == 0:
        v2 = vol
        tag = "不加（现状）"
    else:
        fall = 1.0 / (1.0 + (dist * 4.0) ** power)
        v2 = vol * fall[..., None]
        tag = f"屏幕衰减 {power:g}"
    print(f"    {tag:16s} 全图平均注入 {v2.mean():.5f}  "
          f"近处占比 {v2[dist < 0.25].sum() / max(v2.sum(), 1e-9) * 100:4.1f}%")
