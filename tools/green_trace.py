"""逐阶段追踪 ref10 建筑区的颜色 —— 定位"残留偏绿"到底来自哪一步。

已知：预处理后的 base 是中性偏灰（偏绿 +0.05），
但量化后的成片是明显偏绿（偏绿 +0.11）。
中间只差「雾 + 体积光」这两步，所以嫌疑就在那里。

同时量"全局染色"：体积光对**全图**平均色的影响，以及它对
不同深度/屏幕位置的注入量 —— 回答用户的"像上了一层全局滤镜"。
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
    bloom,
    depth_fog,
    limit_saturation,
    volumetric_light,
)
from pixelart.palette import LUMA, from_perceptual  # noqa: E402
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
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


p = TuneParams()
img = Image.open(INPUT / "ref10_green_cliff.jpg")
sc = build_scene(img, p)
gh, gw = sc.grid[1], sc.grid[0]

# 建筑区掩膜（明亮 + 低饱和），用 base 定
lum = sc.base @ LUMA
mx, mn = sc.base.max(-1), sc.base.min(-1)
M = (lum >= np.percentile(lum, 88)) & ((mx - mn) / np.clip(mx, 1e-6, None) <= 0.20)
print(f"网格 {gw}x{gh}   建筑区掩膜 {M.mean() * 100:.2f}%   光源 {[round(v, 2) for v in sc.light_xy]}")
print(f"雾色 {hexs(sc.fog_color)}  饱和度 {sat(sc.fog_color):.2f}\n")

print(f"{'阶段':26s} {'建筑区颜色':>10s} {'饱和度':>7s} {'偏绿':>7s} | "
      f"{'全图颜色':>10s} {'全图偏绿':>8s}")
print("-" * 78)


def row(tag, arr):
    print(f"{tag:26s} {hexs(arr[M]):>10s} {sat(arr[M]):7.3f} {grn(arr[M]):+7.3f} | "
          f"{hexs(arr):>10s} {grn(arr):+8.3f}")


row("① base（预处理后）", sc.base)

fogged = depth_fog(sc.base, sc.far, color=sc.fog_color, density=p.density, power=p.power)
row("② + 深度雾", fogged)

air = auto_scatter_color(fogged, *[int(v * (gw - 1)) if i == 0 else int(v * (gh - 1))
                                   for i, v in enumerate(sc.light_xy)],
                         sat_max=p.rays_sat)
print(f"\n   体积光的散射色 air_color = {hexs(air)}  饱和度 {sat(air):.3f}  "
      f"偏绿 {grn(air):+.3f}    ← 取自光源附近 7×7 中位数（再压饱和）")
print(f"   ⚠️ 注意：光源落在 (0.81, 0.19)，**那是一片植被**。"
      f"即使压过饱和，绿色仍然是绿色。\n")

vol = volumetric_light(fogged, sc.far, sc.light_xy, strength=p.rays, air_sat_max=p.rays_sat)
lit = np.clip(fogged + vol, 0, 1)
row("③ + 体积光", lit)

bloomed = bloom(lit, threshold=0.55, strength=p.bloom * 0.5, radii=(2, 5, 11))
row("④ + 辉光（= 合成结果）", bloomed)

tail = p.to_tail()
u8, pal = pixelate(bloomed, tail, palette=None)
row("⑤ 量化（像素尾巴）", np.asarray(u8, np.float32) / 255.0)

print("\n" + "=" * 78)
print("体积光对全图的染色贡献")
print("=" * 78)
print(f"  散射色 {hexs(air)} —— 整幅图都会叠加这个色相的光（按深度差衰减）")
print(f"  注入总量：建筑区 {vol[M].mean():.5f}   全图 {vol.mean():.5f}   比值 "
      f"{vol[M].mean() / max(vol.mean(), 1e-9):.2f}x")
c_air_norm = air / max(air.sum(), 1e-9)
print(f"  散射色归一化后 R/G/B = {c_air_norm[0]:.3f}/{c_air_norm[1]:.3f}/{c_air_norm[2]:.3f}"
      f"   （G 占比 {c_air_norm[1] * 100:.0f}%）")

print("\n" + "=" * 78)
print("对照：如果把散射色强制成中性（sat=0），建筑区会怎样")
print("=" * 78)
for s_max, label in ((0.25, "当前默认 sat_max=0.25"), (0.0, "强制中性 sat=0"), (None, "不压饱和（旧行为）")):
    kw = {} if s_max is None else {"air_sat_max": s_max}
    v2 = volumetric_light(fogged, sc.far, sc.light_xy, strength=p.rays, **kw)
    l2 = np.clip(fogged + v2, 0, 1)
    b2 = bloom(l2, threshold=0.55, strength=p.bloom * 0.5, radii=(2, 5, 11))
    u2, _ = pixelate(b2, tail, palette=None)
    o2 = np.asarray(u2, np.float32) / 255.0
    a2 = auto_scatter_color(fogged, int(sc.light_xy[0] * (gw - 1)),
                            int(sc.light_xy[1] * (gh - 1)),
                            sat_max=(1.0 if s_max is None else s_max))
    print(f"  {label:24s} 散射色 {hexs(a2)}  建筑区 {hexs(o2[M])}  "
          f"饱和度 {sat(o2[M]):.3f}  偏绿 {grn(o2[M]):+.3f}")
