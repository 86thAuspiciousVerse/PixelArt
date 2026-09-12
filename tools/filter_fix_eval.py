"""验收：两个「全局滤镜感」修复的横向效果。

修复一：自动雾色加饱和度上限（0.40）
修复二：体积光加屏幕空间收敛（默认 1.0）

两者都是"只对病态情况生效"的设计，所以判据是**两头看**：

  · 该改善的要改善 —— 雾色过饱和的样本，远景染色应减轻
  · **不该动的不能动** —— 本来就正常的样本，画面必须基本不变
    （否则就是"为了修一个 bug 把好画面也改了"）

指标：
  fog_sat    实际用到的雾色饱和度
  far_grn    远景（深度 q85 以上）平均偏绿指数
  far_sat    远景平均饱和度
  all_delta  全图相对「修复前」的平均色差（色阶）—— 用来看"有没有误伤"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.compose import (  # noqa: E402
    auto_fog_color,
    bloom,
    depth_fog,
    limit_saturation,
    volumetric_light,
)
from pixelart.tune import TuneParams, build_scene  # noqa: E402



def hexs(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(0)
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(0)
    return float((c.max() - c.min()) / max(c.max(), 1e-6))


def grn(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(0)
    return float(c[1] - 0.5 * (c[0] + c[2]))


def composite(sc, p, fog_sat, spread):
    """用给定的 fog_sat / spread 合成一帧（不量化，看纯合成效果）。"""
    # ⚠️ 基线必须显式关掉上限：auto_fog_color 的默认值已经带 sat_max=0.40，
    #    直接用它的默认值会让"修复前"也已经是压过饱和的，评测就白做了
    #    （第一版就是这样，两列数字一模一样，差点得出"修复无效"的错结论）。
    raw = auto_fog_color(sc.base, sc.far, sat_max=1.0)
    c = raw if fog_sat is None else limit_saturation(raw, fog_sat)
    fogged = depth_fog(sc.base, sc.far, color=c, density=p.density, power=p.power)
    vol = volumetric_light(fogged, sc.far, sc.light_xy, strength=p.rays,
                           air_sat_max=p.rays_sat, screen_falloff=spread)
    lit = np.clip(fogged + vol, 0, 1)
    return bloom(lit, threshold=0.55, strength=p.bloom * 0.5, radii=(2, 5, 11)), c


print("修复前 = 雾色不压饱和 + 体积光只按深度衰减")
print("修复后 = 雾色上限 0.40 + 体积光屏幕收敛 1.0\n")
print(f"{'素材':24s} │ {'雾色 sat':>14s} │ {'远景偏绿':>16s} │ {'远景饱和':>16s} │ {'全图色差':>9s}")
print(f"{'':24s} │ {'前 → 后':>14s} │ {'前 → 后':>16s} │ {'前 → 后':>16s} │ {'均值':>9s}")
print("─" * 96)

rows = []
for f in sorted(INPUT.glob("*.jpg")):
    p = TuneParams(grid_long=160, work_long=640)
    sc = build_scene(Image.open(f), p)
    a, ca = composite(sc, p, None, 0.0)          # 修复前
    b, cb = composite(sc, p, 0.40, 1.0)          # 修复后
    far_m = sc.far >= np.quantile(sc.far, 0.85)
    delta = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() * 255)
    rows.append((f.stem, sat(ca), sat(cb), grn(a[far_m]), grn(b[far_m]),
                 sat(a[far_m]), sat(b[far_m]), delta))
    print(f"{f.stem[:22]:24s} │ {sat(ca):6.3f} → {sat(cb):5.3f} │ "
          f"{grn(a[far_m]):+7.3f} → {grn(b[far_m]):+7.3f} │ "
          f"{sat(a[far_m]):7.3f} → {sat(b[far_m]):6.3f} │ {delta:8.2f}")

print("─" * 96)
d_fog = np.mean([r[1] - r[2] for r in rows])
d_grn = np.mean([r[3] - r[4] for r in rows])
d_sat = np.mean([r[5] - r[6] for r in rows])
d_all = np.mean([r[7] for r in rows])
print(f"{'平均':24s} │ {d_fog:+14.3f} │ {d_grn:+16.3f} │ {d_sat:+16.3f} │ {d_all:9.2f}")
print()
print(f"  雾色平均降饱和      {d_fog:+.3f}   （正 = 修复后更中性）")
print(f"  远景偏绿平均减少    {d_grn:+.3f}")
print(f"  远景饱和平均减少    {d_sat:+.3f}")
print(f"  全图平均色差        {d_all:.2f} 色阶（含被刻意改变的样本）")
print()
print("  逐样本判读：色差 < 2 说明「没被误伤」，> 5 说明「确实改了」")
for r in rows:
    tag = ("未动 ✅" if r[7] < 2 else ("轻微" if r[7] < 5 else "已改"))
    print(f"    {r[0][:26]:28s} 色差 {r[7]:6.2f}  {tag}")
