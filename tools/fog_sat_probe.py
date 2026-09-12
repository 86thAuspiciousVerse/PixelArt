"""标定雾色的**饱和度上限** —— 修「像蒙了一层全局颜色滤镜」。

诊断结论：`auto_fog_color` 只做 25% 降饱和，于是 10 张素材里 7 张的
自动雾色饱和度超过 0.15，ref10 甚至到 **0.789**（拿一片饱和蓝天当雾色）。
雾会覆盖全部远景，所以饱和雾色 = **给远景蒙一层有色滤镜**。

真实的空气透视是**降饱和**的：散射把远处物体的颜色洗掉，趋近环境光（近中性）。
所以这里引入 `limit_saturation`（已在散射色上验证过），并标定上限。

标定要两头看：
  - 收太紧 → **暖色场景（暖灯、黄昏）的雾会发灰**，失去氛围
  - 收太松 → 全局滤镜感还在
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.compose import auto_fog_color, limit_saturation  # noqa: E402
from pixelart.tune import TuneParams, build_scene  # noqa: E402

CAPS = (0.20, 0.30, 0.40, None)


def hexs(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(0)
    return "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))


def sat(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(0)
    return float((c.max() - c.min()) / max(c.max(), 1e-6))


def warm(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(0)
    return float(c[0] - c[2])


rows = []
for f in sorted(INPUT.glob("*.jpg")):
    p = TuneParams(grid_long=140, work_long=560)
    sc = build_scene(Image.open(f), p)
    raw = auto_fog_color(sc.base, sc.far)
    rows.append((f.stem, raw))

print("雾色饱和度上限的横向影响（10 张素材）\n")
hdr = f"{'素材':24s} {'原始雾色':>9s} {'饱和':>6s} |"
for c in CAPS:
    hdr += f"  {('不压' if c is None else f'{c:.2f}'):>13s}"
print(hdr)
print("-" * 100)
for name, raw in rows:
    line = f"{name[:22]:24s} {hexs(raw):>9s} {sat(raw):6.3f} |"
    for cap in CAPS:
        c = limit_saturation(raw, 1.0) if cap is None else limit_saturation(raw, cap)
        line += f"  {hexs(c):>9s} {sat(c):4.2f}"
    print(line)

print("\n" + "=" * 100)
print("汇总")
print("=" * 100)
print(f"{'上限':10s} {'平均降饱和量':>14s} {'平均|暖度R-B|':>16s} {'暖度保留率':>12s}")
base_warm = np.mean([abs(warm(limit_saturation(r, 1.0))) for _, r in rows])
for cap in CAPS:
    ds = np.mean([sat(r) - sat(limit_saturation(r, 1.0) if cap is None
                             else limit_saturation(r, cap)) for _, r in rows])
    w = np.mean([abs(warm(limit_saturation(r, 1.0) if cap is None
                          else limit_saturation(r, cap))) for _, r in rows])
    print(f"{('不压' if cap is None else f'{cap:.2f}'):10s} {ds:14.3f} {w:16.3f} "
          f"{w / max(base_warm, 1e-9) * 100:11.0f}%")

print("\n暖色场景保留检验（ref02 暖灯咖啡店，原始暖度最高）")
print("-" * 62)
p = TuneParams(grid_long=140, work_long=560)
sc2 = build_scene(Image.open(INPUT / "ref02_replace_coffeeshop.jpg"), p)
raw2 = auto_fog_color(sc2.base, sc2.far)
for cap in CAPS:
    c = limit_saturation(raw2, 1.0) if cap is None else limit_saturation(raw2, cap)
    print(f"  上限 {('不压' if cap is None else f'{cap:.2f}'):6s} 雾色 {hexs(c)}  "
          f"饱和 {sat(c):.3f}  暖度 {warm(c):+.3f}")

print("\n判读：")
print("  · 上限 0.20 —— 降饱和最狠，但 ref02 的暖雾会被洗掉，暖色场景氛围损失")
print("  · 上限 0.30 —— 多数场景的全局滤镜感明显缓解，暖色保留大半  ← 候选默认")
print("  · 上限 0.40 —— 比较保守，ref10 这种极端(0.79)仍会偏色")
