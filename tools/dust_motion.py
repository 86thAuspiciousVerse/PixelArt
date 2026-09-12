"""粒子运动模型对照：旧的"匀速直线横穿" vs 新的"有界游荡"。

用户报障：「像是有蚊子或者是蚂蚁在**直线爬行**」—— 这是运动模型的问题，
和"黑点"是两件独立的事（黑点见 `tools/dust_probe.py`）。

量法：**只放一颗粒子**（count=1），逐帧记录它的位置，直接量它的行程。
这样没有粒子重叠的干扰，数字是无歧义的。

判据：
  - **总行程**（把每帧位移累加）与**最大离原点距离**：
    匀速直线横穿 → 行程 ≈ 一个画面尺寸（甚至更多，因为 wrap）
    有界摆动     → 行程是摆动弧长，但**离原点距离有上界**（≈ drift）
  - **离原点最大距离 / 画面长边** 是最关键的一个数：
    它直接回答"粒子会不会跑到画面另一头去"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelart.animate import dust_layer, hash01  # noqa: E402

H, W, N, SEED = 72, 120, 48, 5


def old_pos(i, t):
    """修复前的运动模型：整数圈数**匀速直线**位移（wrap 折返）。"""
    x0 = hash01(SEED, i, 1)
    y0 = hash01(SEED, i, 2)
    kx = int(round(hash01(SEED, i, 3) * 2 - 1))
    ky = int(round(hash01(SEED, i, 4) * 2 - 1))
    if ky == 0 and kx == 0:
        ky = -1 if hash01(SEED, i, 5) < 0.5 else 1
    return ((x0 + kx * t) % 1.0 * W) % W, ((y0 + ky * t) % 1.0 * H) % H


def new_pos(i, t, drift=0.045):
    """修复后的模型：Lissajous 有界摆动。"""
    x0, y0 = hash01(SEED, i, 1), hash01(SEED, i, 2)
    ax = drift * (0.30 + 0.70 * hash01(SEED, i, 3))
    ay = drift * (0.30 + 0.70 * hash01(SEED, i, 4))
    fx = 1 + int(hash01(SEED, i, 5) * 2.999)
    fy = 1 + int(hash01(SEED, i, 6) * 2.999)
    phx = 2.0 * np.pi * hash01(SEED, i, 7)
    phy = 2.0 * np.pi * hash01(SEED, i, 8)
    return (((x0 + ax * np.sin(2 * np.pi * fx * t + phx)) % 1.0) * W,
            ((y0 + ay * np.sin(2 * np.pi * fy * t + phy)) % 1.0) * H)


def track(fn, label, n_par=6):
    print(f"  {label}")
    travel, maxoff, steps = [], [], []
    ts = [k / N for k in range(N + 1)]
    for i in range(n_par):
        pts = np.array([fn(i, t) for t in ts])            # (N+1, 2)
        x0 = np.array(fn(i, 0.0))
        # 逐帧步长
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        # wrap 造成的假跳变（>画面一半）剔除
        d = d[d < min(W, H) * 0.5]
        travel.append(float(d.sum()))
        # 离"基准位置"的最大距离：用 hash 给的原点，不用帧 0（帧 0 已在摆动）
        base = np.array([hash01(SEED, i, 1) * W, hash01(SEED, i, 2) * H])
        off = np.linalg.norm(((pts - base + np.array([W / 2, H / 2])) %
                              np.array([W, H])) - np.array([W / 2, H / 2]), axis=1)
        maxoff.append(float(off.max()))
        steps.append(float(d.mean()))
    print(f"    逐帧步长       {np.mean(steps):6.2f} px/帧")
    print(f"    总行程         {np.mean(travel):6.1f} px   (画面长边 {W} px)")
    print(f"    离基准最大距离 {np.mean(maxoff):6.2f} px   = {np.mean(maxoff) / W * 100:4.1f}% 画面宽")
    print(f"    行程/最大偏移  {np.mean(travel) / max(np.mean(maxoff), 1e-6):6.2f}"
          f"   （大 = 来回跑；小 = 单向直行）")
    return np.mean(maxoff) / W, np.mean(travel) / W


print(f"网格 {W}x{H}   跟踪 6 颗粒子   一个循环 {N} 帧\n")
print("=== 修复前：整数圈数匀速直线 ===")
o_off, o_tr = track(old_pos, "旧 dust_layer")
print("\n=== 修复后：Lissajous 有界摆动 ===")
n_off, n_tr = track(new_pos, "新 dust_layer")

print("\n判读：")
print(f"  『离基准最大距离』占画面宽 —— 旧 {o_off * 100:.1f}%  →  新 {n_off * 100:.1f}%")
print("     旧模型大 = 粒子会跑到画面另一头（横穿，观感=蚂蚁爬行）")
print("     新模型小 = 粒子只在自己附近游荡（观感=悬浮）")
print(f"  『总行程/最大偏移』—— 旧 {o_tr / max(o_off, 1e-9):.1f} vs 新 {n_tr / max(n_off, 1e-9):.1f}")
print("     比值大 = 在原地来回（摆动）；接近 1 = 单向直行（爬行）")
