"""天空层替换（M5.7）的单测。

覆盖四条约定：
1. 双保守门：无天空（深度全低）→ **逐位不变**；天空占比 < 2% → 逐位不变。
2. 替换后：掩膜内出现预设渐变色、亮度结构正确（顶暗/地平线亮，夜空）。
3. 确定性：同参数同星图（逐位相同）；星密度 0 = 无星。
4. 白天模式无星。
"""

from __future__ import annotations

import numpy as np

from pixelart.sky import SKY_PRESETS, sky_replace


def _scene_with_sky(h=90, w=160, sky_depth=0.995, ground_depth=0.3):
    """合成一张"上天空下地面"的深度图与配图。"""
    far = np.full((h, w), ground_depth, dtype=np.float32)
    far[: h // 2, :] = sky_depth
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[: h // 2] = (0.4, 0.5, 0.7)          # 假天空（会被替换）
    base[h // 2:] = (0.2, 0.35, 0.2)          # 地面（不能被碰到）
    return base, far


def test_night_replaces_sky_with_dark_gradient():
    base, far = _scene_with_sky()
    out, info = sky_replace(base, far, mode="night", stars=0.0)
    assert info is not None and info["fraction"] > 0.4
    # 天空区变暗（夜空渐变 ≤ 原假天空亮度）
    assert out[: 45, :].mean() < base[: 45, :].mean()
    # 渐变结构：最顶行比地平线附近更暗
    assert out[2, :].mean() < out[43, :].mean()


def test_no_sky_returns_bit_identical():
    base, _ = _scene_with_sky()
    far = np.full(base.shape[:2], 0.3, dtype=np.float32)   # 室内：无无限远
    out, info = sky_replace(base, far, mode="night", stars=1.0)
    assert info is None
    assert np.array_equal(out, base)                        # 逐位不变


def test_small_fraction_returns_bit_identical():
    base, far = _scene_with_sky()
    far[:] = 0.3
    far[0, 0] = 0.995                                       # 只有 1 个像素 → < 2%
    out, info = sky_replace(base, far, mode="night")
    assert info is None
    assert np.array_equal(out, base)


def test_stars_deterministic_and_density_knob_works():
    base, far = _scene_with_sky()
    a, _ = sky_replace(base, far, mode="night", stars=1.0, seed=7)
    b, _ = sky_replace(base, far, mode="night", stars=1.0, seed=7)
    assert np.array_equal(a, b)                             # 同参数同星图
    hi, _ = sky_replace(base, far, mode="night", stars=2.0)
    lo, _ = sky_replace(base, far, mode="night", stars=0.5)
    # 星密度高 → 天空区与纯渐变的偏离更大
    grad = np.clip(np.linspace(0, 1, base.shape[0])[:, None, None], 0, 1)
    top = np.array(SKY_PRESETS["night"]["top"], np.float32) / 255
    hor = np.array(SKY_PRESETS["night"]["hor"], np.float32) / 255
    pure = top * (1 - grad) + hor * grad
    dev_hi = float(np.abs(hi[: 45] - pure[: 45]).mean())
    dev_lo = float(np.abs(lo[: 45] - pure[: 45]).mean())
    assert dev_hi > dev_lo


def test_day_has_no_stars():
    base, far = _scene_with_sky()
    out, info = sky_replace(base, far, mode="day", stars=2.0)
    assert info is not None and info["stars"] == 0


def test_ground_never_touched():
    base, far = _scene_with_sky()
    out, _ = sky_replace(base, far, mode="night", stars=1.0)
    # 羽化允许地平线以下 ~3px 的过渡带（防锯齿，设计使然）；更深处必须逐位不变
    assert np.array_equal(out[50:], base[50:])
