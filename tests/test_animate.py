"""时间参数的单元测试。

这个文件固化的全部是**时序一致性铁律**（arch-01 §8 D5/D6）——
每一条被"优化"掉，成片就会在循环处出现接缝或者整片闪烁。

1. 一切随时间变化的量必须严格以 1 为周期 → ``frame(0) == frame(N)``
2. 抖动相位不能含 t（在 test_pixelate.py 里守）
3. 静态图必须完全不受 t 影响（drift/flicker 默认 0 时逐位相同）
"""

import numpy as np
import pytest

from pixelart.animate import (
    DEFAULT_FLICKER_FREQS,
    flicker_gain,
    hash01,
    loop_noise_2d,
    periodic_sines,
)

TS = np.linspace(0.0, 1.0, 97)


# ------------------------------------------------------------ 周期性（铁律）
@pytest.mark.parametrize("t", TS)
def test_periodic_sines_is_exactly_periodic(t):
    """``s(t + 1) == s(t)`` —— 循环无缝的前提。"""
    assert abs(periodic_sines(t) - periodic_sines(t + 1.0)) < 1e-9
    assert abs(periodic_sines(t) - periodic_sines(t + 3.0)) < 1e-9    # 多个周期也要成立


@pytest.mark.parametrize("freqs", [(1,), (1, 3, 7, 11), (2, 4, 6), (1, 2, 3, 4, 5)])
def test_periodicity_holds_for_any_integer_freq_set(freqs):
    for t in TS:
        assert abs(periodic_sines(t, freqs) - periodic_sines(t + 1.0, freqs)) < 1e-9


@pytest.mark.parametrize("t", TS)
def test_loop_noise_2d_is_exactly_periodic(t):
    """⚠️ 三维循环噪声：整个场在 t 与 t+1 必须（近似）逐位相等。

    这是"雾漂移/粒子不会在循环接缝跳变"的唯一保证。
    实现上用**整数时间频率**的谱合成，所以误差只在浮点舍入量级。
    """
    a = loop_noise_2d(t, (24, 32), octaves=3, seed=3)
    b = loop_noise_2d(t + 1.0, (24, 32), octaves=3, seed=3)
    assert np.abs(a - b).max() < 1e-6, f"循环接缝：最大差 {np.abs(a - b).max():.3e}"


def test_non_integer_frequency_is_rejected():
    """⚠️ 非整数频率会让周期 > 1 → 循环有接缝。必须直接拒绝，不能悄悄接受。"""
    with pytest.raises(ValueError, match="整数"):
        periodic_sines(0.3, (1, 1.5))
    with pytest.raises(ValueError):
        periodic_sines(0.3, (0.5,))


# --------------------------------------------------------------- 真的在动
def test_signals_actually_vary_over_time():
    """周期性容易被写成"恒等于常数"——那也满足周期，但画面是死的。"""
    v = [periodic_sines(t) for t in TS]
    assert max(v) - min(v) > 0.5, f"信号太平：值域 {max(v) - min(v):.3f}"

    n0 = loop_noise_2d(0.0, (24, 32), seed=3)
    n1 = loop_noise_2d(0.5, (24, 32), seed=3)
    assert np.abs(n0 - n1).mean() > 0.05, "噪声场没有随时间变化"


def test_flicker_gain_depth_zero_is_identity():
    """``depth=0`` 必须完全不动，否则静态图会抖。"""
    assert all(flicker_gain(t, depth=0.0) == 1.0 for t in TS)


def test_flicker_gain_honours_bias():
    """``bias`` 给出基准强度（比如想让光源整体更亮一点）。"""
    g = [flicker_gain(t, depth=0.1, bias=0.5) for t in TS]
    assert min(g) >= 0.4 - 1e-9 and max(g) <= 0.6 + 1e-9


def test_flicker_gain_is_bounded_by_depth():
    for depth in (0.1, 0.25, 0.4):
        g = [flicker_gain(t, depth=depth) for t in np.linspace(0, 1, 400)]
        assert min(g) >= 1.0 - depth - 1e-9, "闪烁超出下界"
        assert max(g) <= 1.0 + depth + 1e-9, "闪烁超出上界"
        assert max(g) - min(g) > 0.1, "闪烁幅度太小，看不出来"


# ------------------------------------------------------------- 静态图不受影响
def test_static_frame_is_independent_of_t():
    """⚠️ 静态图就是视频的第 0 帧 —— 但不该因为传了 t 就变样。

    ``drift`` / ``flicker`` 默认 0 时，输出必须与 t 完全无关。
    """
    from pixelart.compose import depth_fog, volumetric_light

    rng = np.random.default_rng(1)
    rgb = np.clip(rng.random((36, 48, 3), dtype=np.float32) * 0.7, 0, 1)
    far = np.linspace(0, 1, 36, dtype=np.float32)[:, None].repeat(48, 1)

    for t in (0.0, 0.37, 0.91):
        assert np.array_equal(depth_fog(rgb, far, density=0.7, t=t),
                              depth_fog(rgb, far, density=0.7, t=0.0))
        assert np.array_equal(volumetric_light(rgb, far, (0.5, 0.5), t=t),
                              volumetric_light(rgb, far, (0.5, 0.5), t=0.0))


def test_compose_time_effects_are_periodic():
    """开启 drift / flicker 之后，合成结果仍必须严格循环。"""
    from pixelart.compose import depth_fog, volumetric_light

    rng = np.random.default_rng(2)
    rgb = np.clip(rng.random((36, 48, 3), dtype=np.float32) * 0.7, 0, 1)
    far = np.linspace(0, 1, 36, dtype=np.float32)[:, None].repeat(48, 1)

    a = depth_fog(rgb, far, density=0.7, drift=0.35, t=0.0)
    b = depth_fog(rgb, far, density=0.7, drift=0.35, t=1.0)
    assert np.abs(a - b).max() < 1e-6

    va = volumetric_light(rgb, far, (0.5, 0.5), flicker=0.25, t=0.0)
    vb = volumetric_light(rgb, far, (0.5, 0.5), flicker=0.25, t=1.0)
    assert np.abs(va - vb).max() < 1e-6


def test_compose_time_effects_actually_animate():
    from pixelart.compose import depth_fog, volumetric_light

    rng = np.random.default_rng(3)
    rgb = np.clip(rng.random((36, 48, 3), dtype=np.float32) * 0.7, 0, 1)
    far = np.linspace(0, 1, 36, dtype=np.float32)[:, None].repeat(48, 1)

    assert np.abs(depth_fog(rgb, far, density=0.7, drift=0.35, t=0.0)
                  - depth_fog(rgb, far, density=0.7, drift=0.35, t=0.5)).mean() > 1e-4
    assert not np.array_equal(volumetric_light(rgb, far, (0.5, 0.5), flicker=0.25, t=0.0),
                              volumetric_light(rgb, far, (0.5, 0.5), flicker=0.25, t=0.5))


# ------------------------------------------------------------------- 哈希
def test_hash01_is_deterministic_and_spread():
    """粒子要靠它生成"固定的随机"——必须可复现、且分布均匀。"""
    assert hash01(0, 0) == hash01(0, 0)
    assert hash01(7, 3) != hash01(7, 4)
    v = np.array([hash01(i, 3) for i in range(4000)])
    assert v.min() >= 0.0 and v.max() < 1.0
    assert abs(v.mean() - 0.5) < 0.02, f"分布不均，均值 {v.mean():.4f}"


def test_default_flicker_freqs_are_coprime():
    """⚠️ 频率必须互质：否则波形会在一个周期内**提前重复**，看起来像卡带。

    取到 1/3/7/11 这四个两两互质的数，合成波形在 [0,1) 内不重复。
    """
    from math import gcd

    for i, a in enumerate(DEFAULT_FLICKER_FREQS):
        for b in DEFAULT_FLICKER_FREQS[i + 1:]:
            assert gcd(a, b) == 1, f"{a} 与 {b} 不互质，波形会提前重复"
