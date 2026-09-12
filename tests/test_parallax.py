"""分层视差（M5）的回归测试 —— 每条对应一个真实事故或一条铁律。

为什么要这些测试：

- **循环闭合（时序铁律 3）**：视差位移是全管线里第一个"直接搬像素"的时间项。
  波形选错（sin 会在 t=1 留下 -2.45e-16 的残留）就会破坏 frame(0)==frame(N)。
  `1 - cos(τt)` 能逐位闭合是**实测出来的**（cos(τ) 恰好等于 1.0），这里钉死。
- **非空转**：上一轮抓过"数学上在动、视觉上没动"的假特征
  （相邻帧平均差 0.26/255）。所以这里要求 t=0.5 与 t=0 的 u8 差**大于肉眼阈值**。
- **amp=0 必须原样返回**：默认参数（视差关）的输出必须与旧版**逐位相同** ——
  这是"默认体验不被新特性改变"的约束。
- **分层单调**：近层位移必须显著大于远层 —— 视差的全部意义就在这个差上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pixelart.parallax import layer_weights, parallax_offsets, parallax_wave, parallax_warp
from pixelart.pipeline import AnimParams, compose_frame, finish_frame
from pixelart.tune import TuneParams, build_scene

ASSET = "ref01_roadsign_night.jpg"


# ══════════════════════════════════════════════════════════════════
def test_wave_closes_bitexact():
    """波形在 t=0 与 t=1 **逐位**为 0 —— 循环闭合的立足点。

    ⚠️ 不要"顺手优化"成 sin：sin(τ) = -2.45e-16，逐位就不是 0 了。
    """
    assert parallax_wave(0.0) == 0.0
    assert parallax_wave(1.0) == 0.0
    assert parallax_wave(0.0) == parallax_wave(1.0)
    assert parallax_wave(0.5) == pytest.approx(2.0)


def _synthetic():
    """上=远(1.0) 下=近(0.0) 的两段深度 + 竖条纹纹理（位移可测）。"""
    gh, gw = 60, 120
    far = np.vstack([np.ones((gh // 2, gw)),
                     np.zeros((gh - gh // 2, gw))]).astype(np.float32)
    xs = np.arange(gw, dtype=np.float32)[None, :]
    base = (((xs % 12) < 6).astype(np.float32))[..., None] \
        * np.ones((gh, gw, 3), dtype=np.float32) * 0.8
    return base, far


def test_amp_zero_returns_original_arrays():
    """amp=0 必须**原样返回**（同一对象）—— 默认体验与旧版逐位一致。"""
    base, far = _synthetic()
    b, f, _ = parallax_warp(base, far, 0.5, amp=0.0)
    assert b is base and f is far


def test_loop_closure_bitexact():
    """t=0 与 t=1 的 warp 输出逐位相同。"""
    base, far = _synthetic()
    b0, f0, _ = parallax_warp(base, far, 0.0, amp=0.5)
    b1, f1, _ = parallax_warp(base, far, 1.0, amp=0.5)
    assert np.array_equal(b0, b1)
    assert np.array_equal(f0, f1)


def test_motion_is_not_noop():
    """非空转：中间时刻的输出必须与 t=0 有实质差异。"""
    base, far = _synthetic()
    b0, _, _ = parallax_warp(base, far, 0.0, amp=0.5)
    for t in (0.25, 0.5, 0.75):
        bt, _, _ = parallax_warp(base, far, t, amp=0.5)
        assert float(np.abs(bt - b0).mean()) > 0.01, f"t={t} 几乎没动"


def test_layer_motion_is_monotonic():
    """近层位移必须显著大于远层 —— 视差的全部意义。"""
    base, far = _synthetic()
    ox, oy, info = parallax_offsets(far, 0.5, amp=0.5)
    gh = far.shape[0]
    near = float(np.abs(ox[gh // 2:]).mean())          # 下半 = 近
    far_ = float(np.abs(ox[:gh // 2]).mean())          # 上半 = 远
    assert near > far_ * 3.0
    assert info["max_off_px"] > 0


def test_weights_far_gain_smaller_than_near():
    """运动系数：远层严格小于近层（远景"贴在玻璃后"比"跟着乱动"好）。"""
    far = np.linspace(0.0, 1.0, 100, dtype=np.float32)[None, :]
    w = layer_weights(far, layers=8)
    assert w[0, 0] > w[0, -1]


def test_no_nan_and_no_invalid_output():
    """splat + 兜底之后不允许出现 NaN/Inf（破洞用原图混合填充）。"""
    rng = np.random.RandomState(3)
    far = rng.rand(40, 90).astype(np.float32)
    base = rng.rand(40, 90, 3).astype(np.float32)
    b, f, info = parallax_warp(base, far, 0.35, amp=0.8)
    assert np.isfinite(b).all() and np.isfinite(f).all()
    assert 0.0 <= info["holes_pct"] <= 100.0


# ── 端到端：compose_frame ──────────────────────────────────────────
@pytest.fixture(scope="module")
def scene():
    params = TuneParams.from_query({
        "asset": [ASSET], "grid_long": ["120"], "work_long": ["480"],
        "aspect": ["native"],
    })
    from PIL import Image
    return build_scene(Image.open(ROOT / "assets" / "input" / ASSET), params)


def _u8(scene, t, anim):
    """compose → finish，取 **u8 帧本体**。

    ⚠️ finish_frame 返回二元组 (u8 帧, 边缘图)。第一版没解包，
    pytest 报 "'tuple' object has no attribute 'astype'" 才发现 ——
    这个报错离真因很远（看起来像 compose 返回错了）。
    """
    return finish_frame(compose_frame(scene, t=t, anim=anim), scene.tail)[0]


def test_compose_default_path_unchanged(scene):
    """parallax=0 时，带不带这个字段输出**逐位相同** —— 默认体验不变。"""
    anim_old = AnimParams(fog_drift=0.3, light_flicker=0.16,
                          dust_count=50, dust_bright=0.5)
    anim_zero = AnimParams(**{**anim_old.to_dict(), "parallax": 0.0})
    a = compose_frame(scene, t=0.3, anim=anim_old)
    b = compose_frame(scene, t=0.3, anim=anim_zero)
    assert np.array_equal(a, b)


def test_compose_loop_closure_with_parallax(scene):
    """端到端闭合：quantize 之后的 u8 帧 frame(0) == frame(1) 逐位相同。"""
    anim = AnimParams(fog_drift=0.3, light_flicker=0.16, dust_count=50,
                      dust_bright=0.5, parallax=0.45)
    assert np.array_equal(_u8(scene, 0.0, anim), _u8(scene, 1.0, anim))


def test_compose_motion_visible(scene):
    """端到端非空转：quantize 后 frame(0.5) 与 frame(0) 必须有肉眼可见差异。"""
    anim = AnimParams(fog_drift=0.3, light_flicker=0.16, dust_count=50,
                      dust_bright=0.5, parallax=0.45)
    f0 = _u8(scene, 0.0, anim)
    f5 = _u8(scene, 0.5, anim)
    d = float(np.abs(f5.astype(int) - f0.astype(int)).mean())
    assert d > 2.0, f"平均差 {d:.2f}/255 —— 视觉上没动 = 白做"
