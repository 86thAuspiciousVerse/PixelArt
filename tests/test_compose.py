"""场景合成的单元测试。

重点固化两个踩坑得来的结论：

1. 深度是"排名"不是"距离"，直接当距离用会让雾吞掉整幅画面
2. 雾必须在远端发力（power > 1），否则中景就糊
"""

import numpy as np
import pytest

from pixelart.compose import (
    auto_fog_color,
    auto_scatter_color,
    bloom,
    blur,
    brightest_center,
    bright_pass,
    depth_equalize,
    depth_fog,
    depth_remap,
    god_rays,
    limit_saturation,
    tint,
    volumetric_light,
)


def _gradient(h=64, w=96):
    """竖直渐变图：底部亮（近），顶部暗（远）。"""
    yy = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None, None]
    return np.repeat(np.repeat(yy, w, axis=1), 3, axis=2) * np.array([0.9, 0.6, 0.35], np.float32)


def _linear_depth(h=64, w=96):
    return np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None].repeat(w, 1)


# ------------------------------------------------------------------ blur
@pytest.mark.parametrize("shape", [(16, 16), (16, 16, 1), (16, 16, 3)])
def test_blur_preserves_shape(shape):
    rng = np.random.default_rng(0)
    a = rng.random(shape, dtype=np.float32)
    assert blur(a, 1.5).shape == shape


def test_blur_zero_radius_is_identity():
    rng = np.random.default_rng(1)
    a = rng.random((8, 8, 3), dtype=np.float32)
    assert np.array_equal(blur(a, 0), a)


def test_blur_smooths():
    a = np.zeros((16, 16, 3), dtype=np.float32)
    a[8, 8] = 1.0
    b = blur(a, 2.0)
    assert b.max() < 1.0            # 能量被摊开
    assert b[7:10, 7:10].sum() > 0  # 但没丢


# ------------------------------------------------------------------ 深度处理
def test_depth_equalize_flattens_distribution():
    """偏斜输入 → 重映射后应当接近均匀。"""
    skewed = np.concatenate([
        np.full(9000, 0.9, dtype=np.float32),
        np.linspace(0.0, 0.5, 1000, dtype=np.float32),
    ]).reshape(100, 100)
    eq = depth_equalize(skewed, strength=1.0)
    q = np.percentile(eq, [10, 50, 90])
    assert q[0] < 0.2 and 0.4 < q[1] < 0.6 and q[2] > 0.8
    assert 0.0 <= eq.min() and eq.max() <= 1.0


def test_depth_equalize_strength_zero_is_identity():
    d = np.linspace(0, 1, 50, dtype=np.float32).reshape(5, 10)
    assert np.allclose(depth_equalize(d, strength=0.0, clip=(0.0, 100.0)), d, atol=1e-6)


def test_depth_remap_maps_percentiles_to_unit_range():
    rng = np.random.default_rng(2)
    d = rng.normal(0.8, 0.05, (40, 40)).astype(np.float32)
    r = depth_remap(d, 2, 98)
    assert not np.isclose(np.median(r), 0.9), "线性重映射会保住偏斜，这符合它的定位"
    assert 0.0 <= r.min() and r.max() <= 1.0


# ------------------------------------------------------------------ 雾
def test_auto_fog_color_is_robust_to_a_bright_blob():
    """远景里放一块极亮区域，自动雾色不应被它带跑偏（用中位数）。"""
    rgb = np.full((64, 64, 3), 0.20, dtype=np.float32)
    rgb[0:6, 0:6] = 1.0                      # 一块超亮区域
    far = np.ones((64, 64), dtype=np.float32)
    c = auto_fog_color(rgb, far, q=0.80, lift=0.0, desat=0.0)
    assert c.max() < 0.35, f"被亮块带偏了: {c}"


def test_depth_fog_near_untouched_far_replaced():
    """第 0 行是近（z=0），最后一行是远（z=1）—— 与图像坐标一致：上远下近。"""
    rgb = _gradient()
    far = _linear_depth()
    out = depth_fog(rgb, far, color=(0.0, 0.0, 0.0), density=3.0, power=1.0)
    assert np.allclose(out[0], rgb[0], atol=0.02), "最近处应保持原色"
    assert out[-1].mean() < 0.12, "最远处应几乎只剩雾色"


def test_depth_fog_power_keeps_midground_clear():
    """这是踩坑点：power=1 时中景就被糊掉，power 大时中景应当基本保留。"""
    rgb = _gradient()
    far = _linear_depth()
    soft = depth_fog(rgb, far, color=(0.0, 0.0, 0.0), density=1.4, power=1.0)
    hard = depth_fog(rgb, far, color=(0.0, 0.0, 0.0), density=1.4, power=3.0)
    mid = rgb.shape[0] // 2
    # 中景：power=3 应当比 power=1 保留更多原始亮度
    assert hard[mid].mean() > soft[mid].mean()


def test_depth_fog_is_monotonic_in_distance():
    rgb = np.ones((40, 8, 3), dtype=np.float32)
    far = _linear_depth(40, 8)
    out = depth_fog(rgb, far, color=(0.0, 0.0, 0.0), density=2.0, power=1.0)
    col = out[:, 0, 0]
    assert np.all(np.diff(col) <= 1e-6), "越远应当越暗（越接近雾色）"


def test_depth_fog_range():
    rng = np.random.default_rng(3)
    rgb = rng.random((32, 32, 3), dtype=np.float32)
    far = rng.random((32, 32), dtype=np.float32)
    out = depth_fog(rgb, far, density=2.0)
    assert out.shape == rgb.shape and 0.0 <= out.min() and out.max() <= 1.0


# ------------------------------------------------------------------ 高光 / 光轴 / 辉光
def test_bright_pass_keeps_only_bright():
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[:8] = 0.9
    rgb[8:] = 0.1
    bp = bright_pass(rgb, threshold=0.6)
    assert bp[:8].mean() > 0.1
    assert bp[8:].mean() < 1e-6


def test_god_rays_shape_and_nonnegative():
    rgb = np.zeros((48, 64, 3), dtype=np.float32)
    rgb[24, 32] = 1.0
    rays = god_rays(bright_pass(rgb, 0.5), (0.5, 0.5), samples=12, strength=0.6)
    assert rays.shape == rgb.shape
    assert rays.min() >= 0.0 and rays.max() <= 1.0
    assert rays.sum() > 0.0, "光轴应当往外散出能量"


def test_god_rays_mask_suppresses_outside():
    rgb = np.zeros((48, 64, 3), dtype=np.float32)
    rgb[24, 32] = 1.0
    bp = bright_pass(rgb, 0.5)
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[20:28, 28:36] = 1.0
    rays = god_rays(bp, (0.5, 0.5), samples=12, strength=0.8, mask01=mask)
    assert rays[0, 0].sum() == 0.0


def test_brightest_center():
    rgb = np.zeros((40, 60, 3), dtype=np.float32)
    rgb[10:14, 45:50] = 1.0
    cx, cy = brightest_center(rgb, blur_radius=1.0)
    assert 0.70 < cx < 0.86 and 0.20 < cy < 0.40


def test_bloom_only_brightens():
    rng = np.random.default_rng(4)
    rgb = rng.random((32, 32, 3), dtype=np.float32) * 0.3
    rgb[14:18, 14:18] = 0.95
    out = bloom(rgb, threshold=0.5, strength=0.8)
    assert out.shape == rgb.shape
    assert out.mean() >= rgb.mean()
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_tint():
    rgb = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    assert np.allclose(tint(rgb, (0.0, 0.0, 0.0), 1.0), 0.0)


# ---------------------------------------------------------------- 体积散射色
def test_limit_saturation_caps_and_keeps_luminance():
    """饱和度上限只压色度、不动平均亮度。"""
    c = np.array([0.618, 0.793, 0.392], dtype=np.float32)      # 饱和黄绿 sat=0.51
    out = limit_saturation(c, 0.25)
    sat = (out.max() - out.min()) / out.max()
    assert sat <= 0.26, f"饱和度没压下来：{sat:.3f}"
    assert abs(float(out.mean()) - float(c.mean())) < 0.02, "平均亮度应当保持"
    # 本来就不饱和的颜色不该被改动
    grey = np.array([0.5, 0.52, 0.49], dtype=np.float32)
    assert np.allclose(limit_saturation(grey, 0.25), grey)


def test_auto_scatter_color_survives_a_fully_colored_window():
    """⚠️ 回归：光源定位到一片**纯色**物体上时，散射色不能是饱和色。

    实测事故：ref10_green_cliff 的 brightest_center 落在了**草丛**上，
    旧代码取窗口均值得到 (0.618, 0.793, 0.392) —— 饱和黄绿，
    叠加后整幅图变绿（用户报障的第二个来源）。
    窗口里全是绿色像素，所以选像素/换统计量都救不回来，只有饱和度上限能救。
    """
    h = w = 40
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:] = np.array([0.618, 0.793, 0.392], dtype=np.float32)   # 一整片草丛
    rgb += np.random.default_rng(0).normal(0, 0.004, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0, 1)

    c = auto_scatter_color(rgb, 20, 20, radius=3, sat_max=0.25)
    sat = (c.max() - c.min()) / c.max()
    assert sat <= 0.26, f"散射色仍是饱和色（sat={sat:.3f}），会整幅染色"
    # 偏绿指数要明显低于旧行为（旧代码取均值 → 0.288）。
    # 注意偏绿指数和饱和度是绑定的：纯绿在 sat=0.25 时偏绿指数上限就是 0.25，
    # 所以这里只能要求"比旧行为好一大截"，不能要求归零。
    gi = float(c[1] - 0.5 * (c[0] + c[2]))
    assert gi < 0.15, f"偏绿指数仍偏高（{gi:.3f}，旧行为是 0.288）"


def test_auto_scatter_color_preserves_warm_light():
    """⚠️ 上限不能伤到合法的暖色光场景。

    实测标定：把 sat_max 收到 0.15 会把 ref02（REPLACE 咖啡店暖灯）的
    暖度从 +0.232 抽到 +0.130，光晕发灰。0.25 是保住暖色的下限。
    """
    warm = np.array([0.90, 0.83, 0.68], dtype=np.float32)        # 暖灯，sat≈0.24
    rgb = np.tile(warm, (40, 40, 1)).astype(np.float32)
    c = auto_scatter_color(rgb, 20, 20, radius=3, sat_max=0.25)
    assert float(c[0] - c[2]) > 0.15, f"暖色被抽掉了（R−B={float(c[0] - c[2]):.3f}）"


def test_auto_scatter_color_median_ignores_one_blown_highlight():
    """取中位数而不是均值：窗口里单个炸白的高光不该把散射色带成白色。"""
    rgb = np.full((40, 40, 3), 0.70, dtype=np.float32)
    rgb[20, 20] = 1.0                                            # 单个纯白高光
    c = auto_scatter_color(rgb, 20, 20, radius=3, sat_max=0.25)
    assert abs(float(c.mean()) - 0.70) < 0.02, f"被单个高光带跑了（{float(c.mean()):.3f}）"


def test_volumetric_light_does_not_green_a_green_light_position():
    """端到端：光源落在绿地上时，体积光不该让画面整体变绿。"""
    h, w = 48, 64
    rgb = np.full((h, w, 3), 0.25, dtype=np.float32)
    rgb[:, :, 1] = 0.45                                          # 整片偏绿
    rgb[30:34, 40:44] = np.array([0.62, 0.79, 0.39], dtype=np.float32)   # 只把"最亮处"放这里
    far = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
    xy = (42 / (w - 1), 32 / (h - 1))

    vol = volumetric_light(rgb, far, xy, strength=0.55)
    if float(vol.sum()) > 0:
        c = vol.sum(axis=(0, 1)) / vol.sum()
        assert float(c[1] - 0.5 * (c[0] + c[2])) < 0.12, "体积光注入的颜色仍然明显偏绿"
