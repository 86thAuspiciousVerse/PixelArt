"""像素尾巴的单元测试。

除了正确性，这里还固化了几条**架构级约束**（时序一致性），
免得以后重构时不小心把抖动相位写成依赖帧号。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelart.dither import BAYER8, bayer_matrix, bayer_signed, bayer_tile
from pixelart.palette import (
    ensure_neutral_highlight,
    from_perceptual,
    LUMA,
    palette_from_frames,
    palette_from_image,
    palette_to_hex,
    parse_hex_palette,
    refine_palette,
    snap_exact,
    to_perceptual,
)
from pixelart.pixelate import PixelTail, pixelate, render_frame
from pixelart.resample import (
    fit_to_aspect,
    structure_aware_downsample,
    tone_map,
    upscale_nearest,
)

# --------------------------------------------------------------------- #
# 抖动
# --------------------------------------------------------------------- #


def test_bayer_shape_and_uniqueness():
    m = bayer_matrix(8)
    assert m.shape == (8, 8)
    assert m.min() >= 0.0 and m.max() < 1.0
    assert len(np.unique(m)) == 64, "8x8 Bayer 矩阵应当 64 个取值互不相同"
    assert bayer_matrix(4).shape == (4, 4)
    assert bayer_matrix(16).shape == (16, 16)


def test_bayer_rejects_non_power_of_two():
    for bad in (0, 1, 3, 6, 10):
        with pytest.raises(ValueError):
            bayer_matrix(bad)


def test_bayer_signed_range_and_shape():
    s = bayer_signed(16, 24)
    assert s.shape == (16, 24, 1)
    assert s.min() >= -0.5 and s.max() < 0.5
    assert bayer_tile(8, 8).shape == (8, 8, 1)
    assert BAYER8.shape == (8, 8)


# --------------------------------------------------------------------- #
# 降采样 / 影调
# --------------------------------------------------------------------- #


def test_tone_map_identity_when_no_scurve():
    x = np.linspace(0, 1, 11, dtype=np.float32)
    assert np.allclose(tone_map(x, 0.0, 1.0, 0.0), x)


def test_tone_map_levels():
    # black=0.25 / white=0.75 把 [0.25, 0.75] 线性拉满到 [0, 1]
    x = np.array([0.25, 0.50, 0.75], dtype=np.float32)
    y = tone_map(x, black=0.25, white=0.75, scurve=0.0)
    assert np.allclose(y, [0.0, 0.5, 1.0])


def test_structure_aware_preserves_hard_edge():
    """左右各半的硬边界，降采样后仍应分明 —— 这就是「保轮廓」的含义。"""
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[:, 32:] = 255
    d = structure_aware_downsample(Image.fromarray(a), (8, 8))
    assert d.shape == (8, 8, 3)
    assert d[:, :4].mean() < 0.25
    assert d[:, 4:].mean() > 0.75


def test_structure_aware_output_range():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
    d = structure_aware_downsample(img, (20, 13))
    assert d.shape == (13, 20, 3)
    assert d.dtype == np.float32
    assert 0.0 <= d.min() and d.max() <= 1.0


def test_fit_to_aspect():
    assert fit_to_aspect(Image.new("RGB", (100, 100)), 16 / 9).size == (100, 56)
    assert fit_to_aspect(Image.new("RGB", (200, 100)), 16 / 9).size == (178, 100)


def test_upscale_nearest_requires_integer_multiple():
    img = Image.new("RGB", (10, 6))
    assert upscale_nearest(img, 40).size == (40, 24)
    with pytest.raises(ValueError):
        upscale_nearest(img, 45)


# --------------------------------------------------------------------- #
# 色板
# --------------------------------------------------------------------- #


def test_hex_roundtrip():
    pal = parse_hex_palette(["#000000", "#ffffff", "#ff0000", "#808080"])
    assert pal.shape == (4, 3)
    assert palette_to_hex(pal) == ["#000000", "#ffffff", "#ff0000", "#808080"]


def test_perceptual_roundtrip():
    x = np.linspace(0, 1, 9, dtype=np.float32).reshape(1, -1, 1).repeat(3, axis=2)
    assert np.allclose(from_perceptual(to_perceptual(x)), x, atol=1e-6)


def test_snap_exact_returns_palette_entries():
    pal = parse_hex_palette(["#000000", "#ffffff"])
    rng = np.random.default_rng(3)
    x = rng.random((5, 7, 3), dtype=np.float32)
    out = snap_exact(x, pal)
    for c in out.reshape(-1, 3):
        assert np.min(np.abs(pal - c).sum(axis=1)) < 1e-5


def test_hex_palette_used_in_perceptual_space():
    """hex 色板必须被转换到感知空间，否则中灰会匹配错亮度。"""
    pal = parse_hex_palette(["#808080"], perceptual=True)
    assert not np.allclose(pal, [128 / 255.0] * 3)
    assert np.allclose(pal, np.sqrt([128 / 255.0] * 3), atol=1e-6)


# --------------------------------------------------------------------- #
# pixelate 主体
# --------------------------------------------------------------------- #


def test_pixelate_output_shape_and_dtype():
    rng = np.random.default_rng(4)
    u8, pal = pixelate(rng.random((16, 24, 3), dtype=np.float32), PixelTail(n_colors=8))
    assert u8.shape == (16, 24, 3) and u8.dtype == np.uint8
    assert pal.ndim == 2 and pal.shape[1] == 3


def test_pixelate_only_emits_palette_colors():
    rng = np.random.default_rng(5)
    x = rng.random((24, 32, 3), dtype=np.float32)
    pal = parse_hex_palette(["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"])
    u8, _ = pixelate(x, PixelTail(edge_strength=0.0), palette=pal)
    ref = np.rint(pal * 255).astype(np.int16)
    flat = u8.reshape(-1, 3).astype(np.int16)
    dist = np.abs(flat[:, None, :] - ref[None, :, :]).max(axis=2).min(axis=1)
    assert dist.max() == 0, "输出颜色必须全部来自色板"


def test_pixelate_uniform_input():
    x = np.full((6, 6, 3), 0.5, dtype=np.float32)
    u8, _ = pixelate(x, PixelTail(dither=0.0, edge_strength=0.0),
                     palette=parse_hex_palette(["#808080"]))
    assert np.all(np.abs(u8.astype(int) - 128) <= 1)


def test_pixelate_rejects_bad_shape():
    with pytest.raises(ValueError):
        pixelate(np.zeros((4, 4), dtype=np.float32))


def test_pixelate_does_not_darken_overall():
    """⚠️ 回归测试：边缘压暗的梯度必须从**抖动前的内容**上算。

    曾经从"抖动后的结果"上算梯度 —— 抖动本身就是高频信号，于是整幅被判成
    "处处是边缘"，统一乘以 (1-0.35)，画面整体压暗约 30%。
    用户可见症状就是"像素化之后又灰又糊"。
    """
    # 用**平滑**内容：随机噪声本身就"处处是边缘"，测不出这个 bug。
    # 平滑内容下，唯一的高频信号来源就是抖动本身 —— 正好能暴露误触发。
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float32)
    v = 0.35 + 0.25 * (np.sin(xx / 9.0) * np.cos(yy / 7.0) + 1.0) / 2.0
    x = np.stack([v, v * 0.9, v * 0.8], axis=-1).astype(np.float32)
    u8, _ = pixelate(x, PixelTail(n_colors=32, edge_strength=0.35))
    delta = float(u8.mean()) / 255.0 - float(x.mean())
    assert abs(delta) < 0.05, f"整体亮度漂移过大: {delta:+.3f}"


def test_edge_darkening_still_works_on_real_edges():
    """修掉误触发之后，真正的边界仍应被压暗。"""
    x = np.full((32, 32, 3), 0.75, dtype=np.float32)
    x[:, 16:] = 0.25                       # 一条硬边界
    pal = parse_hex_palette(["#404040", "#bfbfbf"])
    off, _ = pixelate(x, PixelTail(dither=0.0, edge_strength=0.0), palette=pal)
    on, _ = pixelate(x, PixelTail(dither=0.0, edge_strength=0.35), palette=pal)
    assert on.mean() <= off.mean() + 1e-6, "边界处应当被压暗"
    # 但平坦区域（离边界远）不应受影响
    assert np.array_equal(on[:, :8], off[:, :8])


def test_palette_keeps_bright_neutral():
    """⚠️ 白墙被涂成绿墙的回归测试。

    median cut 按**像素数量**分配色板。当画面同时存在大面积的绿/蓝
    和一小块明亮的中性色（白墙/白雪/白衣服）时，那块中性色会被并进
    绿/蓝的盒子，代表色变成"发绿的浅色" —— 白墙就被涂成绿墙。
    """
    rng = np.random.default_rng(5)
    h = w = 96
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :int(w * 0.55)] = np.array([0.28, 0.56, 0.30])          # 大面积绿
    img[:, int(w * 0.55):] = np.array([0.16, 0.40, 0.62])          # 大面积蓝
    img[10:26, 40:56] = np.array([0.82, 0.83, 0.80])               # 一块白墙（约 5%）
    img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1).astype(np.float32)

    pal0 = palette_from_image(img, 24)
    pal1 = ensure_neutral_highlight(img, pal0)

    wall = np.array([0.82, 0.83, 0.80], dtype=np.float32)
    seed = np.sqrt(wall)
    d0 = np.sqrt(((pal0 - seed[None, :]) ** 2).sum(axis=1)).min()
    d1 = np.sqrt(((pal1 - seed[None, :]) ** 2).sum(axis=1)).min()
    assert d1 < 0.07, f"色板里仍没有白墙能吸附的中性亮色（最近距离 {d1:.3f}）"
    assert len(pal1) >= len(pal0), "补项不应减少既有色板"


def test_neutral_guard_rejects_a_tinted_bright_color():
    """⚠️ 白墙被涂成绿墙（第二次）—— 判据必须问「有没有中性项」，不能问「有没有近的项」。

    用户报障：`ref10_green_cliff` 里白色建筑被整体涂成绿色，和草地一个色。
    根因是色板里最亮的项是**偏绿的白**（`#c1d7aa` 那类），而旧的
    `ensure_neutral_highlight` 用「感知空间欧氏距离 < 0.07」判定"已经有中性色了"。
    `sqrt` 会把高亮度处的**色相差异压扁** —— 偏绿的白离真中性只有 0.06 左右，
    于是被判成"已有" → 不补项 → 白墙只能吸附到那个偏绿的项 → 变绿。

    这个测试直接构造那个陷阱色板：唯一的高亮项是个偏绿的白。
    旧判据会提前返回（长度不变）→ 断言失败；新判据看线性饱和度，正确补项。
    """
    img = np.full((96, 96, 3), np.array([0.20, 0.55, 0.22], np.float32))   # 大面积绿
    img[:, :40] = np.array([0.14, 0.38, 0.60], np.float32)                 # 大片蓝
    img[10:26, 40:56] = np.array([0.82, 0.83, 0.80], np.float32)           # 一块白墙

    pal = np.vstack([
        parse_hex_palette(["#c3d4b4"]),            # 偏绿的"白"：陷阱在这里
        parse_hex_palette(["#1c8a2e", "#135f22", "#2aa83c"]),
        parse_hex_palette(["#1d4e8a", "#0b2b52", "#2f6fb0"]),
        parse_hex_palette(["#101014"]),
    ])

    wall = np.array([0.82, 0.83, 0.80], np.float32)
    seed = np.sqrt(wall)
    d_before = float(np.sqrt(((pal - seed[None, :]) ** 2).sum(axis=1)).min())
    assert d_before < 0.07, f"测试前提不成立：陷阱色离中性太远（{d_before:.3f}），骗不到旧判据"

    out = ensure_neutral_highlight(img, pal)

    assert len(out) > len(pal), "色板里没有中性项时应当补一项，不能因为'有个偏绿的白离得近'就跳过"
    added = out[len(pal):]
    c = from_perceptual(added)
    sat = (c.max(axis=1) - c.min(axis=1)) / np.clip(c.max(axis=1), 1e-6, None)
    assert float(sat.max()) <= 0.10, f"补进来的项必须真正低饱和，实得 sat={sat.max():.3f}"


ASSET = Path(__file__).resolve().parents[1] / "assets" / "input" / "ref10_green_cliff.jpg"


@pytest.mark.skipif(not ASSET.exists(), reason="测试素材未入库")
def test_refine_palette_gives_bright_low_chroma_pixels_their_own_ramp():
    """⚠️ 白色建筑被涂成绿色（第三次）—— 槽位预算被多数派色区垄断。

    这次不是"缺一项"的问题。`ref10_green_cliff` 里植被占像素多数：
    median cut 按像素数量分槽位，于是**所有亮部槽位都给了绿**，
    明亮的灰白建筑一个都没有 → 只能吸到偏绿的项上 →
    建筑和草地变成一个颜色，玻璃/混凝土/阴影的分层全丢（用户报障）。

    而且旧守卫**拦不住**：它看到色板里有 `sat=0.21` 的"淡绿"就认定
    "已有中性色了"。**饱和度小 ≠ 中性，淡绿的饱和度本来就小。**

    修法：赎回**移除代价≈0 的冗余槽位**（这张图上是十几个挤在同一个窄亮度带
    里的蓝），换成明亮低饱和区域的专属色阶。删除数 == 新增数。

    用真实素材而不是合成图：这个 bug 依赖"多数派色区 + 罕见亮中性区"的真实
    像素分布，合成图只有个别随机种子能复现出来，测不稳。
    """
    src = Image.open(ASSET).convert("RGB")
    gh = 480
    gw = max(2, round(src.width * gh / src.height) // 2 * 2)
    grid = (gw, gh)
    ref = src.resize((grid[0] * 2, grid[1] * 2), Image.Resampling.LANCZOS)
    base = structure_aware_downsample(ref, grid, PixelTail().var_gain)

    pal0 = palette_from_image(base, 32)
    pal1 = refine_palette(base, pal0)

    # 1) 尺寸必须不变（删多少补多少）
    assert len(pal1) == len(pal0), f"色板尺寸应保持：{len(pal0)} → {len(pal1)}"

    # 2) 色板里必须出现真正的中性亮色（线性饱和度 <= 0.12）
    c = from_perceptual(pal1)
    sat = (c.max(axis=1) - c.min(axis=1)) / np.clip(c.max(axis=1), 1e-6, None)
    lum = base @ LUMA
    bright = c[lum.argmax() * 0 + (sat <= 0.12) & (pal1 @ LUMA >= np.percentile(base @ LUMA, 85))]
    assert len(bright) > 0, "修复后色板里仍没有明亮的中性项"

    # 3) 建筑区（明亮低饱和）实际吸附到的颜色必须更中性
    mx, mn = base.max(axis=-1), base.min(axis=-1)
    M = (lum >= np.percentile(lum, 88)) & ((mx - mn) / np.clip(mx, 1e-6, None) <= 0.20)
    assert M.sum() > 50, "测试前提：应当找得到建筑区"

    def absorbed_sat(pal):
        got = pal[((to_perceptual(base) - pal[:, None, None, :]) ** 2).sum(axis=-1).argmin(axis=0)]
        g = from_perceptual(got[M])
        return float(((g.max(axis=1) - g.min(axis=1)) / np.clip(g.max(axis=1), 1e-6, None)).mean())

    s0, s1 = absorbed_sat(pal0), absorbed_sat(pal1)
    assert s1 < s0 * 0.75, f"建筑区染色应明显减轻：{s0:.3f} → {s1:.3f}"


def test_refine_palette_keeps_size_and_error():
    """赎回槽位必须「删多少补多少」——色板尺寸不变，整体误差不上升。

    ⚠️ 原型阶段踩过：删了 14 个槽位却补 0 个，色板 32→18，误差直接翻倍。
    """
    rng = np.random.default_rng(12)
    h = w = 96
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = np.array([0.25, 0.50, 0.28], np.float32)
    img[:, :40] = np.array([0.06, 0.44, 0.58], np.float32)
    img[20:40, 50:70] = np.array([0.78, 0.80, 0.76], np.float32)
    img = np.clip(img + rng.normal(0, 0.010, img.shape), 0, 1).astype(np.float32)

    pal0 = palette_from_image(img, 24)
    pal1 = refine_palette(img, pal0)
    assert len(pal1) == len(pal0), f"色板尺寸应保持不变：{len(pal0)} → {len(pal1)}"

    q = to_perceptual(img).reshape(-1, 3)

    def err(pal):
        return float(np.sqrt(((q[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2).min(axis=1)).mean())

    e0, e1 = err(pal0), err(pal1)
    assert e1 <= e0 * 1.05, f"量化误差不应上升超过 5%：{e0:.5f} → {e1:.5f}"


def test_refine_palette_is_deterministic():
    """⚠️ 时序一致性：色板必须在同一输入上**逐位可复现**。

    视频模式下色板要跨帧共享，如果重算得到不同的色板，整片就会闪。
    （内部有抽样，所以抽样必须用固定 seed。）
    """
    rng = np.random.default_rng(13)
    img = rng.random((200, 160, 3), dtype=np.float32)
    pal = palette_from_image(img, 32)
    a = refine_palette(img, pal)
    b = refine_palette(img, pal)
    assert np.array_equal(a, b), "同一输入两次调用必须得到完全相同的色板"


def test_refine_palette_skips_when_already_neutral():
    """色板里已经有足够多的中性项时不应折腾（别把好色板改坏）。"""
    img = np.full((64, 64, 3), 0.55, dtype=np.float32)
    img[20:40, 20:40] = 0.80
    pal = parse_hex_palette(["#8c8c8c", "#dddddd", "#3a3a3a", "#555555", "#aaa9a8", "#1a1a1a"])
    out = refine_palette(img, pal)
    assert np.array_equal(out, pal), "已经有中性项时不应改动色板"


def test_ensure_neutral_highlight_skips_when_already_covered():
    img = np.full((32, 32, 3), 0.8, dtype=np.float32)              # 全图就是亮中性
    pal = parse_hex_palette(["#cccccc", "#333333"])
    out = ensure_neutral_highlight(img, pal)
    assert len(out) == len(pal), "已经有中性亮色时不应补项"


def test_ensure_neutral_highlight_skips_when_too_rare():
    img = np.full((64, 64, 3), 0.30, dtype=np.float32)
    img[0:2, 0:2] = 0.85                                            # 只有 4 个像素
    pal = parse_hex_palette(["#444444"])
    out = ensure_neutral_highlight(img, pal, min_fraction=0.05)
    assert len(out) == len(pal)


def test_dither_is_time_invariant():
    """⚠️ 架构约束：抖动相位只跟位置有关。同一输入两次调用必须逐像素一致。"""
    rng = np.random.default_rng(6)
    x = rng.random((32, 32, 3), dtype=np.float32)
    p = PixelTail(n_colors=8, dither=0.15)
    a, _ = pixelate(x, p)
    b, _ = pixelate(x, p)
    assert np.array_equal(a, b)


def test_dither_changes_result_and_stays_in_palette():
    grad = np.linspace(0.3, 0.7, 64, dtype=np.float32)[None, :, None].repeat(8, 0).repeat(3, 2)
    pal = parse_hex_palette(["#444444", "#888888"])
    plain, _ = pixelate(grad, PixelTail(dither=0.0, edge_strength=0.0), palette=pal)
    dith, _ = pixelate(grad, PixelTail(dither=0.20, edge_strength=0.0), palette=pal)
    assert not np.array_equal(plain, dith), "抖动应当确实改变输出"
    assert len(np.unique(dith.reshape(-1, 3), axis=0)) >= 2


def test_shared_palette_keeps_color_set_bounded():
    """⚠️ 架构约束：多帧共用色板时，全部输出颜色必须落在同一组里。"""
    rng = np.random.default_rng(7)
    frames = [rng.random((16, 16, 3), dtype=np.float32) for _ in range(4)]
    pal = palette_from_frames(frames, n_colors=16)
    used = set()
    for f in frames:
        u8, _ = pixelate(f, PixelTail(n_colors=16, edge_strength=0.0), palette=pal)
        used.update(map(tuple, u8.reshape(-1, 3).tolist()))
    assert len(used) <= len(pal)


def test_render_frame_end_to_end():
    yy, xx = np.mgrid[0:180, 0:320]
    img = np.stack([(xx % 256), (yy % 256), ((xx + yy) % 256)], axis=-1).astype(np.uint8)
    out, pal = render_frame(Image.fromarray(img), (80, 45), PixelTail(n_colors=12))
    assert out.size == (80, 45)
    assert pal.shape[1] == 3


def test_palette_from_image_shape():
    rng = np.random.default_rng(8)
    pal = palette_from_image(rng.random((40, 40, 3), dtype=np.float32), n_colors=10)
    assert pal.ndim == 2 and pal.shape[1] == 3 and 1 <= len(pal) <= 10


def test_quantize_continuous_keeps_colors_beyond_palette():
    """连续色模式：跳过吸附 —— 输出颜色不受色板约束，但抖动/边缘仍生效。"""
    import numpy as np
    from dataclasses import replace
    rng = np.random.default_rng(3)
    img = rng.uniform(0.2, 0.9, (48, 64, 3)).astype(np.float32)
    pt = replace(PixelTail(n_colors=8), quantize_mode="continuous")
    out, pal = pixelate(img, pt, palette=None)
    arr = np.asarray(out)
    uniq = len(np.unique(arr.reshape(-1, 3), axis=0))
    assert len(pal) <= 8 + 2                     # 色板仍是 8 色（未被消费也不报错）
    assert uniq > 32                             # 输出远超色板约束 → 吸附确实被跳过
    # 抖动仍生效：dither=0 与 0.1 的输出不同
    pt0 = replace(pt, dither=0.0)
    out0, _ = pixelate(img, pt0, palette=None)
    assert not np.array_equal(np.asarray(out0), arr)


def test_quantize_palette_mode_still_bounds_colors():
    """对照：palette 模式输出颜色数 ≤ 色板数。"""
    import numpy as np
    rng = np.random.default_rng(4)
    img = rng.uniform(0.2, 0.9, (48, 64, 3)).astype(np.float32)
    pt = PixelTail(n_colors=8)
    out, pal = pixelate(img, pt, palette=None)
    uniq = len(np.unique(np.asarray(out).reshape(-1, 3), axis=0))
    # ⚠️ palette=None 时 refine/ensure 可能增长色板（8→9~10），上限按派生色板算
    assert uniq <= len(pal)
