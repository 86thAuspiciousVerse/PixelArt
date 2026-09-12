"""调参参数集与预览渲染的单元测试。

M3 的界面本身没有被测试覆盖（那是浏览器的事），但**参数映射**与**预览渲染**
必须覆盖 —— 因为它们是"界面调好了、命令行跑出来不一样"这类问题的唯一防线。

这里固化的几条都是踩过坑的设计决定，不是形式化的覆盖：
- 预览**保持时长只降帧率**（否则运动速度不是所见即所得）
- 预览**共用一块色板**（否则看到根本不存在的闪烁）
- ``AUTO`` 哨兵值的语义
"""

import numpy as np
import pytest

from pixelart.palette import from_perceptual, palette_from_frames
from pixelart.pipeline import AnimParams, Scene
from pixelart.pixelate import PixelTail
from pixelart.tune import (
    AUTO,
    TuneParams,
    aspect_value,
    build_scene,
    display_scale,
    preview_stats,
    render_preview,
    render_still,
    render_sweep,
)


def fake_scene(gh: int = 40, gw: int = 64, tail_colors: int = 16) -> Scene:
    """不依赖深度模型的最小场景（测参数映射不需要真实图片）。"""
    rng = np.random.default_rng(0)
    grad = np.linspace(0.15, 0.55, gh, dtype=np.float32)[:, None].repeat(gw, 1)
    base = np.stack([grad, grad * 1.05, grad * 1.15], -1)
    base += rng.normal(0, 0.01, base.shape).astype(np.float32)
    base[4:8, 44:50] = 0.95
    return Scene(
        base=np.clip(base, 0, 1).astype(np.float32),
        far=np.linspace(0.05, 0.95, gh, dtype=np.float32)[:, None].repeat(gw, 1),
        grid=(gw, gh), scale=2, light_xy=(0.73, 0.15),
        fog_color=np.array([0.5, 0.6, 0.78], np.float32),
        tail=PixelTail(n_colors=tail_colors), src_size=(gw, gh),
    )


# ------------------------------------------------------------ 参数序列化
def test_query_round_trip_is_exact():
    """参数经 URL 往返必须**完全一致** —— 界面靠它拼 <img src>。"""
    p = TuneParams(density=0.83, rays_x=0.31, rays_auto_center=False,
                   colors=24, aspect="16:9", dust_count=170, seconds=4.5)
    qs = dict(kv.split("=", 1) for kv in p.to_query().split("&"))
    assert TuneParams.from_query(qs).to_dict() == p.to_dict()


def test_bool_survives_query():
    """布尔字段在 URL 里是 0/1，还原时必须回到布尔。"""
    assert TuneParams.from_query({"rays_auto_center": "0"}).rays_auto_center is False
    assert TuneParams.from_query({"rays_auto_center": "1"}).rays_auto_center is True
    assert TuneParams.from_query({"rays_auto_center": "true"}).rays_auto_center is True


def test_bad_values_are_ignored_not_raised():
    """界面会把用户输入直接拼进 URL —— 非法值必须静默忽略，不能 500。"""
    p = TuneParams.from_query({"density": "abc", "colors": "", "aspect": "16:9"})
    assert p.density == TuneParams().density        # 回退默认
    assert p.aspect == "16:9"                       # 合法值仍生效


def test_unknown_fields_are_ignored():
    TuneParams.from_query({"nonsense": "1", "density": "0.5"})


# ------------------------------------------------------------ AUTO 语义
def test_auto_sentinel_semantics():
    """⚠️ AUTO 用负数表示"自动估计"，三个通道**任一**为 AUTO 就整体走自动。

    这个哨兵值必须和 [0,1] 的合法色彩值区分开，否则"自动"和"纯黑"会混淆。
    """
    assert AUTO < 0, "AUTO 必须是负数，才不会与合法色值冲突"
    assert TuneParams().fog_color() is None
    assert TuneParams().rays_color() is None
    assert TuneParams(fog_r=0.0, fog_g=0.0, fog_b=0.0).fog_color() == (0.0, 0.0, 0.0)
    assert TuneParams(fog_r=0.1, fog_g=-1, fog_b=0.2).fog_color() is None


def test_light_xy_falls_back_to_scene():
    """光源位置自动时用 scene 估计的，手动时用参数里的。"""
    sc = fake_scene()
    auto = TuneParams(rays_auto_center=True)
    assert auto.light_xy(sc) == sc.light_xy
    manual = TuneParams(rays_auto_center=False, rays_x=0.2, rays_y=0.8)
    assert manual.light_xy(sc) == (0.2, 0.8)


# ------------------------------------------------------------ 参数映射
def test_to_tail_and_anim_map_the_right_fields():
    p = TuneParams(colors=24, dither=0.2, edge_strength=0.4, edge_gain=3.0,
                   var_gain=50, fog_drift=0.5, flicker=0.3, dust_count=120)
    t = p.to_tail()
    a = p.to_anim()
    assert (t.n_colors, t.dither, t.edge_strength, t.edge_gain, t.var_gain) == (24, 0.2, 0.4, 3.0, 50)
    assert (a.fog_drift, a.light_flicker, a.dust_count) == (0.5, 0.3, 120)
    assert isinstance(t, PixelTail) and isinstance(a, AnimParams)


def test_aspect_parsing():
    assert aspect_value("native") == 0.0
    assert aspect_value("keep") == 0.0
    assert abs(aspect_value("16:9") - 16 / 9) < 1e-9
    assert aspect_value("2.16") == 2.16
    assert aspect_value(1.5) == 1.5


# ------------------------------------------------- ⭐ 预览必须"同速"
@pytest.mark.parametrize("fps", [6, 8, 10, 12, 15])
def test_preview_keeps_duration_and_only_drops_fps(fps):
    """⚠️ 预览**只降帧率、不改时长** —— 这条最容易做错。

    动画速度由归一化相位 ``t`` 的扫法决定。如果预览靠"少渲几帧"提速，
    ``t`` 就会在更短的时间里走完一个循环，雾漂移和粒子会**快好几倍** ——
    参数就不再是所见即所得了。

    正确做法：渲 ``时长 × preview_fps`` 帧，让 ``t`` 在同样的墙钟时间里走完。
    """
    sc = fake_scene()
    p = TuneParams(seconds=6.0)
    n = p.preview_frames(sc, preview_fps=fps)
    assert n == int(round(6.0 * fps))
    # 帧率 = 帧数 / 时长，与成片同速（只是更卡顿）
    assert abs(n / p.seconds - fps) < 1e-9


def test_preview_needs_at_least_two_frames():
    assert TuneParams(seconds=0.1).preview_frames(fake_scene(), preview_fps=2) >= 2


# ------------------------------------------------- ⭐ 预览共用一块色板
def test_preview_uses_one_shared_palette():
    """全片颜色必须都属于同一块色板（时序铁律 1）。

    逐帧各求色板 → 播放时色板漂移 = 人为造出闪烁。
    """
    sc = fake_scene()
    p = TuneParams(seconds=2, dust_count=60, fog_drift=0.4)
    frames, pal, stats = render_preview(sc, p, preview_fps=6, include_last=True)
    assert stats["loop_ok"] is True, f"循环不闭合：{stats['loop_diff']}"
    assert stats["colors_outside"] == 0

    pal_u8 = {tuple(int(v) for v in row) for row in
              np.clip(from_perceptual(pal) * 255.0, 0, 255).astype(np.uint8)}
    for i, f in enumerate(frames):
        assert set(map(tuple, f.reshape(-1, 3).tolist())) <= pal_u8, f"第 {i} 帧有色板外的颜色"


def test_passing_a_palette_reuses_it():
    """外部传入色板时必须**直接用它**，不能自己另求一块。"""
    sc = fake_scene()
    p = TuneParams(seconds=1, dust_count=0)
    custom = palette_from_frames([sc.base], n_colors=6)
    frames, pal, _ = render_preview(sc, p, preview_fps=4, palette=custom)
    assert np.array_equal(pal, custom), "传入的色板没有被沿用"


# --------------------------------------------------- 静态必须真的静态
def test_zero_animation_is_bit_identical_across_time():
    """动画幅度全关时，各帧必须逐位相同 —— 预览也要满足这条。"""
    sc = fake_scene()
    p = TuneParams(seconds=2, fog_drift=0.0, flicker=0.0, dust_count=0)
    frames, _, stats = render_preview(sc, p, preview_fps=6, include_last=False)
    for f in frames[1:]:
        assert np.array_equal(f, frames[0]), "关掉动画后各帧还是不同"
    assert stats["amplitude"]["static"] == 1.0


def test_animation_switches_actually_differ():
    """反过来：打开动画后各帧必须真的不同（否则滑杆是坏的）。"""
    sc = fake_scene()
    p = TuneParams(seconds=2, fog_drift=0.5, flicker=0.3, dust_count=80)
    frames, _, stats = render_preview(sc, p, preview_fps=6, include_last=False)
    assert not np.array_equal(frames[0], frames[len(frames) // 2])
    assert stats["amplitude"]["strong"] > 0


# ------------------------------------------------------------ 其它
def test_display_scale_is_integer_and_reasonable():
    """预览放大必须是**整数倍** —— 非整数倍会让方块参差，预览失去参考价值。"""
    sc = fake_scene(gw=240, gh=128)
    for target in (300, 720, 1080):
        s = display_scale(sc, target)
        assert isinstance(s, int) and s >= 1
        assert abs(max(sc.grid) * s - target) <= max(sc.grid) / 2 + 1


def test_render_still_shape_matches_grid():
    sc = fake_scene(gw=64, gh=40)
    u8, pal = render_still(sc, TuneParams(colors=12))
    assert u8.shape == (40, 64, 3) and u8.dtype == np.uint8
    # ⚠️ colors 是**目标值**不是硬上限：refine_palette 赎回槽位是 1:1 换
    #    （尺寸不变），但 ensure_neutral_highlight 可能再补一项 → 最多 +1。
    assert 1 <= len(pal) <= 12 + 1


def test_colors_param_takes_effect_without_rebuilding_scene():
    """⚠️「色板颜色数」改了必须**立刻生效**，不需要重建场景。

    踩过的坑：渲染路径读的是 ``scene.tail``（建场景那一刻的快照），
    而 ``pixelate`` 是从 tail 里取 ``n_colors`` 的 —— 于是改颜色数滑杆
    完全没反应，画面色阶数不变，而且不报错（很容易以为是自己看错了）。
    修法：渲染一律用 ``params.to_tail()``。
    """
    sc = fake_scene(tail_colors=32)          # 场景是用 32 色建的
    for want in (4, 8, 24):
        _, pal = render_still(sc, TuneParams(colors=want))
        assert len(pal) <= want + 1, (
            f"colors={want} 时色板却有 {len(pal)} 项 —— 说明读了 scene.tail 而非参数")


def test_dither_param_takes_effect_without_rebuilding_scene():
    """同理：抖动强度逐帧可调，也必须立刻生效（不必重建场景）。"""
    sc = fake_scene()
    a, _ = render_still(sc, TuneParams(dither=0.0, dust_count=0))
    b, _ = render_still(sc, TuneParams(dither=0.35, dust_count=0))
    assert not np.array_equal(a, b), "抖动强度改了却没影响输出"


def test_render_sweep_varies_the_requested_parameter():
    sc = fake_scene()
    p = TuneParams(dust_count=0)
    out = render_sweep(sc, p, "density", [0.0, 2.0])
    assert [v for v, _ in out] == [0.0, 2.0]
    a, b = out[0][1].astype(np.int32), out[1][1].astype(np.int32)
    assert not np.array_equal(a, b), "扫描取不同值却给出相同结果"


def test_manual_light_position_actually_takes_effect():
    """⚠️ 界面的「光源位置」手动 X/Y 滑杆必须真的起作用。

    踩过的坑：``compose_frame`` 里硬用了 ``scene.light_xy``，
    ``TuneParams.light_xy(scene)`` 虽然写了但**从没被调用** ——
    手动滑杆完全无效，而且不报错。实测 XY=(0.2,0.2) 与 (0.5,0.5)
    输出**逐位相同**。

    这类"参数定义了但没接线"的 bug 很难靠肉眼发现：
    界面能拖、有数值、画面也像是变了（因为别的因素在动），
    只有把两个极端取值渲出来比对才能确定。
    """
    sc = fake_scene()
    a, _ = render_still(sc, TuneParams(rays_auto_center=False, rays_x=0.15, rays_y=0.15))
    b, _ = render_still(sc, TuneParams(rays_auto_center=False, rays_x=0.85, rays_y=0.85))
    assert not np.array_equal(a, b), "手动光源位置的两个极端取值给出相同输出 —— 滑杆没接线"

    # 自动模式必须忽略手动值
    c, _ = render_still(sc, TuneParams(rays_auto_center=True, rays_x=0.15, rays_y=0.15))
    d, _ = render_still(sc, TuneParams(rays_auto_center=True, rays_x=0.85, rays_y=0.85))
    assert np.array_equal(c, d), "自动模式下不该受手动值影响"


def test_fog_saturation_cap_reduces_far_tint():
    """⚠️ 雾色的饱和度上限必须真的压住——这是「整幅图像蒙了层滤镜」的根源。

    实测 10 张素材里 7 张的自动雾色饱和 >0.15，ref10 达 0.79
    （它拿一片饱和蓝天当了雾色），而雾覆盖全部远景。
    """
    from pixelart.compose import auto_fog_color

    sc = fake_scene()
    capped = auto_fog_color(sc.base, sc.far, sat_max=0.40)

    def sat(c):
        c = np.asarray(c, np.float32).ravel()
        return float((c.max() - c.min()) / max(c.max(), 1e-6))

    assert sat(capped) <= 0.401, f"上限没生效：{sat(capped):.3f}"
    # 默认值必须就是 0.40（否则"修复"只在显式传参时才有）
    assert sat(auto_fog_color(sc.base, sc.far)) <= 0.401, "默认值没带上限"


def test_screen_falloff_confines_the_light():
    """⚠️ 体积光的屏幕空间收敛必须真的把光"收"到光源附近。

    原来只按**深度差**衰减，与光源深度相近的像素会均匀吃到带色的光 ——
    观感就是整幅图被往某个色相上拉。加上屏幕衰减后，
    近处贡献占比应明显提高、全图注入量明显下降。
    """
    from pixelart.compose import depth_fog, volumetric_light

    sc = fake_scene(gh=64, gw=96)
    fogged = depth_fog(sc.base, sc.far, color=sc.fog_color, density=0.7, power=3.0)
    gh, gw = sc.grid[1], sc.grid[0]
    yy, xx = np.mgrid[0:gh, 0:gw]
    d = np.hypot(xx / gw - sc.light_xy[0], yy / gh - sc.light_xy[1])

    off = volumetric_light(fogged, sc.far, sc.light_xy, strength=0.55, screen_falloff=0.0)
    on = volumetric_light(fogged, sc.far, sc.light_xy, strength=0.55, screen_falloff=1.0)

    near_off = off[d < 0.25].sum() / max(off.sum(), 1e-9)
    near_on = on[d < 0.25].sum() / max(on.sum(), 1e-9)
    assert near_on > near_off, f"收敛后近处占比没提高：{near_off:.3f} → {near_on:.3f}"
    assert on.mean() < off.mean(), "收敛后全图注入量应当下降"


def test_stats_report_fog_saturation_and_ray_color():
    """自检统计必须暴露雾色饱和度与实际散射色 —— 界面靠它标出"过饱和"。

    ⚠️ 散射色的采样输入要和产线一致（从**雾后**的图取），
    否则会得出误导性的值。第一版从 ``scene.base`` 取，得到纯白 [1,1,1]。
    """
    from pixelart.palette import palette_from_frames
    from pixelart.pipeline import AnimParams, compose_frame, finish_frame

    sc = fake_scene()
    p = TuneParams(seconds=1, dust_count=0)
    kw = p.compose_kwargs(sc)
    far = []
    for i in range(2):
        c, ct = compose_frame(sc, t=i / 2, return_content=True, **kw)
        far.append(finish_frame(c, p.to_tail(), palette=None, edge_ref=ct)[0])
    pal = palette_from_frames([far[0]], n_colors=16)
    st = preview_stats(far + [far[0]], pal, 2, p, sc)

    assert "fog_sat_actual" in st and 0.0 <= st["fog_sat_actual"] <= 1.0
    rc = st["rays_color_used"]
    assert len(rc) == 3 and all(0.0 <= v <= 1.0 for v in rc)
    assert max(rc) > 0.05, f"散射色统计异常（不该是全黑）：{rc}"
    # 必须是**低饱和**的（空气散射近乎无色）
    mx, mn = max(rc), min(rc)
    assert (mx - mn) / max(mx, 1e-6) <= 0.30, f"散射色统计饱和度偏高：{rc}"


# ══ 参数接线：每个旋钮都必须真的到达渲染调用 ══
def test_manual_fog_color_overrides_auto():
    """⚠️ 手动雾色（fog_r/g/b）必须覆盖自动估计。

    踩过的坑：``compose_frame`` 硬传 ``color=scene.fog_color``，
    ``TuneParams.fog_color()`` 虽然写了但**从没被调用** ——
    手动雾色完全无效（实测改与不改输出逐位相同）。
    这是"参数定义了但没接进渲染调用"这一类 bug 的第 3 次出现
    （前两次是 light_xy、fog_sat）。
    """
    sc = fake_scene()
    auto, _ = render_still(sc, TuneParams(dust_count=0))
    manual, _ = render_still(sc, TuneParams(dust_count=0, fog_r=0.95, fog_g=0.05, fog_b=0.05))
    assert not np.array_equal(auto, manual), "手动雾色没生效 —— 又没接线"


def test_fog_saturation_cap_is_live():
    """⚠️ ``fog_sat`` 滑杆必须**每帧**生效，不能只在建场景那一刻算一次。

    踩过的坑：``prepare_scene`` 把雾色算好（含上限）存进 scene，
    之后 ``compose_frame`` 直接用那个固定值 —— 于是滑杆拖动毫无反应。
    修法：scene 存**原始**雾色（sat_max=1.0），上限在每帧施加。
    """
    sc = fake_scene()
    lo, _ = render_still(sc, TuneParams(dust_count=0, fog_sat=0.0))
    hi, _ = render_still(sc, TuneParams(dust_count=0, fog_sat=0.8))
    assert not np.array_equal(lo, hi), "fog_sat 改了没反应 —— 上限没在每帧施加"


def test_dust_and_all_anim_params_reach_the_sequence_path():
    """⚠️ 尘埃等动画参数必须同时影响**单帧**与**整段**两条路径。

    用户报障：尘埃控件似乎只影响静止帧，一点播放就跳回预设效果。
    界面侧的根因是 frames[] 没失效；但 Python 侧也要保证两条路径一致 ——
    所以这里逐个字段验"改了它，单帧和整段都变了"。
    """
    import dataclasses

    sc = fake_scene(gh=48, gw=64)
    base = TuneParams(seconds=1.0, fps=8, grid_long=240, work_long=960)
    s0, _ = render_still(sc, base)
    p0, _, _ = render_preview(sc, base, preview_fps=4, include_last=False)

    cases = {
        "dust_count": {"dust_count": 0},
        "dust_bright": {"dust_bright": 1.5},
        "dust_twinkle": {"dust_twinkle": 0.0},
        "dust_fade_far": {"dust_fade_far": 0.0},
        "dust_light_boost": {"dust_light_boost": 3.0},
        "flicker": {"flicker": 0.5},
        "fog_drift": {"fog_drift": 0.9},
    }
    for name, ov in cases.items():
        p = dataclasses.replace(base, **ov)
        s1, _ = render_still(sc, p)
        q, _, _ = render_preview(sc, p, preview_fps=4, include_last=False)
        n = min(len(p0), len(q))
        d_seq = max(float(np.abs(p0[i].astype(np.int16) - q[i].astype(np.int16)).mean())
                    for i in range(n))
        assert not np.array_equal(s0, s1), f"{name} 没影响单帧"
        assert d_seq > 0, f"{name} 没影响整段序列"
