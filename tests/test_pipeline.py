"""M2 视频管线的单元测试 —— 全部固化**时序一致性铁律**。

这是整个项目里最重要的一组护栏：M1（静态）出错了看得见，M2（视频）出错了
往往只是"看起来有点闪"，很难归因。所以循环性、色板共享、静态/视频同源
这三件事必须由测试来守。

测试用**合成图**（快、确定），不依赖 10 张素材和深度模型 ——
深度模型只在需要真场景的地方用，且用最小网格。
"""

import numpy as np
import pytest
from PIL import Image

from pixelart.animate import loop_noise_2d
from pixelart.palette import palette_from_frames
from pixelart.pipeline import (
    AnimParams,
    Scene,
    check_loop,
    compose_frame,
    finish_frame,
    grid_from_aspect,
    mean_abs_diff,
)
from pixelart.pixelate import PixelTail


def make_scene(gh=32, gw=48, seed=0) -> Scene:
    """造一个不依赖深度模型的最小场景。

    深度用竖直渐变（近 → 远），模拟"下近上远"，足以驱动雾与体积光。
    """
    rng = np.random.default_rng(seed)
    base = rng.random((gh, gw, 3), dtype=np.float32) * 0.6 + 0.15
    base[:4] = np.array([0.55, 0.72, 0.88], dtype=np.float32)     # 一条"天空"
    base[gh - 8] = 0.9                                            # 一个亮点当光源
    far = np.linspace(0.05, 0.95, gh, dtype=np.float32)[:, None].repeat(gw, 1)
    return Scene(
        base=base.astype(np.float32),
        far=far,
        grid=(gw, gh),
        scale=2,
        light_xy=(0.5, float(gh - 8) / gh),
        fog_color=np.array([0.5, 0.6, 0.75], dtype=np.float32),
        tail=PixelTail(n_colors=16),
        src_size=(gw, gh),
    )


# ------------------------------------------------- ⭐ 铁律 1：循环无缝
def test_compose_frame_is_periodic_in_t():
    """``compose_frame(t=1.0)`` 必须等于 ``compose_frame(t=0.0)``。

    这是「4 秒视频首尾能接上」的根本保证。如果不成立，
    循环播放时会在接缝处看到一次跳变。
    """
    sc = make_scene()
    a = AnimParams()
    f0 = compose_frame(sc, t=0.0, anim=a)
    f1 = compose_frame(sc, t=1.0, anim=a)
    assert np.abs(f0 - f1).max() < 1e-6, f"合成在 t=1 处不闭合（{np.abs(f0 - f1).max():.3e}）"


def test_full_pipeline_loop_is_bit_exact():
    """整条管线（合成 → 量化）之后，第 0 帧与第 N 帧必须**逐位相同**。

    注意这里必须过 ``pixelate`` —— 量化是阶跃函数，合成阶段差 1e-6
    有可能被量化放大成整整一个色阶。所以要在**最终输出**上验。
    """
    sc = make_scene()
    a = AnimParams(fog_drift=0.5, light_flicker=0.4, dust_count=60)
    n = 12
    frames = [finish_frame(compose_frame(sc, t=i / n, anim=a), sc.tail, palette=None)[0]
              for i in range(n + 1)]

    res = check_loop(frames, n_frames=n)
    assert res["ok"], f"循环有接缝：最大通道差 {res['max_diff']}，位置 {res['where']}"


def test_all_combinations_of_anim_params_stay_periodic():
    """逐个旋钮单独打开都必须保持循环 —— 不能只有"全关"时才对。"""
    sc = make_scene()
    for kwargs in ({"fog_drift": 0.5}, {"light_flicker": 0.4}, {"dust_count": 80},
                   {"fog_drift": 0.5, "light_flicker": 0.4, "dust_count": 80}):
        a = AnimParams(**kwargs)
        f0 = compose_frame(sc, t=0.0, anim=a)
        f1 = compose_frame(sc, t=1.0, anim=a)
        assert np.abs(f0 - f1).max() < 1e-6, f"{kwargs} 下 t=1 不闭合"


# ------------------------------------------------- ⭐ 铁律 2：静态图不受 t 影响
def test_static_scene_with_no_anim_is_time_invariant():
    """动画幅度全关时，任何 t 都必须给出**完全相同**的帧。

    这样"静态图"就真的只是"视频的第 0 帧"，而不是"看起来一样的另一条路径"。
    """
    sc = make_scene()
    a = AnimParams(fog_drift=0.0, light_flicker=0.0, dust_count=0)
    ref = compose_frame(sc, t=0.0, anim=a)
    for t in (0.13, 0.5, 0.87):
        assert np.array_equal(compose_frame(sc, t=t, anim=a), ref)


# ------------------------------------------------- ⭐ 铁律 3：全片共用色板
def test_shared_palette_covers_every_frame():
    """全片所有帧的颜色必须都是那块**共用色板**里的颜色。

    逐帧各求色板会让颜色集合逐帧漂移 —— 观感就是整片闪烁。
    """
    sc = make_scene()
    a = AnimParams(fog_drift=0.5, light_flicker=0.3, dust_count=40)
    n = 8
    composed = [compose_frame(sc, t=i / n, anim=a) for i in range(n)]
    pal = palette_from_frames(composed, n_colors=sc.tail.n_colors, max_samples=8)

    from pixelart.palette import from_perceptual

    pal_u8 = {tuple(int(v) for v in row) for row in
              np.clip(from_perceptual(pal) * 255.0, 0, 255).astype(np.uint8)}

    for i, c in enumerate(composed):
        u8, _ = finish_frame(c, sc.tail, palette=pal)
        got = set(map(tuple, u8.reshape(-1, 3).tolist()))
        assert got <= pal_u8, f"第 {i} 帧出现了不属于共用色板的颜色"


def test_frame_is_bit_identical_across_calls():
    """同一 (输入, t, 色板) 必须可复现 —— 否则谈不上一致性。"""
    sc = make_scene()
    a = AnimParams(fog_drift=0.4, light_flicker=0.3, dust_count=50)
    c1 = compose_frame(sc, t=0.37, anim=a)
    c2 = compose_frame(sc, t=0.37, anim=a)
    assert np.array_equal(c1, c2), "同一输入两次调用结果不同（有未固定的随机源）"

    p = palette_from_frames([c1], n_colors=16)
    u1, _ = finish_frame(c1, sc.tail, palette=p)
    u2, _ = finish_frame(c2, sc.tail, palette=p)
    assert np.array_equal(u1, u2)


# ------------------------------------------------- ⭐ 铁律 4：抖动相位不含 t
def test_pixelate_never_receives_time():
    """``finish_frame`` 的签名里**不能有 t** —— 抖动相位一旦含时间就是整屏闪烁。

    这个测试通过签名反射来守：以后有人想加 t，会在这里被拦下。
    """
    import inspect

    sig = inspect.signature(finish_frame)
    assert "t" not in sig.parameters, "finish_frame 不该接收时间参数（抖动相位必须锁屏幕空间）"
    assert not any("time" in p for p in sig.parameters), "finish_frame 不该接收任何时间参数"


# ------------------------------------------------------------------- 其他
def test_animation_actually_moves():
    """周期性很容易被写成"恒定不变"——那也满足循环，但画面是死的。"""
    sc = make_scene()
    a = AnimParams(fog_drift=0.5, light_flicker=0.4, dust_count=80)
    f0 = finish_frame(compose_frame(sc, t=0.0, anim=a), sc.tail, palette=None)[0]
    fh = finish_frame(compose_frame(sc, t=0.5, anim=a), sc.tail, palette=None)[0]
    assert mean_abs_diff(f0, fh) > 0.05, "半周期处几乎没变化，动画等于没做"


def test_grid_from_aspect_is_even_and_keeps_ratio():
    for (w, h) in ((1920, 1080), (813, 1348), (2560, 1080)):
        gw, gh = grid_from_aspect(w, h, 480)
        assert gw % 2 == 0 and gh % 2 == 0, "网格必须取偶数"
        assert max(gw, gh) == 480 or max(gw, gh) == 480, "长边应当等于给定值"
        assert abs((gw / gh) - (w / h)) < 0.02, f"宽高比变了：{gw / gh:.3f} vs {w / h:.3f}"


def test_check_loop_detects_a_real_seam():
    """自检本身要有效：人为造一个接缝，必须被查出来。"""
    sc = make_scene()
    frames = [finish_frame(compose_frame(sc, t=i / 6, anim=AnimParams()), sc.tail,
                           palette=None)[0] for i in range(6)]
    frames.append(frames[0] + 1)          # 人为把末帧改掉
    res = check_loop(frames, n_frames=6)
    assert not res["ok"], "自检没发现人为接缝"
    assert res["max_diff"] > 0


def test_loop_noise_does_not_drift_with_long_time():
    """长时间后仍要循环（不能用 t 的整数部分累积误差）。"""
    a = loop_noise_2d(0.5, (16, 24), seed=1)
    for k in (1, 2, 5, 20):
        b = loop_noise_2d(0.5 + k, (16, 24), seed=1)
        assert np.abs(a - b).max() < 1e-6, f"t+{k} 处不循环"


def _smooth_scene(gh: int = 64, gw: int = 96, sigma: float = 0.01, seed: int = 0) -> Scene:
    """平滑内容 + 少量真实边界 + 一个亮光源。

    ⚠️ 不要用纯随机噪声当测试内容：噪声本身就让边缘检测器处处饱和，
    粒子再加多少边界都"看不出来差异"，测不出这个 bug。
    真实照片降采样之后是**大部分平滑 + 少量硬边**，这里照这个比例造。
    """
    rng = np.random.default_rng(seed)
    grad = np.linspace(0.15, 0.55, gh, dtype=np.float32)[:, None].repeat(gw, 1)
    base = np.stack([grad, grad * 1.05, grad * 1.15], -1)
    base = base + rng.normal(0, sigma, base.shape).astype(np.float32)
    base[40:, :] *= 0.5                       # 一条真实的内容硬边界
    base[6:12, 66:74] = 0.95                  # 一个亮光源
    return Scene(
        base=np.clip(base, 0, 1).astype(np.float32),
        far=np.linspace(0.05, 0.95, gh, dtype=np.float32)[:, None].repeat(gw, 1),
        grid=(gw, gh), scale=2, light_xy=(0.73, 0.14),
        fog_color=np.array([0.5, 0.6, 0.78], np.float32),
        tail=PixelTail(n_colors=32), src_size=(gw, gh),
    )


# ⭐ 粒子不能变成"黑点"：边缘压暗只许看内容，不许看大气
def test_edge_ref_prevents_dust_becoming_black_dots():
    """⚠️ 用户报障「画面上有黑点在移动」——纯加性的粒子被边缘压暗搞成了黑点。

    机制：粒子是"单个亮像素 + 周围暗像素"，在边缘检测器眼里是**极强的边界**。
    边缘压暗把 `g` 饱和到 1.0，于是粒子自己与相邻像素被乘上 (1−0.35)，
    在深色画面上就是好几个色阶的落差。实测（真实素材 ref04）：
    单帧 **3292** 个像素因粒子而变暗，最深 **82/255** 色阶。

    修法：把"不含粒子"的合成图当边缘参考（`edge_ref`）传给像素尾巴。
    颗粒子是**大气现象，不是内容**，而边缘压暗是内容风格化步骤。

    这与约束 5（梯度必须从抖动前的内容上求）是同一道理的另一层：
    **也别从瞬时大气上求。**

    ⚠️ 判据方向别搞反：要数的是「**有粒子的版本比没粒子的版本更暗**」的像素
    （= 黑点）。数成"更亮"会把粒子正常该有的加性提亮算进来，结论完全反过来。
    """
    sc = _smooth_scene()
    a_on = AnimParams(fog_drift=0.0, light_flicker=0.0, dust_count=150, dust_bright=0.55)
    a_off = AnimParams(fog_drift=0.0, light_flicker=0.0, dust_count=0)

    c_on, ct_on = compose_frame(sc, t=0.3, anim=a_on, return_content=True)
    c_off, ct_off = compose_frame(sc, t=0.3, anim=a_off, return_content=True)
    assert np.array_equal(ct_on, ct_off), "关掉粒子不该改变边缘参考图"

    def luma(x):
        return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    def black_dots(edge_ref_on, edge_ref_off) -> int:
        u_on, _ = finish_frame(c_on, sc.tail, palette=PAL, edge_ref=edge_ref_on)
        u_off, _ = finish_frame(c_off, sc.tail, palette=PAL, edge_ref=edge_ref_off)
        d = luma(u_on.astype(np.float32)) - luma(u_off.astype(np.float32))
        return int((d <= -2).sum())

    PAL = palette_from_frames([c_on], n_colors=32)

    bad = black_dots(None, None)                 # 旧行为：边缘参考含粒子
    ok = black_dots(ct_on, ct_off)               # 修复后：边缘参考不含粒子

    assert bad > 50, f"测试前提不成立：旧行为没有明显的黑点（{bad} 像素）"
    assert ok <= bad * 0.05, f"edge_ref 没压住黑点：旧 {bad} → 新 {ok} 像素"


def test_edge_ref_equals_composed_when_passed_the_same_image():
    """``edge_ref`` 只是"换一张图求梯度"，不该改变其它任何环节。

    显式把合成图自己当边缘参考，必须与 ``edge_ref=None`` **逐位相同** ——
    这条守住"这个参数没有偷偷参与配色/抖动"。
    """
    sc = _smooth_scene()
    c, _ = compose_frame(sc, t=0.3, anim=AnimParams(dust_count=0), return_content=True)
    pal = palette_from_frames([c], n_colors=32)
    a, _ = finish_frame(c, sc.tail, palette=pal, edge_ref=None)
    b, _ = finish_frame(c, sc.tail, palette=pal, edge_ref=c)
    assert np.array_equal(a, b), "edge_ref 传同一张图时，结果必须与不传完全一致"


def test_flat_edge_ref_produces_no_darkening():
    """平坦的边缘参考 → 梯度处处为 0 → **一点压暗都不该有**。

    这正面证明 ``edge_ref`` 只通过"梯度"起作用：给它一张平坦的图，
    所有像素都不该被压暗（输出不会比不压暗时更暗）。
    """
    sc = _smooth_scene()
    c, _ = compose_frame(sc, t=0.3, anim=AnimParams(dust_count=0), return_content=True)
    pal = palette_from_frames([c], n_colors=32)

    u_none, _ = finish_frame(c, sc.tail, palette=pal, edge_ref=None)
    flat = np.full_like(c, 0.5)
    u_flat, _ = finish_frame(c, sc.tail, palette=pal, edge_ref=flat)

    def luma(x):
        return x.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    diff = luma(u_flat) - luma(u_none)
    assert float(diff.min()) >= -1e-6, "平坦的边缘参考不该产生任何压暗"
    assert float(diff.max()) > 0.5, "本该被压暗的地方没有差别，测试没有意义"


def test_dust_layer_is_purely_additive():
    """粒子图层必须是**纯加性**的 —— 它自己不该产生任何暗像素。

    暗点是下游（边缘压暗）的锅，不是粒子本身的。这条守住归因边界：
    以后有人再看到黑点，能立刻排除"粒子图层本身"这个嫌疑。
    """
    sc = make_scene(gh=48, gw=64)
    on = compose_frame(sc, t=0.2, anim=AnimParams(dust_count=100))
    off = compose_frame(sc, t=0.2, anim=AnimParams(dust_count=0))
    assert float((on - off).min()) >= -1e-6, "粒子让某些像素变暗了（不应该是纯加性吗）"


def test_dust_particles_stay_near_their_base_position():
    """⚠️ 粒子必须「游荡」而不是「匀速直线横穿」。

    用户报障：「像是有蚊子或者是蚂蚁在**直线爬行**」。
    旧实现用 ``frac(x0 + k·t)``（整数圈数直线位移），每颗粒子以恒定速度
    **横穿整个画面**并在边缘折返 —— 观感就是蚂蚁排队行军。

    现在用 Lissajous 摆动，粒子应当留在基准位置附近。
    实测：离基准最大距离 33.3% 画面宽 → **3.4%**。

    用 ``count=1`` + 变化 ``seed`` 来逐个追踪粒子 —— 因为 ``dust_layer``
    只生成前 ``count`` 颗，``count=1`` 永远是第 0 颗；靠 seed 换粒子最干净。
    """
    from pixelart.animate import dust_layer, hash01

    h, w, n = 72, 120, 32
    worst = 0.0
    for seed in range(5):
        pts = []
        for k in range(n + 1):
            layer = dust_layer((h, w), None, k / n, count=1, seed=seed)
            ys, xs = np.nonzero(layer[..., 0] > 0)
            assert len(ys) == 1, f"count=1 应当恰好产生一个像素，实得 {len(ys)}"
            pts.append((float(xs[0]), float(ys[0])))

        base = np.array([hash01(seed, 0, 1) * w, hash01(seed, 0, 2) * h])
        p = np.array(pts)
        d = np.abs(p - base)
        d[:, 0] = np.minimum(d[:, 0], w - d[:, 0])      # wrap 后的最短距离
        d[:, 1] = np.minimum(d[:, 1], h - d[:, 1])
        worst = max(worst, float(np.linalg.norm(d, axis=1).max()))

    assert worst < w * 0.15, (
        f"粒子离基准最远跑出 {worst:.1f}px（画面宽 {w}）—— 像是横穿而不是游荡")


def test_dust_motion_is_smooth_not_jumpy():
    """粒子不能逐帧乱跳（那会变成闪烁噪点）——每帧位移必须是小步长。"""
    from pixelart.animate import dust_layer

    h, w, n = 72, 120, 24
    pts = []
    for k in range(n + 1):
        layer = dust_layer((h, w), None, k / n, count=1, seed=3)
        ys, xs = np.nonzero(layer[..., 0] > 0)
        pts.append((float(xs[0]), float(ys[0])))
    step = np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)
    assert step.max() <= 6.0, f"单帧最大位移 {step.max():.1f}px —— 太跳了"
    assert step.mean() > 0.05, "粒子几乎不动"


def test_grid_from_aspect_never_exceeds_source():
    """⚠️ 网格不能超过源图尺寸 —— 否则就是在凭空造细节。

    处理**任意图片**（M3 的上传）时才会暴露：参数网格长边默认 480，
    但用户完全可能传一张 60×40 的小图。不限制的话网格会算成 200×132，
    比源图还大 3 倍 —— 降采样那步实际变成**升采样**，
    每个"像素方块"里填的是插值出来的假细节，既不像像素画也不是原图。
    """
    for (w, h) in ((60, 40), (200, 150), (480, 270), (1920, 1080), (4000, 3000)):
        gw, gh = grid_from_aspect(w, h, 480)
        assert gw <= w, f"{w}x{h} 的网格宽 {gw} 超过了源图"
        assert gh <= h, f"{w}x{h} 的网格高 {gh} 超过了源图"

    # 大图不受影响：仍然按长边取值
    assert max(grid_from_aspect(1920, 1080, 480)) == 480
    assert max(grid_from_aspect(4000, 3000, 480)) == 480
    # 小图被压到源图尺寸
    assert grid_from_aspect(60, 40, 480) == (60, 40)
    assert grid_from_aspect(200, 150, 480) == (200, 150)


def test_grid_from_aspect_stays_even_and_keeps_ratio():
    """取偶数（避免半像素）且保持宽高比 —— 两条都不能因为加了上限而破。"""
    for (w, h) in ((60, 40), (200, 150), (777, 333), (1920, 1080), (813, 1348)):
        gw, gh = grid_from_aspect(w, h, 480)
        assert gw % 2 == 0 and gh % 2 == 0, f"{w}x{h} -> {(gw, gh)} 出现奇数边"
        if max(w, h) >= 480:                      # 未被压缩时才要求比例一致
            assert abs((gw / gh) - (w / h)) < 0.03, f"{w}x{h} 比例变了：{gw / gh:.3f}"
