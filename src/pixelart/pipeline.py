"""场景与视频管线 —— 把「时间」串成一条完整的、可循环的流水线。

⭐ 核心架构（arch-01 §8 D5/D6）：**把「时间」当成一个普通参数。**

    t 取一个值   → 静态图
    t 扫过 [0,1) → 无缝循环视频

两条输出**共用同一份代码**，静态图就是视频的第 0 帧。所以这里没有
"静态模式"和"视频模式"两套逻辑，只有 ``t`` 取一个值还是扫一个区间。

───── 为什么要有 ``Scene`` 这个对象 ─────

管线里有两类工作，成本差一个数量级：

    「一次」（贵，与帧数无关）        「每帧」（便宜，× 帧数）
    ① 单目深度推理                   ④ 深度雾
    ② 结构感知降采样 + 色阶 + 锐化     ⑤ 体积光
    ③ 光照中心 / 雾色估计             ⑥ 辉光
                                     ⑦ 尘埃粒子
                                     ⑧ 像素尾巴（色板吸附 + 抖动 + 边缘压暗）

``Scene`` 就是「一次」那部分的产物，缓存起来给所有帧复用。
30fps × 180 帧下，这是唯一能让离线渲染跑得完的结构。

───── 三条时序铁律（在 pipeline 层面落地）─────

1. **全片共用一块色板**：``palette_from_frames`` 抽样求一次，逐帧只做吸附。
   逐帧各求色板 = 整片闪烁。
2. **抖动相位锁屏幕空间**：由 ``pixelate`` 内部保证，这里**绝不把 t 传进去**。
3. **一切随时间变化的量严格以 1 为周期**：由 ``animate`` 保证
   （整数频率正弦 + 三维循环噪声 + 整数圈数粒子）。

自检由 :func:`check_loop` 提供：``frame(0) == frame(N)``。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .analyze import DepthEstimator
from .animate import dust_layer, flicker_gain
from .compose import (
    BLOOM_RADII,
    BLOOM_THRESHOLD,
    auto_fog_color,
    bloom,
    brightest_center,
    depth_equalize,
    depth_fog,
    limit_saturation,
    volumetric_light,
)
from .palette import palette_from_frames, to_perceptual
from .pixelate import PixelTail, pixelate
from .resample import (
    auto_levels,
    fit_native,
    fit_to_aspect,
    structure_aware_downsample,
    tone_map,
    unsharp,
)

__all__ = [
    "AnimParams",
    "Scene",
    "grid_from_aspect",
    "prepare_scene",
    "compose_frame",
    "finish_frame",
    "render_video",
    "render_still",
    "check_loop",
]


# ---------------------------------------------------------------- 动画参数
@dataclass
class AnimParams:
    """随时间变化的**幅度**旋钮。全部为 0 时退化成静态图。

    ⚠️ 这些是**幅度**不是速度：速度由 ``t`` 的扫法（帧数）决定。
    这样同一套参数在 4 秒和 8 秒的片子里观感一致。
    """

    fog_drift: float = 0.30          # 雾的浓淡漂移幅度
    light_flicker: float = 0.16      # 光源闪烁幅度
    parallax: float = 0.0            # 分层视差推拉幅度（0 = 关；见 parallax.py）
    dust_count: int = 200            # 尘埃数量（0 = 关掉）
    dust_bright: float = 0.55        # 尘埃亮度
    dust_twinkle: float = 0.55       # 尘埃闪烁深度
    dust_fade_far: float = 0.65      # 远处尘埃减弱（避免远景浮一层噪点）
    dust_light_boost: float = 1.8    # 光源附近尘埃增亮

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 场景
@dataclass
class Scene:
    """「一次」那部分的产物 —— 所有帧共用。"""

    base: np.ndarray                 # (Gh, Gw, 3) 预处理后的像素网格色（线性）
    far: np.ndarray                  # (Gh, Gw) 深度，0 = 近 1 = 远
    grid: tuple[int, int]            # (Gw, Gh)
    scale: int                       # 整数放大倍率
    light_xy: tuple[float, float]    # 光源归一化位置
    fog_color: np.ndarray            # (3,) 雾色
    tail: PixelTail
    src_size: tuple[int, int] = (0, 0)
    aspect: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def out_size(self) -> tuple[int, int]:
        """最终输出像素尺寸（网格 × 整数倍）。"""
        return (self.grid[0] * self.scale, self.grid[1] * self.scale)


def grid_from_aspect(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """按长边定像素网格，保持原始宽高比，并取偶数（避免半像素）。

    ⚠️ **网格不能超过源图尺寸** —— 否则就是在凭空造细节。

    这是处理**任意图片**（M3 的上传功能）时才会暴露的问题：
    参数默认网格长边 480，但用户完全可能传一张 60×40 的小图。
    不做限制的话网格会算成 200×132，比源图还大 3 倍 ——
    降采样那一步实际变成**升采样**，然后每个"像素方块"里填充的是插值出来的
    假细节，看上去既不像像素画、也不是原图。
    正确行为是把网格压到源图尺寸（60×40），之后照常**整数倍**放大用于显示。

    大图不受影响（480 网格对 1920 宽的源图仍然远小于源图）。
    """
    long_edge = max(2, int(long_edge))
    if w >= h:
        gw, gh = long_edge, round(h * long_edge / w)
    else:
        gw, gh = round(w * long_edge / h), long_edge
    if w > 0 and h > 0:
        gw = min(gw, int(w))
        gh = min(gh, int(h))
    return max(2, gw // 2 * 2), max(2, gh // 2 * 2)


def prepare_scene(
    img: Image.Image,
    grid_long: int = 480,
    work_long: int = 1920,
    aspect: float = 0.0,
    tail: PixelTail | None = None,
    levels: float = 0.9,
    sharpen: float = 0.55,
    detail: float = 0.0,
    sky_on: bool = False,
    sky_mode: str = "night",
    sky_stars: float = 1.0,
    depth_estimator: DepthEstimator | None = None,
) -> Scene:
    """跑完「一次」的两步：深度分析 + 预处理，并定下光照中心与雾色。

    ⚠️ 顺序很重要：**先定网格，再令工作分辨率 = 网格 × 整数倍**。
    否则放大不是整数倍，像素方块会参差不齐，直接毁掉"利"的感觉（约束 12）。
    """
    img = img.convert("RGB")
    tail = tail or PixelTail()
    aspect = float(aspect)

    grid = grid_from_aspect(img.width, img.height, grid_long)
    scale = max(2, round(work_long / grid_long))
    work = (grid[0] * scale, grid[1] * scale)

    ref = (fit_native(img, max(work)) if aspect <= 0 else fit_to_aspect(img, aspect))
    ref = ref.resize(work, Image.Resampling.LANCZOS)

    # ① 深度（贵，一次）
    est = depth_estimator or DepthEstimator()
    far_full = est.predict_far(ref)
    far_grid = depth_equalize(
        np.asarray(
            Image.fromarray(far_full.astype(np.float32), mode="F").resize(grid, Image.Resampling.BOX),
            dtype=np.float32,
        ),
        strength=0.5,
    )

    # ② 预处理（一次）
    base = structure_aware_downsample(ref, grid, tail.var_gain, detail=detail)
    base = auto_levels(base, clip=(1.0, 99.0), strength=levels)
    base = tone_map(base, tail.tone_black, tail.tone_white, tail.tone_scurve)
    base = unsharp(base, amount=sharpen)

    # 原始深度缩到网格（供天空掩膜 / 光源检测的天空排除使用）。
    # ⚠️ 必须用**均衡化前**的深度 —— depth_equalize 会把天空的 0.995 饱和
    #    抹掉（实测均衡化后天空占比 0%，掩膜失效）。
    far_sky = np.asarray(
        Image.fromarray(far_full.astype(np.float32), mode="F").resize(
            (grid[0], grid[1]), Image.Resampling.BOX), dtype=np.float32)

    # ②.5 天空层替换（M5.7，默认关）：双保守门 + 程序化渐变 + 星场。
    #   在场景构建阶段改 **base** → 下游（雾/体积光/色板/量化）与 CPU/GPU
    #   两条路径自动一致；替换掉的脏天空也不再浪费自动色板的槽位。
    #   ⚠️ 放在光照/雾色估计**之前**：新天空干净，雾色随它协调。
    if sky_on:
        from .sky import sky_replace
        base, _sky_info = sky_replace(base, far_sky, mode=sky_mode, stars=sky_stars)

    # ③ 光照中心与雾色：在**合成前**的基础色上估，随 t 不变（避免逐帧漂移）
    # ⭐ 光源检测 v2：排除天空后的最亮区域（masks.detect_light_source）。
    #    ref01 那类画面里星空/天空是全图最亮，旧检测会把光源放到天上；
    #    掩膜排除后光源落到画面实体上。无法排除时回退旧检测。
    from .masks import detect_light_source
    from .masks import sky_mask as _sky_mask
    if _sky_mask(far_sky).any():
        light_xy = detect_light_source(base, far_sky, blur_radius=3.0)
    else:
        light_xy = brightest_center(base, blur_radius=3.0)
    # ⚠️ 这里存**原始**雾色（sat_max=1.0，不压饱和），饱和度上限留给每帧施加。
    #    否则 `fog_sat` 滑杆只能在建场景那一刻生效一次 ——
    #    实测症状：拖动 fog_sat 毫无反应（因为 compose_frame 用的是这里算好的固定值）。
    fog_color = auto_fog_color(base, far_grid, sat_max=1.0)

    return Scene(
        base=base,
        far=far_grid,
        grid=grid,
        scale=scale,
        light_xy=light_xy,
        fog_color=fog_color,
        tail=tail,
        src_size=img.size,
        aspect=aspect,
        stats={
            "depth_q": [float(v) for v in np.percentile(far_grid, [5, 25, 50, 75, 95])],
        },
    )


# ------------------------------------------------------------ 每帧：合成
def compose_frame(
    scene: Scene,
    t: float = 0.0,
    density: float = 0.7,
    power: float = 3.0,
    rays: float = 0.55,
    bloom_strength: float = 0.85,
    anim: AnimParams | None = None,
    rays_color=None,
    rays_sat: float = 0.25,
    fog_color_override=None,
    fog_sat: float = 0.40,
    screen_falloff: float = 1.0,
    light_xy: tuple[float, float] | None = None,
    cone_angle: float = 0.0,
    cone_dir_deg: float = 80.0,
    cone_reach: float = 0.8,
    cone_gain: float = 1.6,
    cone_shaft: float = 0.0,
    cone_x: float = -1.0,
    cone_y: float = -1.0,
    light2_on: bool = False,
    light2_x: float = 0.7,
    light2_y: float = 0.3,
    light2_gain: float = 1.0,
    light2_spread: float = 1.0,
    fog_tint: float = 1.0,
    return_content: bool = False,
):
    """合成一帧（像素网格分辨率的**连续色**，还没量化）。

    这是"每帧"那部分的核心。``t`` 只从这里进入 ——
    **绝不往下传给 ``pixelate``**（抖动相位不能含时间）。

    Args:
        fog_color_override: 手动雾色（覆盖 scene 里自动估的那个）。
            ``None`` = 用自动估计的。
        fog_sat: 雾色的饱和度上限（见 ``compose.auto_fog_color``）。
            ⚠️ 雾覆盖全部远景，过饱和的雾色就是"整幅图像蒙了层滤镜"。
            ``None`` = 不压。**对手动雾色同样生效**（一个规则比两个好记）。
        screen_falloff: 体积光的**屏幕空间**衰减（0 = 全图均匀叠加）。
            ⚠️ 只按深度差衰减时，与光源深度相近的像素会均匀吃到带色的光，
            观感同样像全局滤镜。
        light_xy: 光源位置覆盖（归一化）。``None`` = 用 ``scene.light_xy``。
            ⚠️ 必须支持覆盖，否则界面的手动光源滑杆形同虚设。
        return_content: 为 True 时额外返回 ``(composed, content)``。
            ``content`` 是**不含粒子**的合成图，专门用来做边缘检测 ——
            粒子是大气现象不是内容，见 :func:`finish_frame` 的 ``edge_ref``。
    """
    a = anim or AnimParams()
    gh, gw = scene.grid[1], scene.grid[0]

    # ⭐ 分层视差（M5）：把深度切层做极缓慢推拉，近层动得多、远层几乎不动。
    #    必须发生在**雾/体积光之前** —— 像素搬走后它的深度也跟着搬
    #    （雾按深度算，光轴按深度衰减；拿位移后的 far 才不会"内容动了、
    #    空气没动"）。见 parallax.py 的设计注释。
    #    amp=0（默认）走原数组，一切与旧版逐位相同。
    if getattr(a, "parallax", 0.0) > 0:
        from .parallax import parallax_warp
        base_c, far_c, _ = parallax_warp(scene.base, scene.far, t, a.parallax)
    else:
        base_c, far_c = scene.base, scene.far

    # ⚠️ 雾色的两个旋钮都必须在这里生效：
    #    · 手动雾色（fog_r/g/b）没接进来过 → 拖了完全没反应
    #    · 饱和度上限 fog_sat 也没接进来过 → 同样没反应
    #    原因都是这里硬传 `color=scene.fog_color`，把 TuneParams 的 override 绕过去了。
    #    这是"参数定义了但没接进渲染调用"这一类 bug 的第 3、4 次出现 ——
    #    所以现在的原则是：**渲染入口的每个旋钮都要显式接上来**。
    fc = scene.fog_color if fog_color_override is None else np.asarray(
        fog_color_override, dtype=np.float32)
    if fog_sat is not None:
        fc = limit_saturation(fc, fog_sat)
    fogged = depth_fog(
        base_c, far_c,
        color=fc, density=density, power=power,
        t=t, drift=a.fog_drift, fog_tint=fog_tint,
    )

    # ⚠️ light_xy 允许覆盖：界面上「光源位置」的手动 X/Y 滑杆要靠它生效。
    #    之前这里硬用 scene.light_xy，导致手动滑杆**完全无效且不报错**
    #    （实测 XY=(0.2,0.2) 与 (0.5,0.5) 输出逐位相同）。
    lxy = scene.light_xy if light_xy is None else (float(light_xy[0]), float(light_xy[1]))
    # ⭐ 锥顶点解耦（M5.4）：<0 = 跟随光源（volumetric_light 内部走原路径，逐位不变）
    cxy = None
    if cone_x >= 0 or cone_y >= 0:
        cxy = (float(cone_x) if cone_x >= 0 else float(lxy[0]),
               float(cone_y) if cone_y >= 0 else float(lxy[1]))
    vol = volumetric_light(
        fogged, far_c, lxy,
        strength=rays, air_color=rays_color, air_sat_max=rays_sat,
        t=t, flicker=a.light_flicker, screen_falloff=screen_falloff,
        # ⚠️ 光锥的三个旋钮**必须**在这里显式接上来 ——
        #    "参数定义了但没接进渲染调用"是这个项目的老毛病（第 3、4、5 次），
        #    这次是 test_cone.py 立刻报 TypeError 才发现的。
        cone_angle=cone_angle, cone_dir_deg=cone_dir_deg,
        cone_reach=cone_reach, cone_gain=cone_gain, shaft=cone_shaft,
        fog_tint=fog_tint, cone_xy=cxy,
    )
    lit = np.clip(fogged + vol, 0.0, 1.0)
    # ── 副光源（M5.6）：线性加法（物理正确：辐照度相加）──
    # ⚠️ 合成顺序必须与 GPU 链一字对应：GPU 是 volumetric2 pass 读主链输出、
    #    出口 clip(输入 + vol2) → CPU 就是 clip(clip(fogged+vol1) + vol2)。
    #    散射色在光源2邻域自动采样（同 light1 的机制）；v1 无锥、同闪烁。
    if light2_on:
        lxy2 = (float(light2_x), float(light2_y))
        vol2 = volumetric_light(
            fogged, far_c, lxy2,
            # ⚠️ 副光源散射色**永远自动**（在它自己的邻域采样）——
            #    不继承主光的手动颜色（ rays_color 是主光的语义）。
            strength=light2_gain, air_color=None, air_sat_max=rays_sat,
            t=t, flicker=a.light_flicker, screen_falloff=light2_spread,
            occlude_gain=5.0, falloff_gain=2.5,
            fog_tint=fog_tint,
        )
        lit = np.clip(lit + vol2, 0.0, 1.0)

    # ⚠️ 边缘检测的参考图：**不含粒子**。粒子会被误判成内容边界。
    content = lit

    if a.dust_count > 0:
        # 尘埃：加在辉光之前，这样它自己也会被辉光晕开一点
        dust = dust_layer(
            (gh, gw), far_c, t=t,
            count=a.dust_count, seed=11,
            # ⚠️ 必须用 lxy（带手动覆盖的那个），不是 scene.light_xy。
            #    原来这里硬传 scene.light_xy，而 volumetric_light 用的是 lxy ——
            #    结果是"手动光源位置"只移动了光柱、移动不了尘埃增亮区，
            #    两个本该一致的东西用两个不同的光源位置。
            #    这与 light_xy / fog_sat 那几次是同一类问题的第 5 次出现：
            #    **同一个概念在两处各取一次，迟早会分叉。**
            light_xy=lxy,
            light_boost=a.dust_light_boost,
            twinkle=a.dust_twinkle,
            fade_far=a.dust_fade_far,
        )
        lit = np.clip(lit + dust * a.dust_bright, 0.0, 1.0)

    # bloom 只做小半径的"发亮"，大范围的光晕交给体积光
    out = bloom(lit, threshold=BLOOM_THRESHOLD,
                strength=bloom_strength * 0.5, radii=BLOOM_RADII)
    if return_content:
        return out, content
    return out


def finish_frame(
    composed: np.ndarray,
    tail: PixelTail,
    palette: np.ndarray | None = None,
    edge_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """像素尾巴：量化 → 抖动 → 边缘压暗。**必须是最后一步。**

    ⚠️ 这里**没有 t 参数，也不要加**。抖动相位一旦含时间就是整屏闪烁。

    ⚠️ ``edge_ref`` 请传**不含粒子**的合成图。边缘压暗是**内容风格化**步骤
    （给物体轮廓描边），把单像素的粒子误判成强边界会把粒子周围压暗成黑点 ——
    实测单帧 3292 个像素被压暗、最深 82/255 色阶（用户报障"黑点在移动"）。
    """
    return pixelate(composed, tail, palette=palette, edge_ref=edge_ref)


def upscale(u8: np.ndarray, scale: int) -> Image.Image:
    """整数倍最近邻放大（像素画必须整数倍）。"""
    im = Image.fromarray(np.ascontiguousarray(u8))
    return im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)


# ---------------------------------------------------------------- 视频
def render_video(
    scene: Scene,
    n_frames: int = 120,
    anim: AnimParams | None = None,
    density: float = 0.7,
    power: float = 3.0,
    rays: float = 0.55,
    bloom_strength: float = 0.85,
    rays_color=None,
    rays_sat: float = 0.25,
    include_last: bool = True,
    on_progress=None,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """渲染整段循环视频，返回 ``(帧列表 uint8, 全局色板, 统计)``。

    ⭐ **全片共用一块色板**（时序铁律 1）：
    先把所有帧合成到连续色空间，再 ``palette_from_frames`` 抽样求一次色板，
    最后逐帧只做吸附。逐帧各求色板 = 整片闪烁。

    ``t`` 的取法：第 i 帧取 ``t = i / n_frames``。
    ``include_last`` 为 True 时额外渲一帧 ``t = 1.0`` —— 用于自检
    ``frame(N) == frame(0)``（循环接缝）。它不该被写进视频循环体，
    但要用来验算。

    Args:
        n_frames: 循环体帧数（不含自检帧）。
        on_progress: 可选回调 ``fn(i, n)``，用于命令行进度。
    """
    a = anim or AnimParams()
    if n_frames < 2:
        raise ValueError("n_frames 至少为 2，否则谈不上循环")

    total = n_frames + (1 if include_last else 0)
    composed: list[np.ndarray] = []
    contents: list[np.ndarray] = []
    for i in range(total):
        t = i / n_frames
        c, ct = compose_frame(
            scene, t=t, density=density, power=power, rays=rays,
            bloom_strength=bloom_strength, anim=a,
            rays_color=rays_color, rays_sat=rays_sat, return_content=True,
        )
        composed.append(c)
        contents.append(ct)
        if on_progress:
            on_progress(i + 1, total)

    # ⭐ 全局色板：抽样求一次，全片共用
    palette = palette_from_frames(composed[:n_frames], n_colors=scene.tail.n_colors,
                                 max_samples=24)

    # ⚠️ edge_ref 传"不含粒子"的合成图：粒子是大气不是内容，
    #    否则每颗粒子会被边缘压暗搞成一个小黑点。
    frames = [finish_frame(c, scene.tail, palette=palette, edge_ref=ct)[0]
              for c, ct in zip(composed, contents)]

    body = frames[:n_frames]
    loop_err = _loop_error(frames[n_frames], body[0]) if include_last else None
    stats = {
        "n_frames": n_frames,
        "palette_size": int(len(palette)),
        "loop_max_diff": loop_err,
        "loop_ok": (loop_err == 0) if loop_err is not None else None,
    }
    return body, palette, stats


def render_still(
    scene: Scene,
    t: float = 0.0,
    palette: np.ndarray | None = None,
    **kw,
) -> tuple[np.ndarray, np.ndarray]:
    """渲染**一张**静态图 —— 就是 ``t`` 取固定值的视频帧。

    ⚠️ 静态图的色板默认是单帧求的，所以它和同名视频的色板**未必逐位一致**。
    想让静态图与视频第 0 帧完全一致，请传 ``palette``（取视频那块）。
    """
    kw.pop("return_content", None)
    composed, content = compose_frame(scene, t=t, return_content=True, **kw)
    return finish_frame(composed, scene.tail, palette=palette, edge_ref=content)


def _loop_error(a: np.ndarray | None, b: np.ndarray) -> int | None:
    """两帧之间的最大通道差（uint8）。0 = 逐位相同 = 循环无缝。"""
    if a is None:
        return None
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


def check_loop(frames: list[np.ndarray], n_frames: int | None = None) -> dict:
    """循环自检：**``frame(0)`` 必须与 ``frame(N)`` 逐位相同**，否则循环有接缝。

    做法是从帧序列里再渲一次首帧的 ``t``（即 ``t=1.0``）并比对。
    这个函数不重渲染，只检查已渲好的序列：
    ``frames`` 若含自检帧（长度 = n_frames + 1），比对末帧与首帧。

    Returns:
        ``{"ok": bool, "max_diff": int, "where": (y, x) | None}``
    """
    if n_frames is None:
        n_frames = len(frames) - 1
    if len(frames) < 2:
        raise ValueError("至少需要两帧才能做循环自检")
    a, b = frames[0], frames[-1]
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mx = int(d.max())
    where = None
    if mx > 0:
        idx = np.unravel_index(int(d.sum(axis=2).argmax()), d.shape[:2])
        where = (int(idx[0]), int(idx[1]))
    return {"ok": mx == 0, "max_diff": mx, "where": where}


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """两帧平均绝对差（float）。用来确认"真的在动"，以及量的稳定性。"""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
