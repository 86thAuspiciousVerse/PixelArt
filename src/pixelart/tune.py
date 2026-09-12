"""调参参数集与**快速预览**渲染 —— M3 调参界面的内核。

把界面做成"薄薄一层 HTTP 胶水"是这里的设计目标：参数映射、预览渲染、
自检统计全都放在这个模块里，这样它们**能被单元测试覆盖**，
而 ``tools/m3_server.py`` 只负责收发请求。

───── 为什么需要「快速预览」─────

完整渲染一帧的成本（480 网格）约 70 ms 合成 + 80 ms 像素尾巴，
180 帧就是 40 多秒。拖滑杆时不可能等这个。所以预览走两条降级路径：

1. **降网格**：网格长边从 480 降到 240（像素数 1/4，各阶段都快约 4 倍）。
   注意**不能改放大倍率** —— 仍然保持整数倍，否则方块会参差。
2. **降帧率，但保持时长**：这条最容易被做错。

   ⚠️ 预览绝不能靠"少渲几帧"来加速。动画速度由 ``t`` 的扫法决定，
   而 ``t`` 是**归一化到 [0,1) 的循环相位**。如果成片是 6 秒 180 帧
   （每帧 ``t`` 走 1/180），预览若只渲 12 帧并让 ``t`` 同样走 1/12，
   那雾漂移与粒子速度会**快 15 倍** —— 参数就不是"所见即所得"了。

   正确做法：**保持总时长不变，只降 fps**。想预览 6 秒的片子，
   就渲 6 秒 × 10fps = 60 帧。``t`` 在同样的墙钟时间里走完一个周期，
   速度与成片一致，只是更卡顿。代价是时间分辨率下降，但速度是对的。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
from PIL import Image

from .animate import dust_layer  # noqa: F401  (对外暴露，方便上层直接用)
from .palette import from_perceptual, palette_from_frames
from .pipeline import (
    AnimParams,
    Scene,
    compose_frame,
    finish_frame,
    grid_from_aspect,
    prepare_scene,
)
from .pixelate import PixelTail
from .resample import fit_native, fit_to_aspect

__all__ = [
    "TuneParams",
    "aspect_value",
    "build_scene",
    "display_scale",
    "render_preview",
    "preview_stats",
    "render_still",
    "render_sweep",
    "AUTO",
]

#: 颜色字段用这个值表示"自动估计"。故意取负数，和 [0,1] 的色彩值区分开。
AUTO = -1.0

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def aspect_value(a: str | float) -> float:
    """``'native'`` / ``'16:9'`` / ``2.16`` → 浮点宽高比；0.0 表示保持原比例。"""
    if isinstance(a, (int, float)):
        return float(a)
    s = str(a).strip().lower()
    if s in ("native", "keep", "none", ""):
        return 0.0
    if ":" in s:
        x, y = s.split(":", 1)
        return float(x) / float(y)
    return float(s)


@dataclass
class TuneParams:
    """调参界面的**全部**旋钮 —— 界面上的每个滑杆都对应这里的一个字段。

    ⚠️ 分三档看待这些参数（这决定了调参时的顺序）：

    - **必须逐图调**：``density``（雾）、``rays``（光强）、``rays_*``（光源位置/散射色）、
      ``fog_*``（雾色）。自动估计只能当默认值 —— 单张图里没有光照信息，
      ``brightest_center`` 会落在错误的地方（实测会落在草丛和地面反光上）。
    - **动画幅度**：``fog_drift`` / ``flicker`` / ``dust_*``。这部分**没有客观指标**
      可优化，好不好看只能靠眼睛。
    - **结构性设置**：``grid_long`` / ``colors`` / ``aspect``。改动它们要重建场景
      （深度与预处理都变了），所以界面上应当触发一次明确的"重建"。

    默认值与 ``PixelTail`` / 生产路径保持一致，避免"界面调好了、命令行跑出来不一样"。
    """

    # ── 网格与质量（改动需要重建场景）──
    grid_long: int = 240          # 预览网格长边；全量渲染时换成 480
    work_long: int = 960          # 工作分辨率长边（必须是 grid_long 的整数倍）
    aspect: str = "native"        # 'native' 或 '16:9' 等

    # ── 预处理（改动需要重建场景）──
    levels: float = 0.9           # auto_levels 强度
    sharpen: float = 0.55         # unsharp 强度
    var_gain: float = 45.0        # 结构感知降采样的边缘保留权重

    # ── 合成：雾 ──
    density: float = 0.7          # ⚠️ 可用区间很窄，实测 0.5~0.8 可用、1.4 开始毁
    power: float = 3.0            # 距离幂次；<2 会让中景发灰
    fog_r: float = AUTO           # 雾色；三者全为 AUTO 时用 scene 自动估计
    fog_g: float = AUTO
    fog_b: float = AUTO
    fog_sat: float = 0.40         # ⚠️ 自动雾色的饱和度上限（0 = 不压）
    fog_tint: float = 1.0         # ⭐ 雾/散射的上色强度（0 = 雾变中性灰、保色相；1 = 现状）

    # ── 合成：体积光 ──
    rays: float = 0.55            # 强度
    rays_auto_center: bool = True # 光源位置是否自动
    rays_x: float = 0.5           # 手动光源位置（归一化）
    rays_y: float = 0.5
    rays_sat: float = 0.25        # ⚠️ 散射色饱和度上限（空气散射近乎无色）
    rays_spread: float = 1.0      # ⚠️ 体积光的**屏幕空间**衰减（0 = 全图均匀叠加）
    rays_r: float = AUTO          # 散射色；全 AUTO 时走 auto_scatter_color
    rays_g: float = AUTO
    rays_b: float = AUTO
    bloom: float = 0.85           # 辉光强度

    # ── 合成：光锥（把径向光晕塑成形）──
    rays_cone: float = 0.0        # 锥张角（弧度，0 = 关闭 → 与旧版逐位相同）
    rays_dir: float = 80.0        # 锥轴方向（**度**；屏幕 y 向下，90 = 向下）
    rays_reach: float = 0.8       # 沿轴的长度：1 = 几乎不额外衰减
    rays_cone_gain: float = 1.6   # 锥内增益（只塑形用；真要"看得见"靠 rays_shaft）
    rays_shaft: float = 0.0       # ⭐ 光柱强度（独立加性层；0 = 关 → 与旧版逐位相同）
    cone_x: float = -1.0          # ⭐ 锥顶点X（M5.4）：<0 = 跟随光源；≥0 独立放置（可出画）
    cone_y: float = -1.0          # ⭐ 锥顶点Y：同上。穿窗光束（光源在画外）靠它
    light2_on: bool = False       # ⭐ 副光源（M5.6）：lain 的"屏幕 + 窗户"双光源
    light2_x: float = 0.7         # 副光源位置（归一化，手动；v1 无自动检测）
    light2_y: float = 0.3
    light2_gain: float = 1.0      # 副光源强度（体积光注入倍率）
    light2_spread: float = 1.0    # 副光源影响范围收敛（屏幕空间衰减指数）
    quantize_mode: str = "palette"  # ⭐ 取色模式（M5.10）：palette=色板吸附（像素画）
                                  #    continuous=连续色（高保真：跳过吸附，抖动/边缘照常）
    detail: float = 0.0           # ⭐ 细节预算（M5.8，0~1）：低网格下忙碌格子偏向
                                  #    "格内代表性细节像素"（救细线/五官）；0 = 原行为
    sky_on: bool = False          # ⭐ 天空层替换（M5.7）：双保守门，室内图零影响
    sky_mode: str = "night"       # 天空预设：night / dusk / day
    sky_stars: float = 1.0        # 星密度倍率 0~2（day 模式无星）

    # ── 像素尾巴 ──
    colors: int = 32
    dither: float = 0.10
    edge_strength: float = 0.35   # 边缘压暗强度
    edge_gain: float = 2.0        # ⚠️ 标定为 2.0；7.0 会让密集画面漂移 −13%

    # ── 动画幅度（必须为 0 才是真正静态）──
    fog_drift: float = 0.30
    flicker: float = 0.16
    parallax: float = 0.0         # 分层视差推拉幅度（0 = 关；GPU 暂不支持，>0 走服务端）
    dust_count: int = 200
    dust_bright: float = 0.55
    dust_twinkle: float = 0.55
    dust_fade_far: float = 0.65
    dust_light_boost: float = 1.8

    # ── 输出 ──
    seconds: float = 6.0
    fps: int = 30

    # ------------------------------------------------------------ 派生
    def to_tail(self) -> PixelTail:
        """转成像素尾巴的参数集。"""
        return PixelTail(
            var_gain=self.var_gain,
            n_colors=int(self.colors),
            quantize_mode=self.quantize_mode,
            dither=self.dither,
            edge_gain=self.edge_gain,
            edge_strength=self.edge_strength,
        )

    def to_anim(self) -> AnimParams:
        """转成动画幅度参数集。"""
        return AnimParams(
            fog_drift=self.fog_drift,
            light_flicker=self.flicker,
            parallax=self.parallax,
            dust_count=int(self.dust_count),
            dust_bright=self.dust_bright,
            dust_twinkle=self.dust_twinkle,
            dust_fade_far=self.dust_fade_far,
            dust_light_boost=self.dust_light_boost,
        )

    def fog_color(self) -> tuple[float, float, float] | None:
        """显式雾色；三项全为 AUTO 时返回 ``None``（交给自动估计）。"""
        if self.fog_r < 0 or self.fog_g < 0 or self.fog_b < 0:
            return None
        return (float(self.fog_r), float(self.fog_g), float(self.fog_b))

    def rays_color(self) -> tuple[float, float, float] | None:
        """显式散射色；三项全为 AUTO 时返回 ``None``（交给 auto_scatter_color）。"""
        if self.rays_r < 0 or self.rays_g < 0 or self.rays_b < 0:
            return None
        return (float(self.rays_r), float(self.rays_g), float(self.rays_b))

    def light_xy(self, scene: Scene) -> tuple[float, float]:
        """光源位置：自动（用 scene 估计的）或手动。"""
        if self.rays_auto_center:
            return scene.light_xy
        return (float(self.rays_x), float(self.rays_y))

    def compose_kwargs(self, scene: Scene) -> dict:
        """组装 ``compose_frame`` 的关键字参数。

        ⚠️ 光源位置与散射色都依赖 **scene**（自动估计的结果存在 scene 里），
        所以这个函数必须拿到 scene —— 不能只靠 params 自己。
        """
        return dict(
            density=self.density,
            power=self.power,
            rays=self.rays,
            bloom_strength=self.bloom,
            anim=self.to_anim(),
            rays_color=self.rays_color(),
            rays_sat=self.rays_sat,
            fog_sat=self.fog_sat,
            fog_tint=self.fog_tint,
            fog_color_override=self.fog_color(),
            screen_falloff=self.rays_spread,
            cone_angle=self.rays_cone,
            cone_dir_deg=self.rays_dir,
            cone_reach=self.rays_reach,
            cone_gain=self.rays_cone_gain,
            cone_shaft=self.rays_shaft,
            cone_x=self.cone_x,
            cone_y=self.cone_y,
            light2_on=self.light2_on,
            light2_x=self.light2_x,
            light2_y=self.light2_y,
            light2_gain=self.light2_gain,
            light2_spread=self.light2_spread,
            light_xy=self.light_xy(scene),
        )

    def to_query(self) -> str:
        """序列化成 URL 查询串（供界面拼 ``<img src>``）。"""
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                v = int(v)
            parts.append(f"{f.name}={v}")
        return "&".join(parts)

    @classmethod
    def from_query(cls, qs: dict) -> "TuneParams":
        """从查询串字典还原。缺字段用默认值，非法值忽略（不抛，界面更好用）。"""
        out = cls()
        for f in fields(cls):
            if f.name not in qs:
                continue
            raw = qs[f.name]
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            ftype = f.type
            try:
                if ftype is bool or ftype == "bool":
                    setattr(out, f.name, str(raw).lower() in ("1", "true", "yes", "on"))
                elif ftype is int or ftype == "int":
                    setattr(out, f.name, int(round(float(raw))))
                elif ftype is str or ftype == "str":
                    setattr(out, f.name, str(raw))
                else:
                    setattr(out, f.name, float(raw))
            except (TypeError, ValueError):
                continue
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------ 预览
    def preview_frames(self, scene: Scene, preview_fps: int = 10) -> int:
        """预览该渲多少帧 —— **保持时长不变，只降 fps**（见模块文档）。"""
        return max(2, int(round(self.seconds * max(1, preview_fps))))


def display_scale(scene: Scene, target: int = 720) -> int:
    """给预览选一个**整数**放大倍数，让长边接近 ``target``。

    ⚠️ 必须是整数倍：非整数倍会让像素方块参差不齐，预览就失去了参考价值。
    """
    long_edge = max(scene.grid)
    return max(1, int(round(target / max(long_edge, 1))))


def build_scene(img: Image.Image, params: TuneParams, depth_estimator=None) -> Scene:
    """按参数里的网格/预处理设置构建场景（贵：含深度推理，应当缓存）。"""
    return prepare_scene(
        img,
        grid_long=int(params.grid_long),
        work_long=int(params.work_long),
        aspect=aspect_value(params.aspect),
        tail=params.to_tail(),
        levels=params.levels,
        sharpen=params.sharpen,
        detail=params.detail,
        sky_on=params.sky_on,
        sky_mode=params.sky_mode,
        sky_stars=params.sky_stars,
        depth_estimator=depth_estimator,
    )


def render_still(
    scene: Scene,
    params: TuneParams,
    t: float = 0.0,
    palette: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """渲一帧（供拖动滑杆时的即时反馈）。

    ⚠️ 像素尾巴一律用 ``params.to_tail()``，**不要用 ``scene.tail``**。

    ``scene.tail`` 只是**建场景那一刻的快照**。色板颜色数、抖动强度、
    边缘压暗这些是**逐帧可调**的，不需要重建场景 —— 但如果渲染时读
    ``scene.tail``，改滑杆就悄悄不生效了（``pixelate`` 会从 tail 里读
    ``n_colors``，于是"颜色数"滑杆完全没反应）。
    实测症状：改了颜色数但画面色阶数不变，界面上还以为是自己看错了。
    """
    composed, content = compose_frame(
        scene, t=t, return_content=True, **params.compose_kwargs(scene))
    return finish_frame(composed, params.to_tail(), palette=palette, edge_ref=content)


def render_preview(
    scene: Scene,
    params: TuneParams,
    preview_fps: int = 10,
    include_last: bool = True,
    palette: np.ndarray | None = None,
    on_progress=None,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """渲染预览循环，返回 ``(帧, 全片共用色板, 统计)``。

    ⚠️ 与正式渲染一样**全片共用一块色板**（时序铁律 1）——
    预览如果逐帧各求色板，会看到根本不存在的闪烁，把调参带偏。

    ``include_last=True`` 会额外渲 ``t=1.0`` 那一帧用于循环自检；
    它**不属于循环体**，只是拿来验算 ``frame(0) == frame(N)``。
    """
    n = params.preview_frames(scene, preview_fps)
    total = n + (1 if include_last else 0)
    kw = params.compose_kwargs(scene)

    composed: list[np.ndarray] = []
    contents: list[np.ndarray] = []
    for i in range(total):
        c, ct = compose_frame(scene, t=i / n, return_content=True, **kw)
        composed.append(c)
        contents.append(ct)
        if on_progress:
            on_progress(i + 1, total)

    if palette is None:
        palette = palette_from_frames(composed[:n], n_colors=int(params.colors),
                                     max_samples=min(8, n))

    # ⚠️ 用 params.to_tail()（逐帧可调），不是 scene.tail（建场景时的快照）
    tail = params.to_tail()
    frames = [finish_frame(c, tail, palette=palette, edge_ref=ct)[0]
              for c, ct in zip(composed, contents)]
    stats = preview_stats(frames, palette, n, params, scene)
    return frames[:n], palette, stats


def preview_stats(
    frames: list[np.ndarray],
    palette: np.ndarray,
    n_frames: int,
    params: TuneParams,
    scene: Scene,
) -> dict:
    """自检统计 —— 界面上侧栏显示的就是这些数字。

    包含三类（对应三条最容易出错的地方）：

    1. **循环无缝**：``frame(0)`` 与 ``frame(N)`` 的最大通道差，要求 **0**。
    2. **真的在动 / 动得够不够**：相邻帧平均差 + **逐像素整段极差**分档。
       ⚠️ 只看相邻帧差会骗人（实测 0.26/255 时我一度以为"根本没动"），
       必须看整段极差，而且**两头都要看**：太小=没动，全画面均匀=廉价。
    3. **颜色全部来自共用色板**：越界数量必须为 0。

    另外报告光源位置/雾色/网格等元信息，方便判断自动估计是否落错了地方。
    """
    stack = np.stack(frames).astype(np.int16)
    rng_px = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2)

    diffs = [float(np.abs(frames[i].astype(np.float32) - frames[(i + 1) % n_frames].astype(np.float32)).mean())
             for i in range(n_frames)]
    half = float(np.abs(frames[0].astype(np.float32) - frames[n_frames // 2].astype(np.float32)).mean())

    pal_u8 = {tuple(int(v) for v in row) for row in
              np.clip(from_perceptual(palette) * 255.0, 0, 255).astype(np.uint8)}
    seen = set()
    for f in frames:
        seen |= set(map(tuple, f.reshape(-1, 3).tolist()))

    loop_diff = None
    if len(frames) > n_frames:
        loop_diff = int(np.abs(frames[n_frames].astype(np.int16) - frames[0].astype(np.int16)).max())

    moving = {
        "static": float((rng_px == 0).mean()),
        "faint": float(((rng_px >= 1) & (rng_px < 5)).mean()),
        "visible": float(((rng_px >= 5) & (rng_px < 11)).mean()),
        "strong": float((rng_px >= 11).mean()),
    }

    fc = params.fog_color()
    fog_used = np.asarray(fc if fc is not None else scene.fog_color, dtype=np.float32)
    return {
        "grid": [int(scene.grid[0]), int(scene.grid[1])],
        "out_size": [int(scene.out_size[0]), int(scene.out_size[1])],
        "scale": int(scene.scale),
        "src_size": [int(scene.src_size[0]), int(scene.src_size[1])],
        "n_frames": int(n_frames),
        "palette_size": int(len(palette)),
        "loop_diff": loop_diff,
        "loop_ok": (loop_diff == 0) if loop_diff is not None else None,
        "motion": {
            "adj_min": min(diffs),
            "adj_mean": float(np.mean(diffs)),
            "adj_max": max(diffs),
            "half_period": half,
        },
        "amplitude": {
            "max": int(rng_px.max()),
            "mean": float(rng_px.mean()),
            "median": int(np.median(rng_px)),
            **moving,
        },
        "colors_outside": len(seen - pal_u8),
        "colors_seen": len(seen),
        "light_xy": [float(scene.light_xy[0]), float(scene.light_xy[1])],
        "light_xy_used": [float(v) for v in params.light_xy(scene)],
        "fog_color": [float(v) for v in fog_used],
        "fog_color_auto": fc is None,
        # ⚠️ 露出来是为了让界面能把"雾色过饱和"直接标出来 ——
        # 它是"整幅图像蒙了层滤镜"的根源（见 compose.auto_fog_color）
        "fog_sat_actual": float((fog_used.max() - fog_used.min()) /
                                max(float(fog_used.max()), 1e-6)),
        "rays_color_used": [float(v) for v in _rays_used(scene, params)],
        "rays_color_auto": params.rays_color() is None,
        "depth_q": [float(v) for v in np.percentile(scene.far, [5, 25, 50, 75, 95])],
    }


def _rays_used(scene: Scene, params: "TuneParams") -> np.ndarray:
    """实际用到的散射色（供界面展示与"是否过饱和"判断）。

    ⚠️ 采样输入必须与产线一致：``volumetric_light`` 是从**雾后**的图上取样，
    不是从 ``scene.base``。第一版用了 base，结果统计出 [1.0, 1.0, 1.0]（纯白）——
    因为 base 里光源位置是过曝的白，而产线看到的是雾压过之后的颜色。
    """
    rc = params.rays_color()
    if rc is not None:
        return np.asarray(rc, dtype=np.float32)
    from .compose import auto_scatter_color, depth_fog
    gw, gh = scene.grid
    lx, ly = params.light_xy(scene)
    px, py = int(lx * (gw - 1)), int(ly * (gh - 1))
    fogged = depth_fog(scene.base, scene.far, color=scene.fog_color,
                       density=params.density, power=params.power,
                       fog_sat=params.fog_sat)
    return auto_scatter_color(fogged, px, py, radius=3, sat_max=params.rays_sat)


def render_sweep(
    scene: Scene,
    params: TuneParams,
    key: str,
    values: list[float],
    t: float = 0.0,
    palette: np.ndarray | None = None,
) -> list[tuple[float, np.ndarray]]:
    """参数扫描：同一个参数取多个值，各渲一帧。

    这是 ``tools/m1_preview.py --sweep`` 的固化版 —— 项目文档里反复强调
    "真正的成本在参数调优"，而调优的前提是**能一次看多个取值**。
    """
    import dataclasses

    out: list[tuple[float, np.ndarray]] = []
    for v in values:
        p = dataclasses.replace(params, **{key: v})
        u8, _ = render_still(scene, p, t=t, palette=palette)
        out.append((float(v), u8))
    return out


def prepare_variants(
    img: Image.Image,
    grid_long: int,
    work_long: int,
    aspect: str,
    depth_estimator=None,
) -> list[str]:
    """占位：预留"同一张图多组网格对比"的能力（当前未接入界面）。"""
    return []
