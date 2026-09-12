"""调试图（前端「调试图」面板的后端内核，由 ``m3_server`` 的 /api/debug 出 PNG）。

每种图回答一个**明确的调试问题** —— 不是美术输出，配色刻意与主渲染不同
（蓝调深度 / 灰调亮度 / 琥珀调体积），防止和成品混淆：

==========  =============================================================
depth       深度图（暗=近、蓝=远）。雾 / 体积光 / 视差 / 阴影的共同根 ——
            深度估计的灾难（lain 线缆糊成一层、天空误判成近景）在这里一眼可见。
luma        模糊亮度 + 90/95/99 分位**等亮线**。光源检测的输入可视化：
            亮核在哪、集不集中、绿十字为什么落在那里。
vol         体积光**形状**（按最大值归一化增强）。调光锥张角/方向/光柱时
            看的是几何，不该被量化与色调映射压掉的两个数量级能量骗到。
==========  =============================================================

所有图都画上**渲染真正使用的**光源十字（绿）—— 即
``TuneParams.light_xy(scene)`` 的解析结果（自动=场景检测，手动=滑杆），
与 GPU/CPU 渲染严格同源。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _finish(u8: np.ndarray, lx: int, ly: int,
            ax: int | None = None, ay: int | None = None) -> Image.Image:
    """转 PIL 并画十字：绿 = 渲染用的光源；橙 = 解耦时的锥顶点（若不同）。"""
    img = Image.fromarray(np.ascontiguousarray(u8))
    dr = ImageDraw.Draw(img)
    dr.line([(lx - 7, ly), (lx + 7, ly)], fill=(60, 255, 60), width=2)
    dr.line([(lx, ly - 7), (lx, ly + 7)], fill=(60, 255, 60), width=2)
    if ax is not None and ay is not None and (ax, ay) != (lx, ly):
        dr.line([(ax - 9, ay), (ax + 9, ay)], fill=(250, 160, 40), width=2)
        dr.line([(ax, ay - 9), (ax, ay + 9)], fill=(250, 160, 40), width=2)
    return img


def depth_view(sc, lx: int, ly: int) -> Image.Image:
    """深度图：暗=近、蓝=远（与 mask_probe 同一配色，看图不用换脑子）。"""
    x = np.clip(sc.far, 0.0, 1.0)
    u8 = (np.stack([x * 0.2, x * 0.6, x], axis=-1) * 255).astype(np.uint8)
    return _finish(u8, lx, ly)


def luma_view(sc, lx: int, ly: int) -> Image.Image:
    """模糊亮度 + 90/95/99 分位等亮线（蓝/橙/红），光源检测的输入可视化。"""
    from .compose import blur

    base = np.clip(sc.base, 0.0, 1.0)
    lum = blur((base @ _LUMA)[..., None], 3.0)[..., 0]
    p90, p95, p99 = np.percentile(lum, [90, 95, 99])
    u8 = (np.stack([lum, lum, lum], axis=-1) * 255).astype(np.uint8)
    # ±0.004 的窄带 = 等亮线（阈值型参数在这条线上的进退一眼可见）
    u8[np.abs(lum - p90) < 0.004] = (40, 120, 250)
    u8[np.abs(lum - p95) < 0.004] = (250, 160, 40)
    u8[np.abs(lum - p99) < 0.004] = (250, 60, 60)
    return _finish(u8, lx, ly)


def vol_view(sc, *, density: float, power: float, fog_tint: float,
             screen_falloff: float, cone_angle: float, cone_dir_deg: float,
             cone_reach: float, shaft: float, lx: int, ly: int,
             cone_x: float = -1.0, cone_y: float = -1.0) -> Image.Image:
    """体积光**单独**可视化：strength=1 + 按最大值归一 —— 看形状不看强度。

    管线与 compose_frame 一致：先雾后体积（t=0、flicker=0 —— 静态图路径，
    与「静态图就是视频第 0 帧」的约定一致）。
    """
    from .compose import depth_fog, volumetric_light

    base = np.clip(sc.base, 0.0, 1.0)
    fogged = depth_fog(base, sc.far, color=None, density=density,
                       power=power, t=0.0, drift=0.0, fog_tint=fog_tint)
    h, w = sc.far.shape
    cxy = None
    if cone_x >= 0 or cone_y >= 0:
        cxy = (cone_x if cone_x >= 0 else lx / max(w - 1, 1),
               cone_y if cone_y >= 0 else ly / max(h - 1, 1))
    vol = volumetric_light(
        fogged, sc.far,
        (lx / max(w - 1, 1), ly / max(h - 1, 1)),
        samples=28, span=0.85, decay=0.965, strength=1.0,
        occlude_gain=5.0, falloff_gain=2.5, screen_falloff=screen_falloff,
        air_sat_max=0.25, t=0.0, flicker=0.0, cone_angle=cone_angle,
        cone_dir_deg=cone_dir_deg, cone_reach=cone_reach, shaft=shaft,
        fog_tint=fog_tint, cone_xy=cxy)
    mag = np.clip(vol @ _LUMA, 0.0, 1.0)
    # ⚠️ 定标不能用 max：光源自身那个像素必然自曝光（亮度≈1），max 被它绑架。
    # ⚠️ 也不能用全图高分位：锥开时非零支撑可能 < 0.1% 像素，全图 99.9 分位
    #    落在零区 → 除零全黑（两个坑都实测过）。用**非零区**的分位定标 +
    #    gamma 提亮 —— 诊断图要形状可见，不要物理保真。
    nz = mag[mag > 1e-5]
    scale = float(np.percentile(nz, 90)) if nz.size else 0.0
    m = np.clip(mag / max(scale, 1e-6), 0.0, 1.0) ** (1.0 / 1.8)
    u8 = (np.stack([m * 255.0, m * 185.0, m * 80.0], axis=-1)).astype(np.uint8)
    if cxy is not None:
        apx = int(np.clip(cxy[0], 0, 1) * (w - 1))
        apy = int(np.clip(cxy[1], 0, 1) * (h - 1))
        return _finish(u8, lx, ly, apx, apy)
    return _finish(u8, lx, ly)
