"""像素尾巴（Pixel Tail）—— 全部参数的唯一载体与执行入口。

顺序**不可颠倒**（见 docs/arch-01-design.md §8 D4）::

    影调映射 → 调色板吸附 → 有序抖动 → 边缘压暗 → 整数放大

为什么量化必须在最后：Bloom / 雾 / 辉光都需要连续色调。
先量化再合成，等于把画面锁死在一个小色板上再去做渐变 —— 那就是"廉价滤镜感"的来源。

⚠️ 时序一致性：本模块输出的色板必须由调用方**在所有帧之间共享**。
   单帧调用时自动求色板很方便，做视频时务必用 ``palette_from_frames`` 求全局色板再传入。
"""

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from PIL import Image

from .dither import bayer_signed
from .palette import (
    ensure_neutral_highlight,
    LUMA,
    build_lut,
    from_perceptual,
    palette_from_image,
    refine_palette,
    snap_exact,
    snap_lut,
    to_perceptual,
)
from .resample import structure_aware_downsample, tone_map

__all__ = ["PixelTail", "pixelate", "render_frame"]


@dataclass
class PixelTail:
    """像素尾巴的全部旋钮。默认值取自 tools/grid_probe.py 的实测结果。"""

    # --- 降采样 ---
    var_gain: float = 45.0          # 结构感知权重增益；越大越保细节、也越容易出噪点

    # --- 影调 ---
    tone_black: float = 0.0
    tone_white: float = 0.95
    tone_scurve: float = 0.16

    # --- 调色板 ---
    n_colors: int = 32
    quantize_mode: str = "palette"  # palette = 色板吸附（像素画）；continuous = 连续色（高保真，跳过吸附）

    # --- 抖动 ---
    dither: float = 0.10            # 幅度上限（感知空间）
    dither_adaptive: bool = True    # 按亮度调制：中间调最强，纯黑/纯白关掉（避免暗部脏噪点）

    # --- 边缘压暗（模拟描边）---
    # edge_gain 实测标定：7.0 会让密集画面整体压暗 13%（处处触发），
    # 2.0 时漂移约 -2%~-5%，只在真正的强边界上留下描边。详见 out/step_probe 输出。
    edge_gain: float = 2.0
    edge_strength: float = 0.35

    # --- 性能 ---
    use_lut: bool = False           # True 用 3D LUT 近似吸附，快一个量级
    lut_levels: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pixelate(
    rgb01: np.ndarray,
    params: PixelTail | None = None,
    palette: np.ndarray | None = None,
    edge_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """在**像素网格分辨率**的连续色图上执行像素尾巴。

    Args:
        rgb01: (H, W, 3) float32 in [0, 1]，已降到目标网格。
        params: 旋钮；None 用默认值。
        palette: (k, 3) float32 in [0, 1]。**做视频时必须传入全局共享色板**。
        edge_ref: (H, W, 3) —— **只用于求边缘梯度**的参考图。``None`` 时用 ``rgb01``。

            ⚠️ 这个参数是必需的，不是可选优化。边缘压暗是**内容风格化**步骤
            （给物体轮廓描边），它假设输入图里的高频都是"内容边界"。
            但**粒子/尘埃是大气现象，不是内容** —— 单个亮像素在边缘检测器
            眼里就是"极强的边界"，于是粒子自己与相邻像素被压暗
            （实测单帧 3292 个像素被压暗、最深 82/255 色阶），
            观感就是**画面上有黑点在移动**（用户报障）。

            所以视频路径要把"无粒子的合成图"传进来当参考，
            让边缘只反映真实内容。（这与约束 5「梯度必须从抖动前的内容上求」
            是同一个道理的另一层：**也别从瞬时大气上求**。）
    """
    p = params or PixelTail()
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError("rgb01 必须是 (H, W, 3)")
    if edge_ref is not None and edge_ref.shape != rgb01.shape:
        raise ValueError(f"edge_ref 形状 {edge_ref.shape} 与 rgb01 形状 {rgb01.shape} 不一致")

    q = to_perceptual(rgb01)
    h, w = q.shape[:2]

    if palette is None:
        palette = palette_from_image(rgb01, p.n_colors)
        # median cut 按像素数量分配，会把"明亮的中性色"并进绿/蓝盒子里，
        # 导致白墙被涂成绿墙。两步补正：
        #   1) refine_palette         —— 赎回冗余槽位，给"明亮低饱和"区域一批
        #      专属色阶。ref10 白色建筑被涂成绿色就是缺这一步。
        #   2) ensure_neutral_highlight —— 廉价兜底，保证至少有一个中性亮色。
        #
        # ⚠️ 顺序不能反。先跑 ensure 的话，它补的那一个中性项会让
        #    refine_palette 的「该区域已被接住」判据成立，从而整个跳过 ——
        #    实测：先 ensure 后 refine，ref10 的 Δsat 只从 0.190 降到 0.149；
        #    先 refine 后 ensure，降到 0.012。少一项就骗过了守卫。
        palette = refine_palette(rgb01, palette)
        palette = ensure_neutral_highlight(rgb01, palette)
    palette = np.asarray(palette, dtype=np.float32)

    if p.use_lut:
        lut = build_lut(palette, p.lut_levels)

        def snap(x: np.ndarray) -> np.ndarray:
            return snap_lut(x, palette, lut)
    else:
        def snap(x: np.ndarray) -> np.ndarray:
            return snap_exact(x, palette)

    # 抖动：加到连续色上，再一起吸附
    src = q
    if p.dither > 0:
        if p.dither_adaptive:
            luma = q @ LUMA
            amp = np.clip(1.0 - np.abs(2.0 * luma - 1.0), 0.0, 1.0) ** 0.8
        else:
            amp = np.ones((h, w), dtype=np.float32)
        src = np.clip(q + bayer_signed(h, w) * (p.dither * amp)[..., None], 0.0, 1.0)

    # 取色模式（M5.10）：continuous = 跳过色板吸附（保留抖动与边缘压暗的高保真后处理）
    cont = p.quantize_mode == "continuous"
    out = src if cont else snap(src)

    # 边缘压暗：高对比边界压暗一点，模拟像素画手绘的描边
    #
    # ⚠️ 梯度必须从**抖动之前的内容**上算，不能从 out 上算。
    #    抖动本身就是高频信号，若在 out 上求梯度，整幅图会被判成"处处是边缘"，
    #    统一乘以 (1-strength)，实测会把画面整体压暗约 30%，
    #    表现为"像素化之后又灰又糊"。
    #
    # ⚠️ 也不能从**含粒子的**图层上算：粒子是单像素高频，
    #    会被判成强边界 → 粒子周围被压暗成黑点（见 edge_ref 的说明）。
    if p.edge_strength > 0:
        src_for_edge = rgb01 if edge_ref is None else edge_ref
        lum = to_perceptual(src_for_edge) @ LUMA
        gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
        g = np.clip((gx + gy) * p.edge_gain, 0.0, 1.0)[..., None]
        out = out * (1.0 - p.edge_strength * g)
        if not cont:
            out = snap(out)

    u8 = np.clip(from_perceptual(out) * 255.0, 0, 255).astype(np.uint8)
    return u8, palette


def render_frame(
    img: Image.Image,
    grid: tuple[int, int],
    params: PixelTail | None = None,
    palette: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """一步到位：原图 → 目标网格 → 像素尾巴。返回 (PIL 图（网格分辨率）, 色板)。

    这是最常用的入口。视频模式下把第一次得到的 ``palette`` 传进来复用即可。

    ⚠️ 需要区分"内容"与"粒子"时请用 ``pipeline`` 里的
    ``compose_frame`` + ``finish_frame``（后者支持 ``edge_ref``）。
    """
    p = params or PixelTail()
    f = structure_aware_downsample(img, grid, p.var_gain)
    f = tone_map(f, p.tone_black, p.tone_white, p.tone_scurve)
    u8, pal = pixelate(f, p, palette)
    return Image.fromarray(u8), pal
