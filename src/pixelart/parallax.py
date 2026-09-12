"""分层视差推拉（M5）—— 把深度切层做极缓慢的推拉，让视频"像 2.5D"。

═══ 为什么这是 v2 的重头戏 ═══

雾漂移 / 闪烁 / 尘埃都是"画面内部的微动"，整体仍然是**一张静止的照片**。
分层视差让近景与远景以不同幅度相对运动 —— 这是"有纵深"的最强线索，
也是 HANDOFF §9 列的观感收益最大的一项。

═══ 设计决定（每条都有理由）═══

**1. 闭合波形必须是 `1 - cos(τt)`，不能用 sin。**
   循环要求 frame(t=1) 与 frame(t=0) **逐位相同**（时序铁律 3）。
   `sin(τ·1) = -2.45e-16 ≠ sin(0) = 0`，逐位就不同了；
   而 `cos(τ·1)` 在 IEEE754 下**恰好精确等于 1.0**（float64 与 JS 同），
   所以 wave(0) = wave(1) = 0.0 逐位成立。实测过，不是想当然。

**2. 前向 splatting（带权重归一），不是后向采样。**
   后向采样不知道遮挡：前景移走后，它"拉"来的源像素可能是错的层的内容。
   前向 splat 把每个像素按位移摊到 4 个邻格（双线性权重），
   遮挡天然正确（前景权重与背景权重在同一格混合，前景写入更晚/更大）；
   没被任何 splat 覆盖的格 = **破洞**（disocclusion），用**原图该位置内容**兜底
   —— 推拉幅度只有几个像素，"没动够"比"拉错内容"不显眼得多，
   而且后面还有量化 + 抖动 + 边缘压暗兜着。
   覆盖率低的格（wsum 小）按覆盖率与原图线性混合，避免除以小权重放大噪声。

**3. 分层用分位数，不用深度等距。**
   真实深度分布高度不均（ Often 一半像素挤在 0.9~1.0 的远景区）。
   按分位数切层保证每层像素量接近，不会出现空层。

**4. 同层内位移完全一致（刚体层）。**
   这正是"分层"的含义：离散层比连续 morph 更像"纸片景深"，
   也避免层内拉丝。

**5. GPU 暂不移植。** 用户已决定 GPU 一章收尾；界面在 parallax > 0 时
   自动改用服务端渲染（见 m3_ui 的 ``gpuCanRender``）。
   之后要移植时，这里只有逐像素 gather/splat 一段，WGSL 化不难。
"""

from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi

#: 位移幅度相对**画面宽度**的比例（见 parallax_offsets 里的标定说明）。
#: amp=1、wave 峰值、最前层、画面边缘时的总位移 ≈ 1.7 · REL_AMPLITUDE · 幅宽。
REL_AMPLITUDE = 0.06
#: 推拉中心（归一化）。偏上一点：多数构图的地平线在上三分之一附近。
PIVOT = (0.5, 0.42)
#: 随层叠加的垂直漂移（乘在 k 上）—— 纯径向推拉太"对称"，会显得机械。
DRIFT = 0.35
#: 近层 / 远层的运动系数。
NEAR_GAIN = 1.0
FAR_GAIN = 0.12
#: 层数（分位数切层）。
LAYERS = 12
#: splat 的定点标度：每份贡献 = round(clip(v,0,1)·w·SPLAT_SCALE)，用 u32 累加。
#: ⚠️ 为什么必须定点：浮点加法不满足结合律，而 GPU 原子加顺序不确定 →
#:    每次运行都可能不同。整数加法精确且顺序无关（与尘埃同一条理由）。
#: 溢出上界：每个像素最多被 ~4·(1+d) 个源覆盖（推拉幅度 ≤10% 幅宽 → 汇聚系数小），
#:    acc ≤ 8·2^20 = 2^23 ≪ 2^32。取 2^20：量化误差 4.8e-7，比 u8 色阶小 8000 倍。
SPLAT_SCALE = 1 << 20

__all__ = ["parallax_wave", "layer_edges", "layer_weights_from_edges",
           "layer_weights", "parallax_offsets", "parallax_warp",
           "SPLAT_SCALE", "PIVOT", "DRIFT", "NEAR_GAIN", "FAR_GAIN", "LAYERS",
           "REL_AMPLITUDE"]


def parallax_wave(t: float) -> float:
    """推拉波形：0 → 2 → 0 的平滑起伏。

    ⭐ wave(0) = wave(1) = 0.0 **逐位精确**（见模块注释 1）——
    这是循环闭合（时序铁律 3）在这个特征上的立足点。
    """
    return 1.0 - float(np.cos(TAU * float(t)))


def layer_edges(far: np.ndarray, layers: int = 12) -> np.ndarray:
    """层边界（升序、去重）—— **GPU 侧必须由 CPU 算好传来**。

    WGSL 里没有 quantile / 排序，而"按分位数切层"是视觉上要的效果
    （等距切会在真实深度分布上切出空层）。所以边界在这里算一次，
    作为 **float 数组**放进场景包，浏览器只做"数一数有几个边界 ≤ far"
    这件能在着色器里做的事。

    ⚠️ 顺序必须与 :func:`layer_weights` 里完全一致 ——
    两边各算一次 = 迟早分叉（这个项目已经踩过 5 次同类问题）。
    验证：``layer_weights`` 与"用 edges 在 numpy 里重算"逐位相同（有测试）。
    """
    far = np.asarray(far, dtype=np.float32)
    layers = max(2, int(layers))
    return np.unique(np.quantile(far, np.linspace(0.0, 1.0, layers + 1))
                     ).astype(np.float32)


def layer_weights_from_edges(
    far: np.ndarray,
    edges: np.ndarray,
    near_gain: float = 1.0,
    far_gain: float = 0.12,
) -> np.ndarray:
    """用**给定的边界数组**算运动系数 —— 与 WGSL 侧同一套规则。

    规则（必须与 ``parallax_splat.wgsl`` 一字不差）：
      E = 边界数（切成 E-1 段）；``k = 数一数有几个 edges ≤ far``
      ``idx = clamp(k-1, 0, E-2)``；``u = idx / (E-2)``
      ``w = near + (far_gain-near)·u``

    ⚠️ 分母是 **E-2**（段数 - 1），让**最远段恰好落到 far_gain**。
    写成 E-1 会让远层多动 60%（着色器第一版就是这个错，端到端差 0.18）。
    ⚠️ E=2 时 E-2=0 → 除零；用 ``max(...,1)`` 兜住（此时只有一段，
    取 near_gain 是对的）。
    """
    far = np.asarray(far, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    if edges.size < 2:
        return np.full(far.shape, near_gain, dtype=np.float32)
    k = np.searchsorted(edges, far, side="right")
    nseg = max(int(edges.size) - 2, 1)
    idx = np.clip(k - 1, 0, nseg)
    u = idx / float(nseg)
    return (near_gain + (far_gain - near_gain) * u).astype(np.float32)


def layer_weights(
    far: np.ndarray,
    layers: int = 12,
    near_gain: float = 1.0,
    far_gain: float = 0.12,
) -> np.ndarray:
    """每个像素的**运动系数**：近（far≈0）动得多，远（far≈1）几乎不动。

    按分位数切层（设计决定 3）：每层像素量接近，不会出空层。
    同层内系数完全一致（设计决定 4：刚体层）。

    Args:
        far: (H, W) 深度，0 = 近，1 = 远（Scene 的约定）。
        layers: 层数。8~16 都合理；太少（<6）层跳明显，太多没有额外收益。
        near_gain / far_gain: 近层与远层的运动系数。
            远层不给 0 而给一个小值（0.12）：完全不动会让远景看起来
            "贴在玻璃后面"，与近层的运动脱节。
    """
    return layer_weights_from_edges(far, layer_edges(far, layers),
                                    near_gain=near_gain, far_gain=far_gain)


def parallax_offsets(
    far: np.ndarray,
    t: float,
    amp: float = 0.5,
    layers: int = LAYERS,
    near_gain: float = NEAR_GAIN,
    far_gain: float = FAR_GAIN,
    drift: float = DRIFT,
    pivot: tuple[float, float] = PIVOT,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """计算每像素位移（前向：内容往哪里搬）。

    返回 ``(ox, oy, info)``，单位是**像素**，形状同 ``far``。

    * 推拉：以 pivot 为中心的径向位移（近层系数大 → 近处扩张/收缩更明显）。
      幅度按**网格宽度的比例**标定（不是固定像素）——预览与成片观感一致。
    * 漂移：叠加一点随层的垂直位移（推拉太"对称"会显得机械）。
    * t=0 与 t=1 时波形成立为 0，直接走**零位移捷径** ——
      不仅快，更重要的是保证与原图**逐位相同**（循环闭合）。
    """
    far = np.asarray(far, dtype=np.float32)
    gh, gw = far.shape
    wave = parallax_wave(t)
    info = {"wave": wave, "holes_pct": 0.0, "max_off_px": 0.0}
    if amp <= 0 or abs(wave) < 1e-9:
        z = np.zeros((gh, gw), dtype=np.float32)
        return z, z.copy(), info

    if edges is None:
        wgt = layer_weights(far, layers=layers, near_gain=near_gain, far_gain=far_gain)
    else:
        wgt = layer_weights_from_edges(far, edges, near_gain=near_gain, far_gain=far_gain)

    # ⚠️ 归一化分母必须防零：1 像素高的输入会让 gh_-1=0 → 除零 → NaN
    #    （网格最小是 2，正常到不了；但"靠调用方保证"不该用在除法上 ——
    #     实测就是这样在测试脚本里炸出 NaN 的。）
    gh_ = max(gh - 1.0, 1.0)
    gw_ = max(gw - 1.0, 1.0)
    px, py = pivot[0] * gw_, pivot[1] * gh_
    # ⚠️⚠️ 位移必须**与网格成比例**，不能写成固定像素。
    #    原来写的是 `amp * 6.0`（固定 6px）：240 网格上最大位移 ~5px（2% 幅宽），
    #    而同一组参数在 480 成片网格上**相对位移只有一半** ——
    #    直接违背"预览与成片观感一致"（与"显示倍率必须整数倍"是同一类不变量）。
    #    实测（ref04）：固定像素版在 amp=0.4 时"滤掉抖动后的结构差异"仅 1.8%，
    #    用户根本看不出来（实测反馈"我肉眼看不出来"）。
    #    现在按 **幅宽比例**标定：`k = amp · 0.06 · gw · wave`。
    #    换算：最前层、画面边缘、wave 峰值时总位移 ≈ 1.7·amp·0.06·gw
    #      amp=0.25 → 2.5% 幅宽（轻微呼吸）
    #      amp=0.50 → 5.0% 幅宽（明显，推荐起步值）
    #      amp=1.00 → 10%  幅宽（强推拉，会开始露层边界）
    #    实测（ref04）：固定像素版在相当于 amp=0.4 时"滤掉抖动后的结构差异"
    #    只有 1.8%，用户看不出；这一版 amp=0.5 约 5%，动起来一眼可见。
    k = float(amp) * REL_AMPLITUDE * gw * wave
    xs = np.arange(gw, dtype=np.float32)[None, :]
    ys = np.arange(gh, dtype=np.float32)[:, None]
    ox = k * wgt * (xs - px) / gw_
    oy = k * wgt * ((ys - py) / gh_ + drift)
    info["max_off_px"] = float(np.abs(np.stack([ox, oy])).max())
    return ox.astype(np.float32), oy.astype(np.float32), info


def _splat(plane: np.ndarray, xx: np.ndarray, yy: np.ndarray,
           w4: np.ndarray, gh: int, gw: int) -> tuple[np.ndarray, np.ndarray]:
    """把 (plane * w4) 按目标格散加（bincount 展平，比 np.add.at 快一个量级）。

    ``plane`` 必须是 2D (H, W) —— 多通道由调用方逐通道调用。
    """
    assert plane.ndim == 2, "splat 只收 2D 平面"
    flat = (yy * gw + xx).ravel()
    vals = (plane * w4).ravel()
    out = np.bincount(flat, weights=vals, minlength=gh * gw)
    acc = np.bincount(flat, weights=w4.ravel(), minlength=gh * gw)
    return out.reshape(gh, gw), acc.reshape(gh, gw)


def parallax_warp(
    base: np.ndarray,
    far: np.ndarray,
    t: float,
    amp: float = 0.5,
    layers: int = LAYERS,
    near_gain: float = NEAR_GAIN,
    far_gain: float = FAR_GAIN,
    drift: float = DRIFT,
    pivot: tuple[float, float] = PIVOT,
    edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """按视差位移搬 ``base``（颜色）与 ``far``（深度），返回 ``(base', far', info)``。

    * **前向 splatting**：每像素按双线性权重摊到 4 个邻格（设计决定 2）。
    * **破洞兜底**：没被覆盖（或覆盖极薄）的格用**原图内容**按覆盖率混合 ——
      推拉幅度只有几像素，"没动够"远比"拉错层"不显眼。
    * amp ≤ 0 或波形为 0（t=0 / t=1）→ **原数组原样返回**，逐位相同。
    """
    base = np.asarray(base, dtype=np.float32)
    far = np.asarray(far, dtype=np.float32)
    gh, gw = far.shape

    ox, oy, info = parallax_offsets(
        far, t, amp=amp, layers=layers, near_gain=near_gain,
        far_gain=far_gain, drift=drift, pivot=pivot, edges=edges,
    )
    if info["wave"] == 0.0 or amp <= 0:
        return base, far, info

    nx = np.arange(gw, dtype=np.float32)[None, :] + ox
    ny = np.arange(gh, dtype=np.float32)[:, None] + oy
    x0 = np.floor(nx).astype(np.int64)
    y0 = np.floor(ny).astype(np.int64)
    fx = (nx - x0).astype(np.float32)
    fy = (ny - y0).astype(np.float32)

    out_b = np.zeros((gh, gw, 3), dtype=np.float32)
    far_acc = np.zeros((gh, gw), dtype=np.float32)
    wacc = np.zeros((gh, gw), dtype=np.float32)
    for dyi in (0, 1):
        for dxi in (0, 1):
            xx = np.clip(x0 + dxi, 0, gw - 1)
            yy = np.clip(y0 + dyi, 0, gh - 1)
            w4 = (fx if dxi else 1.0 - fx) * (fy if dyi else 1.0 - fy)
            for c in range(3):
                sb, _ = _splat(base[..., c], xx, yy, w4, gh, gw)
                out_b[..., c] += sb
            sf, sa = _splat(far, xx, yy, w4, gh, gw)
            far_acc += sf
            wacc += sa

    # 破洞 / 覆盖过薄：按覆盖率与原图混合（设计决定 2 的兜底）
    alpha = np.clip(wacc * 2.0, 0.0, 1.0)[..., None]
    safe = np.where(wacc > 0, wacc, 1.0)[..., None]
    out_b = out_b / safe
    out_b = out_b * alpha + base * (1.0 - alpha)
    falpha = np.clip(wacc * 2.0, 0.0, 1.0)
    fsafe = np.where(wacc > 0, wacc, 1.0)
    out_f = far_acc / fsafe
    out_f = out_f * falpha + far * (1.0 - falpha)

    info["holes_pct"] = float((wacc <= 0).mean() * 100.0)
    return out_b.astype(np.float32), out_f.astype(np.float32), info
