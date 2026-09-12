"""时间参数 —— 把「动」做成一个普通参数。

核心思想（arch-01 §8 D5/D6）：**时间不是一个模式，而是函数的第 N 个参数。**

- ``t`` 取一个固定值 → 静态图
- ``t`` 从 0 扫到 1 → 一段无缝循环视频

同一份代码，两条输出。静态图就是视频的第 0 帧。

⚠️ 三条铁律（每一条都对应一个会毁掉成片的事故）：

1. **一切随时间变化的量必须严格以 1 为周期。**
   实现手段只有两类 ——
   **(a) 整数频率正弦**：``sin(2π k t)``，k 取整数则 ``t`` 加 1 后值不变；
   **(b) 三维循环噪声**：把 ``t`` 当成噪声体的第三维，且该维可变周期。
   只要还在用第三类（比如 ``np.random`` 或无界相位），循环处必有接缝。
   自检：``frame(0) == frame(N)``。

2. **抖动相位只能依赖屏幕坐标，表达式里绝不能出现 t。**
   出现 t 就是整屏闪烁。抖动在 ``dither.py`` 里，别往那儿传 t。

3. **色板必须全片共用一块。**
   逐帧各求色板 = 整片闪烁。用 ``palette.palette_from_frames`` 求一次。

**为什么频率要取互质的整数**：``sin(2π·1t)``、``sin(2π·3t)``、``sin(2π·7t)``、
``sin(2π·11t)`` 叠加后，整体周期仍是 1，但**在 1 个周期内不会提前重复**
（互质数的最小公倍数是它们的乘积）。这样闪烁看起来是"随机"的，
却又是严格循环的 —— 既无缝又不自暴重复。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GOLDEN_PHASE",
    "DEFAULT_FLICKER_FREQS",
    "periodic_sines",
    "flicker_gain",
    "loop_noise_2d",
    "dust_layer",
    "hash01",
]

#: 默认闪烁频率：四个互质整数。周期都是 1，但合成波形在整个周期内不重复。
DEFAULT_FLICKER_FREQS: tuple[int, ...] = (1, 3, 7, 11)

#: 用黄金比生成"看起来随机、实际确定"的相位，避免手写一堆魔法数。
GOLDEN_PHASE = 0.6180339887498949


def hash01(*keys: int) -> float:
    """把若干个整数散列成 [0, 1) 的确定性伪随机值。

    用来给粒子/相位之类的东西生成"固定的随机"——固定 seed 的哈希噪声，
    **绝不每帧重采样**（那会让粒子逐帧抖动）。

    用的是 splitmix64 的简化版，纯整数运算，跨平台可复现。
    """
    h = 0x9E3779B97F4A7C15
    for k in keys:
        h ^= (int(k) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 30)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
    return (h & 0xFFFFFFFFFFFFFFF) / float(0x1000000000000000)


def periodic_sines(
    t: float,
    freqs: tuple[int, ...] = DEFAULT_FLICKER_FREQS,
    seed: int = 0,
    weights: tuple[float, ...] | None = None,
) -> float:
    """整数频率正弦叠加，返回 **[-1, 1]** 的周期信号（周期严格为 1）。

    每一项是 ``w_k · sin(2π f_k t + φ_k)``，``f_k`` 必须是整数。
    结果按权重和归一化，保证上界是 1 —— 这样调用方拿到的是一个"标准信号"，
    乘上幅度即可，不用担心叠出界。

    Args:
        freqs: 整数频率列表。取互质数可避免周期内提前重复。
        seed: 相位的固定种子（同一 seed 每次得到同样的波形）。
        weights: 各项权重；``None`` 时按 ``1/k`` 递减（低频主导，像呼吸）。
    """
    if not freqs:
        return 0.0
    if any(int(f) != f for f in freqs):
        raise ValueError(f"频率必须是整数（否则周期不是 1）：{freqs}")
    if weights is None:
        weights = tuple(1.0 / (i + 1) for i in range(len(freqs)))
    if len(weights) != len(freqs):
        raise ValueError("weights 长度必须与 freqs 一致")

    acc = 0.0
    for i, (f, w) in enumerate(zip(freqs, weights)):
        phase = 2.0 * np.pi * hash01(seed, i)
        acc += w * np.sin(2.0 * np.pi * int(f) * t + phase)
    return float(acc / max(sum(abs(w) for w in weights), 1e-9))


def flicker_gain(
    t: float,
    freqs: tuple[int, ...] = DEFAULT_FLICKER_FREQS,
    depth: float = 0.18,
    seed: int = 0,
    bias: float = 1.0,
) -> float:
    """光源闪烁的**乘性**增益，返回值落在 ``[bias - depth, bias + depth]`` 内。

    用在体积光/辉光的强度上：``strength * flicker_gain(t)``。

    ⚠️ ``depth`` 是**上界**不是实际振幅：叠加信号内部已按权重归一化，
    实际峰值取决于频率组合（默认 ``(1,3,7,11)`` 只跑到约 ±0.68 倍 depth）。
    想要某个确定振幅就自己乘一下。

    ``depth`` 别开大：真实光源的闪烁是**细微**的，0.15~0.25 已经很明显；
    再大就变成"有人在手抖开关"，而且会让亮度整体起伏（观感很廉价）。
    """
    if depth <= 0:
        return float(bias)
    return float(bias + depth * periodic_sines(t, freqs, seed))


def loop_noise_2d(
    t: float,
    shape: tuple[int, int],
    octaves: int = 3,
    seed: int = 0,
    base_freq: float = 1.5,
) -> np.ndarray:
    """**严格循环**的二维噪声场，返回 ``(H, W)``，值域约 ``[-1, 1]``。

    做法是**谱合成**：把噪声表示成若干空间正弦的和，
    每一项的时间频率都取整数 ——

    ``N(x, y, t) = Σ a_k · sin(2π(fx_k·x + fy_k·y + ft_k·t) + φ_k)``

    ``ft_k`` 是整数，所以 ``N(x, y, t + 1) == N(x, y, t)``，**逐位相等**，
    循环处不可能有接缝。这正是"三维循环噪声"在工程上的落地方式：
    不必去造一个真三维噪声体，只要保证时间轴上的频率是有理数即可。

    用 ``x, y`` 归一化到 [0,1) 再乘空间频率，所以换分辨率不会改变观感。

    Args:
        t: 时间，建议已归一化到 [0, 1)。
        shape: 输出 ``(H, W)``。
        octaves: 叠加多少层。每层空间频率翻倍（标准 fBm）。
        base_freq: 最低层的空间频率。
    """
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("shape 必须为正")

    yy = (np.arange(h, dtype=np.float32) + 0.5) / h
    xx = (np.arange(w, dtype=np.float32) + 0.5) / w
    gx, gy = np.meshgrid(xx, yy, indexing="xy")

    acc = np.zeros((h, w), dtype=np.float32)
    amp_sum = 0.0
    for k in range(max(1, int(octaves))):
        amp = 0.5 ** k
        f = base_freq * (2.0 ** k)
        # 每层的方向、时间频率、相位都由固定 seed 决定
        ang = 2.0 * np.pi * hash01(seed, k, 1)
        fx, fy = f * np.cos(ang), f * np.sin(ang)
        # ⚠️ 时间频率必须是整数，否则循环有接缝
        ft = int(round(1.0 + 4.0 * hash01(seed, k, 2)))
        ph = 2.0 * np.pi * hash01(seed, k, 3)
        acc += amp * np.sin(2.0 * np.pi * (fx * gx + fy * gy + ft * t) + ph)
        amp_sum += amp

    return (acc / max(amp_sum, 1e-9)).astype(np.float32)


def dust_layer(
    shape: tuple[int, int],
    far01: np.ndarray | None = None,
    t: float = 0.0,
    count: int = 220,
    seed: int = 0,
    color=(1.0, 1.0, 1.0),
    light_xy: tuple[float, float] | None = None,
    light_boost: float = 1.6,
    twinkle: float = 0.55,
    fade_far: float = 0.0,
    drift: float = 0.045,
) -> np.ndarray:
    """漂浮尘埃/微粒的**加性**图层，返回 ``(H, W, 3)``。

    这是"有颗粒感"的来源，也是体积光真正被"看见"的载体 ——
    纯梯度是看不见的，光必须打在东西上才显形。

    ⚠️ 返回的图层是**纯加性**的（只会让画面变亮）。如果最终画面出现
    **暗点**，那不是这里的问题，而是**下游把粒子误判成了内容边界** ——
    见 ``pixelate(edge_ref=...)``。实测事故：粒子在边缘检测器眼里是
    "单像素强边界"，导致粒子周围被压暗、深达 82/255 色阶，
    观感就是"黑点在移动"。

    ⚠️ 运动模型是**有界摆动**，不是匀速直行。曾经用
    ``frac(x0 + k·t)``（整数圈数直线位移），结果是每颗粒子以恒定速度
    **横穿整个画面**并在边缘折返 —— 观感就是"蚂蚁在直线爬行"（用户报障）。
    真实浮尘是被气流带着**缓慢游荡**的，不会匀速直线穿越视野。
    现在改用 **Lissajous 式摆动**：``x = x0 + ax·sin(2π fx t + φx)``，
    整条路径有界、平滑、不折返，而且严格周期。

    三条周期性设计，缺一条循环就有接缝：

    1. **摆动频率取整数**，``t`` 加 1 后相位精确回到原处。
    2. **闪烁用整数频率正弦**，不是随机数。
    3. **位置/相位/频率全部来自 :func:`hash01`**，固定 seed、每帧重算但结果相同 ——
       绝不是"每帧重新随机播撒"（那会让粒子逐帧抖动成噪点）。

    渲染在**像素网格分辨率**上，每颗尘埃就是一个像素方块 —— 这正好符合
    像素画的语汇（像素画里的尘埃就是单个亮点）。返回加性图层，
    调用方直接加到画面上。

    Args:
        shape: ``(H, W)``，应当是像素网格分辨率。
        far01: ``(H, W)`` 深度；给定时远处尘埃会被减弱，且靠近光源的更亮。
        t: 时间，[0, 1) 为一个循环。
        count: 尘埃数量。240x480 网格上 150~300 比较自然；再多就成噪点了。
        seed: 固定随机种子。
        color: 尘埃颜色。
        light_xy: 光源归一化位置；给定时附近尘埃加亮（"光柱里的尘埃"）。
        light_boost: 光源附近尘埃的增亮倍数。
        twinkle: 闪烁深度（0 = 不闪）。
        fade_far: 远处尘埃的减弱强度（0 = 不减弱）。给 0.6~1.0 可避免
            远景浮着一层均匀的噪点。
        drift: 摆动幅度（占画面长边的比例）。0.045 ≈ 4.5%，
            在 480 宽上是约 22 px 的游荡范围。**别调大** ——
            幅度一大就又有"在飞"的感觉，而不是"悬浮"。
    """
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("shape 必须为正")
    if far01 is not None and far01.shape[:2] != (h, w):
        raise ValueError(f"far01 形状 {far01.shape[:2]} 与 shape {(h, w)} 不一致")

    out = np.zeros((h, w), dtype=np.float32)
    col = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    if count <= 0:
        return out[..., None] * col

    for i in range(int(count)):
        # 基准位置铺满画面；摆动叠加在它上面
        x0 = hash01(seed, i, 1)
        y0 = hash01(seed, i, 2)

        # 摆动幅度有差异，避免整片"齐步走"
        ax = drift * (0.30 + 0.70 * hash01(seed, i, 3))
        ay = drift * (0.30 + 0.70 * hash01(seed, i, 4))
        # ⚠️ 整数频率（1~3 次每循环）。x/y 取不同频率 → 开放式 Lissajous
        #    轨迹，不会原路折返。
        fx = 1 + int(hash01(seed, i, 5) * 2.999)
        fy = 1 + int(hash01(seed, i, 6) * 2.999)
        phx = 2.0 * np.pi * hash01(seed, i, 7)
        phy = 2.0 * np.pi * hash01(seed, i, 8)

        nx = (x0 + ax * np.sin(2.0 * np.pi * fx * t + phx)) % 1.0
        ny = (y0 + ay * np.sin(2.0 * np.pi * fy * t + phy)) % 1.0
        px = int(nx * w) % w
        py = int(ny * h) % h

        # 闪烁：整数频率（1~5 次每循环），严格周期
        tf = 1 + int(hash01(seed, i, 9) * 4.999)
        tp = 2.0 * np.pi * hash01(seed, i, 10)
        tw = 1.0 - twinkle * 0.5 * (1.0 - np.sin(2.0 * np.pi * tf * t + tp))

        a = 0.35 + 0.65 * hash01(seed, i, 11)      # 基础亮度差异
        a *= tw

        if far01 is not None and fade_far > 0:
            a *= float(1.0 - fade_far * np.clip(far01[py, px], 0, 1))
        if light_xy is not None:
            # 靠近光源的尘埃更亮（仿佛被光柱照亮），按归一化距离衰减
            dx = (px / max(w - 1, 1)) - light_xy[0]
            dy = (py / max(h - 1, 1)) - light_xy[1]
            a *= 1.0 + (light_boost - 1.0) / (1.0 + 6.0 * np.hypot(dx, dy))

        out[py, px] += float(np.clip(a, 0.0, 1.0))

    return np.clip(out, 0.0, 1.0)[..., None] * col
