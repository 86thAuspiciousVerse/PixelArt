"""WebGPU 渲染路径的**数值打包** —— 只在 CPU 侧做一次，结果同时喂给
验收台（``tools/wgsl_lab.py``）和浏览器（经 ``/api/scene``）。

═══ 为什么单独一个模块 ═══

WGSL 里有几样东西没法在着色器里现算：

1. **数组尺寸的常量**（网格 W/H），以及只在"建场景"时定下的一次性估计
   （光源位置、雾色、散射色）。
2. **全局归约量**。``depth_fog`` 的雾漂移里有 ``n.mean()`` ——
   那是**整幅图的均值**，着色器逐像素算不了（要额外的归约 pass，
   而它每帧只有一个标量，不值得）。
3. **噪声的谐波系数**。``loop_noise_2d`` 的每层方向/频率/相位由 ``hash01``
   决定，只跟 seed 有关 —— 算一次即可，没必要让每个像素重复算哈希。

这些量如果**两边各写一份**，迟早会漂移：改了 Python 忘了改 JS，
画面就会"差不多但不对"，而且极难定位（这正是本项目反复踩过的类型）。

所以统一在这里算，序列化成一段 float32，作为 uniform 传给着色器，
或者放进场景包发给浏览器。**单一来源**。
"""

from __future__ import annotations

import numpy as np

from .animate import hash01, loop_noise_2d

__all__ = [
    "TWO_PI",
    "dust_particle_params",
    "dust_fixed_point_scale",
    "dust_splat_uniforms",
    "dust_apply_uniforms",
    "bloom_kernels",
    "bloom_bright_uniforms",
    "bloom_blur_uniforms",
    "bloom_combine_uniforms",
    "FOG_DRIFT_SEED",
    "fog_harmonics",
    "fog_uniforms",
    "scatter_uniforms",
    "volumetric_uniforms",
    "FOG_UNIFORM_VEC4",
    "SCATTER_UNIFORM_VEC4",
    "VOLUMETRIC_UNIFORM_VEC4",
    "parallax_layout", "parallax_splat_uniforms", "parallax_apply_uniforms",
]

TWO_PI = 6.283185307179586

#: 雾漂移噪声的种子。与 ``compose.depth_fog`` 里 ``loop_noise_2d(..., seed=17)`` 一致。
FOG_DRIFT_SEED = 17

#: fog 阶段的 uniform 占几个 vec4（数组布局，避免 struct 对齐踩坑）
FOG_UNIFORM_VEC4 = 6


def fog_harmonics(
    seed: int = FOG_DRIFT_SEED,
    octaves: int = 2,
    base_freq: float = 1.5,
) -> list[dict]:
    """复算 ``loop_noise_2d`` 的谐波系数（与 ``animate.loop_noise_2d`` 逐步对应）。

    ⚠️ 必须与 ``animate.loop_noise_2d`` 里的推导**逐行对应**，
    否则噪声场就变了、雾的漂移也就不再逐位复现。

    返回每层的 ``{amp, fx, fy, ft, phase}``；``ft`` 是**整数**（循环性的来源）。
    """
    out = []
    for k in range(max(1, int(octaves))):
        amp = 0.5 ** k
        f = base_freq * (2.0 ** k)
        ang = TWO_PI * hash01(seed, k, 1)
        fx, fy = f * np.cos(ang), f * np.sin(ang)
        ft = int(round(1.0 + 4.0 * hash01(seed, k, 2)))     # ⚠️ 整数，否则循环有接缝
        ph = TWO_PI * hash01(seed, k, 3)
        out.append({"amp": amp, "fx": fx, "fy": fy, "ft": ft, "phase": ph})
    return out


def grid_phasor(a: float, n: int) -> complex:
    """``(1/n) · Σ_{m=0}^{n-1} e^{i·2π·a·(m+0.5)/n}`` 的**闭式解**。

    ⚠️ 这个函数让"噪声场的网格均值"从 O(W·H) 变成 **O(1)**，
    从而让浏览器端每帧能算出**精确**的 ``noise_mean`` ——
    否则要么每帧算 13 万个点，要么发一张查找表（有插值误差，
    而这个量在 t 上峰峰有 **0.27**，不能用常数或粗糙表近似）。

    推导：把和写成等比级数
    ``Σ_m q^m = (1-q^n)/(1-q)``（``q = e^{i2πa/n}``），再用半角公式
    ``1-e^{iθ} = -2i·e^{iθ/2}·sin(θ/2)`` 化简，虚部相消后得到

    ``S(a, n) = e^{iπa} · sin(πa) / (n · sin(πa/n))``

    当 ``a/n`` 是整数时 ``q = 1``、上式取 0/0 —— 此时每一项都是 ``e^{iπa}``，
    所以直接返回 ``(-1)^(a/n)``（与上式的极限一致）。

    实测与 ``loop_noise_2d(...).mean()`` 吻合到 **1e-8**
    （那只是 float64 求和的舍入，不是公式误差），见
    ``tools/uniform_parity_probe.py``。
    """
    a = float(a)
    n = max(1, int(n))
    m = a / n
    if abs(m - round(m)) < 1e-12:
        return complex((-1.0) ** (int(round(m)) % 2))
    import cmath
    import math
    return (cmath.exp(1j * math.pi * a) * math.sin(math.pi * a)
            / (n * math.sin(math.pi * a / n)))


def fog_noise_mean(t: float, w: int, h: int,
                   harmonics: list[dict] | None = None) -> float:
    """``loop_noise_2d(t, (h, w), ...)`` 的**网格均值**，O(1) 闭式。

    ``mean = Σ_k amp_k · Im{ e^{i(φ_k + 2π·ft_k·t)} · S(fx_k, w) · S(fy_k, h) } / Σ amp_k``

    ⚠️ ``ft_k`` 是整数 → ``t`` 加 1 时相位精确回到原处
    → **``fog_noise_mean(1)`` 与 ``fog_noise_mean(0)`` 逐位相同**，
    时序铁律 3 因此成立（实测确实相同）。
    """
    import cmath
    import math

    hs = harmonics if harmonics is not None else fog_harmonics()
    acc = 0.0
    amp_sum = 0.0
    for hh in hs:
        # ⚠️ 时间项用 fract(ft·t) 而不是 ft·t —— 与雾的着色器同一条理由。
        #    数学上等价（ft 是整数，相差整数倍 2π），但 t=1 时
        #    `ft·1.0` 让相位变成 6π/8π，exp 的结果与相位 0 **不是逐位相同**；
        #    而 `fract(ft·1.0)` 对整数 ft **精确等于 0.0** → t=1 与 t=0 逐位相同。
        #    （实测：不加 fract 时循环闭合判定为 False。）
        ft = int(hh["ft"])
        ph_t = TWO_PI * (float(ft) * float(t) - math.floor(float(ft) * float(t)))
        m = (cmath.exp(1j * (hh["phase"] + ph_t))
             * grid_phasor(hh["fx"], w) * grid_phasor(hh["fy"], h))
        acc += hh["amp"] * m.imag
        amp_sum += hh["amp"]
    return acc / max(amp_sum, 1e-9)


def fog_uniforms(
    w: int,
    h: int,
    t: float,
    fog_color,
    density: float,
    power: float,
    floor: float = 0.0,
    ceiling: float = 1.0,
    drift: float = 0.0,
    harmonics: list[dict] | None = None,
    noise_mean: float | None = None,
    fog_tint: float = 1.0,
) -> np.ndarray:
    """打包 fog 阶段的 uniform（``FOG_UNIFORM_VEC4`` 个 vec4，float32）。

    布局::

        U[0] = (fog_r, fog_g, fog_b, —)
        U[1] = (density, power, floor, 1/(ceiling-floor))
        U[2] = (drift, noise_mean, W, H)
        U[3] = (fx0, fy0, ft0, phase0)
        U[4] = (fx1, fy1, ft1, phase1)
        U[5] = (amp0, amp1, 1/Σamp, t)

    ``noise_mean`` 不传时在这里现算 —— 它只依赖 ``(t, h, w)``，与画面内容无关。
    """
    hs = harmonics if harmonics is not None else fog_harmonics()
    if len(hs) != 2:
        raise ValueError(f"fog 的着色器按 2 层谐波写死了，收到 {len(hs)} 层")

    if noise_mean is None:
        # ⚠️ 用闭式解（O(1)），不要用 `loop_noise_2d(t, (h,w)).mean()` ——
        #    后者要先把整张 W×H 的噪声场算出来再取均值，只是为了得到**一个标量**。
        #    实测两者吻合到 1e-8（float64 求和舍入），而 480×270 下快了三个数量级。
        nm = fog_noise_mean(t, w, h, hs)
    else:
        nm = float(noise_mean)

    amp_sum = sum(hh["amp"] for hh in hs)
    c = np.asarray(fog_color, dtype=np.float32).ravel()
    if c.size != 3:
        raise ValueError("fog_color 必须是 3 个分量")

    u = np.zeros((FOG_UNIFORM_VEC4, 4), dtype=np.float32)
    u[0] = [c[0], c[1], c[2], float(fog_tint)]
    u[1] = [float(density), float(power), float(floor),
            1.0 / max(1e-6, float(ceiling) - float(floor))]
    u[2] = [float(drift), nm, float(w), float(h)]
    for i, hh in enumerate(hs):
        u[3 + i] = [hh["fx"], hh["fy"], float(hh["ft"]), hh["phase"]]
    u[5] = [hs[0]["amp"], hs[1]["amp"], 1.0 / max(amp_sum, 1e-9), float(t)]
    return u.reshape(-1)


#: scatter 阶段的 uniform 占几个 vec4
SCATTER_UNIFORM_VEC4 = 2

#: volumetric 阶段的 uniform 占几个 vec4
VOLUMETRIC_UNIFORM_VEC4 = 6   # M5.1 U[4] 光锥；M5.2 U[5] 光柱强度


def scatter_uniforms(w: int, h: int, lx: int, ly: int,
                     radius: int = 3, sat_max: float = 0.25) -> np.ndarray:
    """打包散射色采样 pass 的 uniform。

    ``U[0] = (lx, ly, W, H)``，``U[1] = (radius, sat_max, —, —)``
    """
    u = np.zeros((SCATTER_UNIFORM_VEC4, 4), dtype=np.float32)
    u[0] = [float(lx), float(ly), float(w), float(h)]
    u[1] = [float(radius), float(sat_max), 0.0, 0.0]
    return u.reshape(-1)


def volumetric_uniforms(
    w: int, h: int,
    samples: int = 28,
    span: float = 0.85,
    decay: float = 0.965,
    strength: float = 0.85,
    occlude_gain: float = 5.0,
    falloff_gain: float = 2.5,
    screen_falloff: float = 1.0,
    threshold: float = 0.48,
    knee: float = 0.3,
    gain: float = 1.0,
    lx: int = 0,
    ly: int = 0,
    cone_angle: float = 0.0,
    cone_dir_deg: float = 80.0,
    cone_reach: float = 0.8,
    cone_gain: float = 1.6,
    shaft: float = 0.0,
    fog_tint: float = 1.0,
    cone_x: float = -1.0,
    cone_y: float = -1.0,
) -> np.ndarray:
    """打包体积光 pass 的 uniform（见 volumetric.wgsl 顶部注释）。

    ⚠️ ``gain`` 是 ``animate.flicker_gain(t, depth=flicker)`` 的结果。
    它是四个整数频率正弦的叠加，**每帧只有一个标量** ——
    放进着色器会让每个像素重复算同样的常量，且容易和 CPU 版算得不一样。
    """
    u = np.zeros((VOLUMETRIC_UNIFORM_VEC4, 4), dtype=np.float32)
    u[0] = [float(samples), float(span), float(decay), float(strength)]
    u[1] = [float(occlude_gain), float(falloff_gain),
            float(screen_falloff), float(threshold)]
    u[2] = [float(knee), float(gain), float(w), float(h)]
    u[3] = [float(lx), float(ly), 0.0, 0.0]
    # ⭐ 锥顶点（M5.4）：槽 U[3].zw。**打包侧解析**：<0 = 跟随光源 → 代入 lx/ly
    #    （默认逐位不变）；≥0 = 归一化坐标 × 网格。WGSL 锥段/光柱段读 .zw。
    #    ⚠️ 舍入必须与 JS 镜像同一字：floor(x+0.5)（Python round 是银行家舍入、
    #    JS Math.round 对负数方向不同 —— 命中 .5 时两侧差 1，探针已抓到过）。
    u[3][2] = float(lx) if cone_x < 0 else float(int(cone_x * (w - 1) + 0.5))
    u[3][3] = float(ly) if cone_y < 0 else float(int(cone_y * (h - 1) + 0.5))
    # ── 光锥（M5.1）── 角度在**这里**从度换成弧度：只有这一处转换，
    #    JS 镜像也必须做同样的转换（由 uniform 逐字节比对守着）。
    u[4] = [float(cone_angle), float(np.radians(cone_dir_deg)),
            float(cone_reach), float(cone_gain)]
    u[5] = [float(shaft), float(fog_tint), 0.0, 0.0]
    return u.reshape(-1)


# ══════════════════════════════════════════════════════════════════
# 辉光（bloom）
# ══════════════════════════════════════════════════════════════════
def bloom_kernels(
    radii=(2.0, 5.0, 11.0),
    truncate: float = 3.0,
) -> tuple[np.ndarray, list[tuple[int, int, int, float]]]:
    """把三个尺度的高斯核拼成一段 float32，并给出查表。

    ⚠️ 核系数**必须由 CPU 算好上传**，不能让 WGSL 现算 ——
    ``exp`` 在 GPU 与 libm 上末位不同，同一个 σ 会得到两份略有差异的核，
    经几十个抽头累积后就被放大成"GPU 的模糊和 CPU 不是同一个模糊"。
    这里算一次、两边共用（blur pass 与参照实现都读这份），
    于是"核"是共享常量而不是两份各自近似的东西。

    Returns:
        ``(flat_kernel, table)``；``table[i] = (offset, count, half, weight)``，
        ``weight`` 是尺度权重 ``1/(i+1)``（与 ``compose.bloom`` 一致）。
    """
    from .compose import gaussian_kernel

    flat: list[np.ndarray] = []
    table: list[tuple[int, int, int, float]] = []
    off = 0
    for i, r in enumerate(radii):
        k = gaussian_kernel(r, truncate)
        flat.append(k)
        half = (len(k) - 1) // 2
        table.append((off, len(k), half, 1.0 / (i + 1)))
        off += len(k)
    return np.concatenate(flat).astype(np.float32), table


def bloom_bright_uniforms(w: int, h: int, threshold: float, knee: float) -> np.ndarray:
    u = np.zeros((1, 4), dtype=np.float32)
    u[0] = [float(w), float(h), float(threshold), float(knee)]
    return u.reshape(-1)


def bloom_blur_uniforms(w: int, h: int, direction: str,
                        entry: tuple[int, int, int, float],
                        accum: bool) -> np.ndarray:
    """``direction`` 取 ``"h"`` 或 ``"v"``。"""
    off, cnt, half, weight = entry
    u = np.zeros((2, 4), dtype=np.float32)
    u[0] = [float(w), float(h), 0.0 if direction == "h" else 1.0,
            1.0 if accum else 0.0]
    u[1] = [float(off), float(cnt), float(half), float(weight)]
    return u.reshape(-1)


def bloom_combine_uniforms(w: int, h: int, strength: float, radii) -> np.ndarray:
    """``mult`` = ``strength / Σ(1/(i+1))`` —— 与 bloom 的归一化一致。"""
    wsum = sum(1.0 / (i + 1) for i in range(len(radii)))
    u = np.zeros((1, 4), dtype=np.float32)
    u[0] = [float(w), float(h), float(strength) / max(wsum, 1e-6), 0.0]
    return u.reshape(-1)


# ══════════════════════════════════════════════════════════════════
# 尘埃粒子
# ══════════════════════════════════════════════════════════════════
#: 每颗粒子在参数缓冲里占几个 float
DUST_STRIDE = 11

#: 定点标度的上限（2^20 → 每颗量化误差 ≤ 4.8e-7）
DUST_MAX_SCALE = 1 << 20


def dust_particle_params(count: int, seed: int = 0, drift: float = 0.045) -> np.ndarray:
    """把每颗粒子的**静态**参数算好，形状 ``(count, 11)`` float32。

    列顺序：``x0 y0 ax ay fx fy phx phy tf tp base``
    —— 与 ``animate.dust_layer`` 里的推导**逐行对应**。

    ⚠️ 为什么由 CPU 算而不是在 WGSL 里算：

    ``animate.hash01`` 是 splitmix64 —— **WGSL 没有 64 位整数**
    （WebGPU 规范里没有 i64/u64，只有需要扩展的 shader-int64，不能依赖）。
    与其在 WGSL 里用两个 u32 手搓 64 位乘法和移位（又长又容易错），
    不如在 CPU 算一遍：

      · 没有 64 位运算 → 没有可移植性风险
      · **hash 只有一份实现** → 不可能两边漂移
      · 缓冲区很小，而且**每场景只算一次**（与帧数无关）

    这和 bloom 的核系数是同一个思路：能在 CPU 算一次的静态量，
    就不要在两个地方各算一遍。
    """
    from .animate import hash01

    n = max(0, int(count))
    out = np.zeros((n, DUST_STRIDE), dtype=np.float32)
    for i in range(n):
        out[i, 0] = hash01(seed, i, 1)                                # x0
        out[i, 1] = hash01(seed, i, 2)                                # y0
        out[i, 2] = drift * (0.30 + 0.70 * hash01(seed, i, 3))        # ax
        out[i, 3] = drift * (0.30 + 0.70 * hash01(seed, i, 4))        # ay
        out[i, 4] = 1 + int(hash01(seed, i, 5) * 2.999)               # fx（整数）
        out[i, 5] = 1 + int(hash01(seed, i, 6) * 2.999)               # fy（整数）
        out[i, 6] = TWO_PI * hash01(seed, i, 7)                       # phx
        out[i, 7] = TWO_PI * hash01(seed, i, 8)                       # phy
        out[i, 8] = 1 + int(hash01(seed, i, 9) * 4.999)               # tf（整数）
        out[i, 9] = TWO_PI * hash01(seed, i, 10)                      # tp
        out[i, 10] = 0.35 + 0.65 * hash01(seed, i, 11)                # base
    return out


def dust_fixed_point_scale(count: int) -> int:
    """选一个既够精度、又保证 ``count · scale < 2^32`` 的 2 的幂。

    u32 定点累加是**精确且与顺序无关**的，这是"粒子能逐位复现"的关键
    （浮点原子加的顺序不确定，根本谈不上与 CPU 一致）。

    ⚠️ 但累加会溢出：上界是 ``count · scale``，必须 < 2^32。
    所以 scale 要按 count 反推 —— 粒子越多，标度越小。
    默认 200 颗时取到上限 2^20（每颗量化误差 ≤ 4.8e-7，远小于 1 个 uint8 色阶）；
    count 大到 4096 以上时才需要降档。
    """
    n = max(1, int(count))
    limit = ((1 << 32) - 1) // n      # 需要 count·scale ≤ 2^32-1，差一就溢出
    scale = DUST_MAX_SCALE
    while scale > 1 and scale > limit:
        scale >>= 1
    return max(1, scale)


def dust_splat_uniforms(w: int, h: int, count: int, t: float,
                        twinkle: float, fade_far: float,
                        light_boost: float, light_xy, scale: int,
                        has_far: bool) -> np.ndarray:
    """打包 splat pass 的 uniform（见 dust_splat.wgsl 顶部注释）。"""
    has_light = light_xy is not None
    lx = float(light_xy[0]) if has_light else 0.0
    ly = float(light_xy[1]) if has_light else 0.0
    u = np.zeros((4, 4), dtype=np.float32)
    u[0] = [float(w), float(h), float(count), float(t)]
    u[1] = [float(twinkle), float(fade_far), float(light_boost),
            1.0 if has_light else 0.0]
    u[2] = [lx, ly, float(scale), 1.0 if has_far else 0.0]
    return u.reshape(-1)


def parallax_layout(npix: int) -> dict:
    """分层视差 warp 缓冲的布局（见 parallax_apply.wgsl 顶部注释）。

    单块缓冲：[base 区 3N][far 区 N]，far 区起点**对齐到 64 个 float**。

    ⚠️ 对齐不是洁癖：下游 fog / volumetric / dust 的 ``far`` 绑定要指向同一块
    缓冲的 far 区，而 WebGPU 要求 ``minStorageBufferOffsetAlignment``
    （标准值 256 字节 = 64 float）整除绑定偏移。对齐之后：
      · 一次分配、一次回读；
      · **不需要**"warp 版 / 原版"两套绑定组 —— 管线形状固定，少一整类分支。
    """
    far_off = ((int(npix) * 3 + 63) // 64) * 64
    return {"far_off": far_off, "total": far_off + int(npix)}


def parallax_splat_uniforms(w: int, h: int, k: float, wave: float,
                            pivot_px, drift: float, n_edges: int,
                            near_gain: float, far_gain: float,
                            scale: int) -> np.ndarray:
    """打包 parallax_splat 的 uniform。

    ``k`` 已经是"像素单位的位移基数"（= amp · 0.06 · W · wave），
    这样着色器里只剩乘层系数与归一化坐标 —— **相对幅度与网格无关**这一点
    由调用方保证（见 parallax.parallax_offsets 的标定注释）。
    """
    u = np.zeros((4, 4), dtype=np.float32)
    u[0] = [float(w), float(h), float(k), float(wave)]
    u[1] = [float(pivot_px[0]), float(pivot_px[1]), float(drift),
            float(n_edges)]
    u[2] = [float(near_gain), float(far_gain), float(scale), 0.0]
    return u.reshape(-1)


def parallax_apply_uniforms(w: int, h: int, scale: int, k: float,
                            far_off: int) -> np.ndarray:
    """打包 parallax_apply 的 uniform（``k``=0 时着色器走逐位拷贝分支）。"""
    u = np.zeros((4, 4), dtype=np.float32)
    u[0] = [float(w), float(h), 1.0 / float(scale), float(k)]
    u[1] = [float(far_off), 0.0, 0.0, 0.0]
    return u.reshape(-1)


def dust_apply_uniforms(w: int, h: int, scale: int, dust_bright: float,
                        color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """打包 apply pass 的 uniform。``U[1]`` 是尘埃颜色。"""
    c = np.asarray(color, dtype=np.float32).ravel()
    u = np.zeros((2, 4), dtype=np.float32)
    u[0] = [float(w), float(h), 1.0 / float(scale), float(dust_bright)]
    u[1] = [float(c[0]), float(c[1]), float(c[2]), 0.0]
    return u.reshape(-1)
