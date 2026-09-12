"""场景合成：深度雾 / 光轴 / 辉光。

全部在 **float 全彩空间**完成 —— 量化与抖动必须留到最后一步（arch-01 §8 D4）。
在像素网格分辨率（默认 480x270）上运算，成本很低。

三个效果的定位：

- :func:`depth_fog`  —— **性价比最高的一个**。指数空气透视，直接制造纵深。
- :func:`god_rays`   —— 屏幕空间光轴（径向模糊），做出"丁达尔"的感觉。
- :func:`bloom`      —— 多尺度高斯辉光，让高光"发出来"。

典型顺序：``depth_fog → god_rays → bloom``。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from .animate import flicker_gain, loop_noise_2d

__all__ = [
    "LUMA",
    "BLOOM_RADII",
    "BLOOM_THRESHOLD",
    "BLOOM_KNEE",
    "to_u8",
    "blur",
    "gaussian_kernel",
    "blur_float",
    "bright_pass",
    "depth_equalize",
    "depth_remap",
    "auto_fog_color",
    "depth_fog",
    "god_rays",
    "volumetric_light",
    "bloom",
    "bloom_float",
    "brightest_center",
    "tint",
    "limit_saturation",
    "auto_scatter_color",
]

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: 辉光的三个尺度（半径）与阈值。**唯一定义处**。
#
# ⚠️ 这三个值原先散在 ``pipeline.compose_frame`` 的调用里，而 GPU 侧的 JS
#    还得再抄一份 —— 那就是"同一个概念在多处各写一份"，本项目已经因此
#    栽过 5 次（light_xy / fog_sat / dust 光源 …）。所以在这里定死，
#    CPU 与 GPU 两侧都引用它。
#    验收台（tools/wgsl_lab.py）与 uniform 一致性检查（uniform_parity_probe.py）
#    也引用同一组值，改了这里两边会一起动。
BLOOM_RADII = (2, 5, 11)
BLOOM_THRESHOLD = 0.55
BLOOM_KNEE = 0.25


# ---------------------------------------------------------------- 基础工具
def to_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def blur(a: np.ndarray, radius: float) -> np.ndarray:
    """高斯模糊（借 PIL 的 C 实现）。

    支持 ``(H, W)``、``(H, W, 1)``、``(H, W, 3)`` 三种形状，返回形状与输入一致。
    """
    if radius <= 0:
        return a

    squeeze = a.ndim == 3 and a.shape[2] == 1
    src = a[..., 0] if squeeze else a
    out = np.asarray(
        Image.fromarray(to_u8(src)).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    ) / 255.0

    if a.ndim == 3:
        return out[..., None] if squeeze else out
    return out                         # 原本是 (H,W)


def gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """归一化的一维高斯核，半径 ``ceil(truncate·σ)``。

    ⚠️ 这个核**同时供 CPU 参照实现与 GPU 使用**（GPU 侧由 CPU 算好后上传为
    一段 float32）。为什么要上传而不是在 WGSL 里现算：

        ``exp`` 在 GPU 与 libm 上末位不同 → 同一个 σ 会算出**略有差异的核系数**，
        经 67 个抽头累积后就会被放大，最后体现为"GPU 的模糊和 CPU 的不一样"。
        与其逐位对齐两边的 exp，不如**只算一次**（见 ``docs/arch-02-webgpu.md``）。

    浮点核里的"核"必须精确一致，这类共享常量最容易两边各写一份然后漂移。
    """
    sigma = float(sigma)
    if sigma <= 0:
        return np.ones(1, dtype=np.float32)
    r = int(np.ceil(float(truncate) * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def blur_float(a: np.ndarray, radius: float, truncate: float = 3.0) -> np.ndarray:
    """**浮点**可分离高斯模糊（clamp 边界）。与 :func:`blur` 的区别见下。

    ⚠️ 为什么需要这个函数，而不是继续用 :func:`blur`：

    ``blur`` 走 PIL（``float → uint8 → GaussianBlur → float``），有两处
    **无法搬到 GPU** 的性质：

      1. **中间降到 uint8**。量化误差会被后面的量化/抖动放大，而且**暗部最吃亏**
         —— 辉光恰恰在暗部最可见。这其实是个质量缺陷，不只是移植障碍。
      2. ``ImageFilter.GaussianBlur`` 是**三次盒式模糊的近似**，盒宽由 Pillow
         版本决定 → 逐位复现既不现实（换版本就崩），也不值得。

    实测两者的差距（``tools/blur_operator_probe.py``）：单算子上差 1.5~3 个 uint8
    色阶，但经过**完整 bloom + 色板吸附**之后只剩 **13/6144 = 0.21%** 的像素不同。
    色板吸附本身就是很好的误差吸收器。

    所以 GPU 路径用这个真高斯，并以此函数作为它的验收参照。
    ``bloom``（PIL 版）保持不变 —— 服务端渲染路径不因为移植而改变行为。
    """
    if radius <= 0:
        return a
    k = gaussian_kernel(radius, truncate)
    r = (len(k) - 1) // 2

    arr = a.astype(np.float64)
    squeeze = arr.ndim == 3 and arr.shape[2] == 1
    if squeeze:
        arr = arr[..., 0]

    for axis in (1, 0):
        widths = [(0, 0)] * arr.ndim
        widths[axis] = (r, r)
        # ⚠️ 边界用**边缘延展**（clamp，即 np.pad 的 'edge'）。
        #    和 GPU 侧保持一致 —— 边界处理不一样，边缘一圈就对不上，
        #    而验收台会把它当成"移植误差"报出来。
        pad = np.pad(arr, widths, mode="edge")
        out = np.zeros_like(arr)
        for i, kv in enumerate(k):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + arr.shape[axis])
            out += pad[tuple(sl)] * float(kv)
        arr = out

    res = arr.astype(np.float32)
    return res[..., None] if squeeze else res


def bloom_float(
    rgb01: np.ndarray,
    threshold: float = 0.55,
    strength: float = 0.75,
    radii=(3.0, 7.0, 15.0),
    knee: float = 0.25,
    truncate: float = 3.0,
) -> np.ndarray:
    """:func:`bloom` 的**浮点**版本 —— GPU 路径的参照实现。

    公式与 :func:`bloom` 完全一致（高光提取 → 各尺度模糊按 1/(i+1) 加权 →
    除以权重和 → 乘 strength → 加回原图），只是把 :func:`blur` 换成
    :func:`blur_float`。理由见 :func:`blur_float`。
    """
    bright = bright_pass(rgb01, threshold, knee=knee)
    acc = np.zeros_like(rgb01)
    wsum = 0.0
    for i, r in enumerate(radii):
        w = 1.0 / (i + 1)
        acc += blur_float(bright, r, truncate) * w
        wsum += w
    return np.clip(rgb01 + acc / max(wsum, 1e-6) * strength, 0.0, 1.0)


def bright_pass(rgb01: np.ndarray, threshold: float = 0.60, knee: float = 0.0) -> np.ndarray:
    """提取高光区域。``knee`` > 0 时用软阈值，过渡更自然。

    ⚠️ 硬阈值与软阈值是**两个不同公式**：
    硬的是 ``mask = t``，软的是 ``clip(t²/(t+knee)/(1+knee), 0, 1)``。
    移植到 WGSL 时若只写软的那支，``knee=0`` 会退化成 ``t²/(t+1e-6)`` ——
    与 ``t`` 差得极远（实测最大差 1.7e-2、89% 的像素越界）。
    当前调用方都传 knee>0，所以那是个**潜伏** bug，只有数值比对能提前发现。
    """
    lum = rgb01 @ LUMA
    denom = max(1e-6, 1.0 - threshold)
    if knee > 0:
        t = np.clip((lum - threshold) / denom, 0.0, 1.0)
        mask = t * t / (t + knee + 1e-6)          # 软化
        mask = np.clip(mask / (1.0 + knee), 0.0, 1.0)
    else:
        mask = np.clip((lum - threshold) / denom, 0.0, 1.0)
    return rgb01 * mask[..., None]


def tint(rgb01: np.ndarray, color, amount: float) -> np.ndarray:
    """整体向某个色偏靠拢（用于统一冷/暖调）。"""
    c = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb01 * (1.0 - amount) + c * amount, 0.0, 1.0)


def limit_saturation(color, sat_max: float = 0.25) -> np.ndarray:
    """把颜色往它的均值（中性）方向压，使线性饱和度不超过 ``sat_max``。

    ⚠️ 这不是"调味"，是**物理约束**。体积光的散射色含义是
    "**空气**被照亮后的颜色"，而空气是弱散射体、近乎无色 ——
    所以散射色**永远不该比光源本身更饱和**。给一幅图叠上饱和色光，
    观感上不是"有雾"，而是"整幅图偏色了"。

    实现上按 ``mean + (c - mean) * k`` 缩放色度：保持平均亮度不变，
    只把"离中性多远"压下来。饱和度和整体缩放无关，所以要迭代几次逼近。
    """
    c = np.clip(np.asarray(color, dtype=np.float32).ravel(), 0.0, 1.0).copy()
    for _ in range(4):
        lo, hi = float(c.min()), float(c.max())
        if hi <= 1e-6:
            return np.full(3, 0.5, dtype=np.float32)
        sat = (hi - lo) / hi
        if sat <= sat_max:
            break
        c = np.clip(float(c.mean()) + (c - float(c.mean())) * (sat_max / sat), 0.0, 1.0)
    return c.astype(np.float32)


def auto_scatter_color(
    rgb01: np.ndarray,
    px: int,
    py: int,
    radius: int = 3,
    sat_max: float = 0.25,
) -> np.ndarray:
    """从光源附近估一个**低饱和**的体积散射色。

    两步：

    1. **取窗口的中位数**（不是均值）。中位数对单个炸白的高光点免疫；
       均值会被它带跑 —— 雾色就是这样踩过坑的（约束 11）。
       ⚠️ 不要改成"只取较亮的那半像素再取中位数"：实测那样会
       **丢掉约 25% 的暖色**（ref02 的暖度 R−B 从 +0.23 掉到 +0.11），
       因为暖光场景里最亮的那批像素往往是**炸白的高光**，色相已经丢了。
       中位数取全窗口既稳又保色。
    2. **压饱和度**（:func:`limit_saturation`）。

    ⚠️ 第 2 条才是修 bug 的关键，不是锦上添花。实测事故：`ref10_green_cliff`
    上光源中心落在**草丛**里，窗口 7×7 **全是绿色像素** ——
    无论怎么选像素、取什么统计量都救不回来（实测 5 种采样变体的偏绿指数
    全是 +0.182，一模一样），**只有饱和度上限能把它拉回中性**。
    旧代码直接取均值，得到 (0.618, 0.793, 0.392) 的饱和黄绿，
    叠上去等于给整幅图加了一层绿光（第二个"变绿"来源）。

    Args:
        px / py: 光源的**像素**坐标（不是归一化值）。
        radius: 取样窗口半径。
        sat_max: 散射色的线性饱和度上限。**实测标定为 0.25**：
            10 张素材里其余 9 张的自然散射色饱和度都 <= 0.25，所以这个上限
            **只对"光源定位错了"的病态情况生效**，不会误伤暖色光场景
            （0.15 会把 ref02 的暖度从 +0.232 抽到 +0.130，不能用）。
    """
    h, w = rgb01.shape[:2]
    y0, y1 = max(0, int(py) - radius), min(h, int(py) + radius + 1)
    x0, x1 = max(0, int(px) - radius), min(w, int(px) + radius + 1)
    patch = rgb01[y0:y1, x0:x1].reshape(-1, 3)
    if patch.shape[0] == 0:
        return np.full(3, 0.7, dtype=np.float32)
    return limit_saturation(np.median(patch, axis=0), sat_max)


# ------------------------------------------------------------------ 深度雾
def depth_equalize(far01: np.ndarray, strength: float = 0.85,
                   clip: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """用经验 CDF 把深度拉成近似均匀分布 —— 也就是**改用「排名」而不是「原始数值」**。

    ⚠️ **这一步比 :func:`depth_remap` 更关键。**

    单目深度模型可靠的是**排序**，不是**度量**。它的原始输出分布常常极度偏斜：
    实测 REPLACE 那张的 5/25/50/75/95 分位是 0.12 / 0.43 / 0.86 / 0.92 / 0.94 ——
    一半以上的像素挤在 0.86~0.95 这一小段里。

    线性拉伸救不了这种分布（它只改端点，不改形状）。结果就是指数雾几乎不随位置变化：
    要么整幅被雾吞掉，要么几乎没雾。按排名重映射后，每一层才真正分到合理的像素比例。

    Args:
        strength: 1.0 = 完全按排名；0.0 = 保持原值。0.7~0.9 通常最自然。
        clip: 先按分位裁掉极端值，避免个别离群点 dominate 排名。
    """
    x = np.clip(far01, *np.percentile(far01, clip)).astype(np.float32)
    flat = x.ravel()
    order = np.argsort(flat, kind="stable")
    rank = np.empty(flat.size, dtype=np.float32)
    rank[order] = np.arange(flat.size, dtype=np.float32) / max(1, flat.size - 1)
    eq = rank.reshape(far01.shape)
    return ((1.0 - strength) * x + strength * eq).astype(np.float32)


def depth_remap(far01: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """按分位数把深度线性重映射到 [0, 1]（只改端点，不改分布形状）。

    需要保留层间相对间距、且分布本身不太偏斜时用这个；
    分布明显偏斜时用 :func:`depth_equalize` 更有效。
    """
    a, b = np.percentile(far01, [lo, hi])
    return np.clip((far01 - a) / max(1e-6, b - a), 0.0, 1.0).astype(np.float32)


def auto_fog_color(
    rgb01: np.ndarray,
    far01: np.ndarray,
    q: float = 0.85,
    tint=None,
    tint_mix: float = 0.0,
    lift: float = 0.04,
    desat: float = 0.25,
    sat_max: float = 0.40,
) -> np.ndarray:
    """从画面自身的**远景区域**估一个雾色。

    空气透视的物理本质是：远处被环境散射光逐渐替换。所以雾色应当接近
    "远处本来是什么颜色"，而不是拍一个常数。这样同一套参数能跨图工作。

    ⚠️ 三个细节都是踩坑得来的：

    - 用**中位数**而不是均值。远景里常有一块很亮的高光/光轴（例如那根灯柱），
      均值会被它拉高，雾色变得又亮又暖，一叠上去整幅糊成褐色。
    - 适度**降饱和**（``desat``）。真实散射会带走一点色彩，不降饱和的雾色
      会显得像蒙了层颜料。
    - ⭐ **还要再加一道饱和度上限**（``sat_max``）。只靠 ``desat=0.25`` 远远不够：
      实测 10 张素材里有 **7 张**的自动雾色饱和度超过 0.15，
      `ref10_green_cliff` 甚至到 **0.789**（它拿一片饱和蓝天当了雾色）。
      而雾会覆盖**全部远景** —— 饱和雾色叠上去就是"给远景蒙一层有色滤镜"，
      用户可见症状是"画面像被往某个主题色上拉"。
      标定为 **0.40**：只对上面那 4 张病态样本生效，其余 6 张完全不碰，
      同时保留 76% 的暖度（0.20 会把暖色场景的雾洗成灰，不能用）。

    Args:
        lift: 抬高亮度 —— 雾是散射光，通常比被它覆盖的物体略亮。
        desat: 向该色的灰度值靠拢的比例。
        sat_max: 线性饱和度的**硬上限**。
    """
    m = far01 >= np.quantile(far01, q)
    c = np.median(rgb01[m], axis=0) if m.any() else np.median(rgb01.reshape(-1, 3), axis=0)
    c = c.astype(np.float32)
    if desat > 0:
        c = c * (1.0 - desat) + float(c.mean()) * desat
    if tint is not None and tint_mix > 0:
        c = c * (1.0 - tint_mix) + np.asarray(tint, dtype=np.float32) * tint_mix
    c = np.clip(c, 0.0, 1.0)
    if sat_max is not None:
        c = limit_saturation(c, sat_max)
    return np.clip(c + lift, 0.0, 1.0)


def depth_fog(
    rgb01: np.ndarray,
    far01: np.ndarray,
    color=None,
    density: float = 1.4,
    power: float = 3.0,
    floor: float = 0.0,
    ceiling: float = 1.0,
    t: float = 0.0,
    drift: float = 0.0,
    fog_sat: float = 0.40,
    fog_tint: float = 1.0,
) -> np.ndarray:
    """指数空气透视（aerial perspective）。

    ``t = 1 - exp(-density * z**power)``：近处 t→0 保留原色，远处 t→1 被雾色吞没。

    ``power`` 很关键：``power=1`` 时雾是均匀铺开的，中景就开始发灰，画面会"糊"。
    取 2.5~3.5 可以让**中景保持通透，只在远端发力** —— 这才是空气透视该有的样子。
    经验曲线：power=3, density=1.4 时 t(0.2)=0.01、t(0.5)=0.16、t(0.8)=0.51、t(1.0)=0.75。

    Args:
        far01: (H, W) 深度，0 = 近、1 = 远。
        color: 雾色；``None`` 时用 :func:`auto_fog_color` 自动估。
        density: 越大雾越浓。
        power: 距离的幂次。>1 把雾推向远端。
        floor / ceiling: 把 t 重映射到 [floor, ceiling]。
        t: 时间，[0, 1) 为一个循环。雾开始"呼吸"（见 ``drift``）。
        drift: 雾的呼吸幅度。0 = 完全静止（静态图必须用 0）。
            >0 时按 :func:`animate.loop_noise_2d` 得到一个**严格循环**的
            空间场，让雾浓淡缓慢漂移 —— 这样即使在循环首尾相接处也无缝。
        fog_sat: 自动雾色的饱和度上限（``color`` 显式给定时不生效）。
            见 :func:`auto_fog_color` —— 雾覆盖全部远景，饱和雾色 = 全局滤镜。
    fog_tint: **雾的上色强度**（M5.3）。1 = 现状（雾色完全上色，整幅图
            被拉向雾色）；0 = 雾色去色为**中性灰**（保留远近明暗空气透视、
            但不改变任何色相 —— 色彩层次全保留）。中间值线性过渡。
            ⚠️ 用户明确要"保留色彩层次丰富"，这个旋钮就是为它加的。
    """
    z = np.clip(far01, 0.0, 1.0).astype(np.float32)
    if color is None:
        color = auto_fog_color(rgb01, z, sat_max=fog_sat)

    d = float(max(density, 1e-6))
    if drift > 0:
        # 严格循环的漂移场：整体浓度升降 ~ 空间上浓淡不均，
        # 后者才是"雾在动"的观感来源（局部浓淡比整体明暗更像空气）。
        n = loop_noise_2d(t, z.shape, octaves=2, seed=17)
        d = d * np.clip(1.0 + drift * (0.6 * n + 0.4 * float(n.mean())), 0.05, None)

    t_fog = 1.0 - np.exp(-d * np.power(z, max(power, 1e-6)))
    t_fog = np.clip((t_fog - floor) / max(1e-6, ceiling - floor), 0.0, 1.0)
    fog = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    # ── 雾的上色强度（M5.3）── 0 = 雾色去色为中性灰（保色相），1 = 原雾色。
    # ⚠️ fog_tint >= 0.999 时**直接用原雾色**（不过这段数学）——
    #    "默认行为逐位不变"。GPU 侧同样实现（fog.wgsl 的 fog_col 段）。
    if fog_tint < 0.999:
        lum = float(fog[0, 0, 0] * 0.2126 + fog[0, 0, 1] * 0.7152
                    + fog[0, 0, 2] * 0.0722)
        fog = np.full_like(fog, lum) + (fog - np.full_like(fog, lum)) * np.float32(fog_tint)
    return np.clip(rgb01 * (1.0 - t_fog[..., None]) + fog * t_fog[..., None], 0.0, 1.0)


# ------------------------------------------------------------------ 光轴
def god_rays(
    bright01: np.ndarray,
    center_xy: tuple[float, float],
    density: float = 0.65,
    decay: float = 0.965,
    samples: int = 24,
    strength: float = 0.55,
    mask01: np.ndarray | None = None,
) -> np.ndarray:
    """屏幕空间光轴（径向模糊），返回**加性**的光轴图层。

    ⚠️ 这是**不感知深度**的旧版：它只做径向模糊，光会穿墙，
    而且叠出来的往往只是"光源周围一圈白"（那其实是 bloom，不是体积光）。
    有深度图时请改用 :func:`volumetric_light`。

    Args:
        bright01: (H,W,3) 高光图（通常来自 :func:`bright_pass`）。
        center_xy: 光源位置，归一化 (x, y)。
        density: 采样跨度，占图宽的比例。
        decay: 每次采样的能量衰减。
        strength: 最终叠加强度。
    """
    h, w = bright01.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = center_xy
    dx = (cx * w - xx) / w
    dy = (cy * h - yy) / h

    acc = np.zeros_like(bright01)
    illum = 1.0
    for i in range(1, int(samples) + 1):
        s = i / samples * density
        sx = np.clip((xx + dx * w * s).astype(np.int32), 0, w - 1)
        sy = np.clip((yy + dy * h * s).astype(np.int32), 0, h - 1)
        acc += bright01[sy, sx] * illum
        illum *= decay

    acc /= max(1, int(samples))
    if mask01 is not None:
        acc *= np.clip(mask01, 0.0, 1.0)[..., None]
    return np.clip(acc * strength, 0.0, 1.0)


@lru_cache(maxsize=8)
def _ray_offsets(h: int, w: int, lx: int, ly: int, samples: int, span: float) -> tuple:
    """预计算体积光沿光线采样的**扁平像素下标**。

    ⚠️ 这些下标只取决于几何（分辨率、光源位置、采样数、跨度），
    **与画面内容无关** —— 所以在视频逐帧渲染时它们是完全不变的，
    缓存起来能省掉每帧 28 次 ``astype(int32)`` + ``clip`` 的开销。

    ``maxsize=8``：正常只会用到一两个（一个场景 + 一次参数扫描），
    每个条目约 28 × H×W × 4 字节 ≈ 14 MB（480×254），不会失控。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (lx - xx) / w
    dy = (ly - yy) / h
    out = []
    for i in range(1, int(samples) + 1):
        s = i / samples * span
        sx = np.clip((xx + dx * w * s).astype(np.int32), 0, w - 1)
        sy = np.clip((yy + dy * h * s).astype(np.int32), 0, h - 1)
        out.append((sy * w + sx).reshape(-1))
    return tuple(out)


def cone_weight(w: int, h: int, lx: int, ly: int,
                cone_angle: float, cone_dir_deg: float,
                cone_reach: float, cone_gain: float = 1.0) -> np.ndarray:
    """光锥的角度权重 (H, W)：锥内 1、锥外 0、边缘平滑过渡。

    ⚠️ 数学必须与 ``tools/wgsl/volumetric.wgsl`` 里那段**一字不差** ——
    否则 CPU 与服务端渲染会分叉（这是第 N 次同类问题，所以这里写成独立函数，
    GPU 侧照抄同一串运算）。

    定义（屏幕空间，等距化之后）：
      · 把像素相对光源的偏移换算到**等距**坐标：x 乘 (w/h)、y 不乘 ——
        否则宽画面上同一个角度会被拉扁（圆形的锥会变成椭圆）。
      · 方向向量 ``axis = (cos θ, sin θ)``，θ 由**度**转弧度。
      · ``cosang = dot(unit(v), axis)``；锥半角 ``θ/2`` 内为 1、``θ`` 外为 0，
        中间用 smoothstep 过渡（``t²(3−2t)``）。
        ⚠️ ``|v| ≈ 0``（光源自身那个像素）方向未定义 → 显式视为在锥内。
      · 沿轴的距离衰减：``1/(1 + (1−reach)·3·|v|)`` —— reach=1 几乎不衰减，
        reach=0 时光柱在画面内就淡掉（"打不到底"的观感）。

    ``cone_angle <= 0`` 时返回全 1（= 关闭，调用方连乘都不该做）。
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 等距化：y 用 (h-1) 归一，x 乘 (w/h) 换算到同一尺度
    vx = (xx - lx) / max(h - 1, 1) * (float(w) / max(h, 1))
    vy = (yy - ly) / max(h - 1, 1)
    vlen = np.sqrt(vx * vx + vy * vy)
    inv = 1.0 / np.maximum(vlen, 1e-6)
    ux, uy = vx * inv, vy * inv
    th = np.radians(cone_dir_deg)
    dot = ux * np.float32(np.cos(th)) + uy * np.float32(np.sin(th))
    # ⚠️ 光源自身那个像素 vlen≈0，方向是**未定义**的 —— 实测它会算出 cosang=0
    #    从而被判到锥外，于是"光源自己"是暗的（一眼假的缺陷）。
    #    这里显式视为在锥内（光源当然最亮）。GPU 侧有同样一行。
    cosang = np.where(vlen > 1e-6, dot, np.float32(1.0))
    outer = float(np.cos(cone_angle))
    inner = float(np.cos(cone_angle * 0.5))
    tt = np.clip((cosang - outer) / max(inner - outer, 1e-6), 0.0, 1.0)
    wgt = tt * tt * (3.0 - 2.0 * tt)
    if cone_reach < 1.0:
        wgt = wgt / (1.0 + (1.0 - float(cone_reach)) * 3.0 * vlen)
    # ⚠️ **锥内增益**：单纯"锥外减掉"几乎看不出效果 —— 实测开/关只差 1.8/255，
    #    因为体积光本身的注入量就小，减掉锥外那些本来就很弱的能量等于没减。
    #    "光柱"要的是**锥内的空气比锥外亮**，所以必须给锥内一个乘性增益。
    if cone_gain != 1.0:
        wgt = wgt * float(cone_gain)
    return wgt.astype(np.float32)


def volumetric_light(
    rgb01: np.ndarray,
    far01: np.ndarray,
    light_xy: tuple[float, float],
    samples: int = 28,
    span: float = 0.85,
    decay: float = 0.965,
    strength: float = 0.85,
    air_color=None,
    occlude_gain: float = 5.0,
    falloff_gain: float = 2.5,
    air_sat_max: float = 0.25,
    t: float = 0.0,
    flicker: float = 0.0,
    screen_falloff: float = 1.0,
    cone_angle: float = 0.0,
    cone_dir_deg: float = 80.0,
    cone_reach: float = 0.8,
    cone_gain: float = 1.6,
    shaft: float = 0.0,
    fog_tint: float = 1.0,
    cone_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """**深度感知**的体积光散射。返回加性图层（已带散射色，直接加到画面上）。

    和 :func:`god_rays` 的区别（这也是"光源周围一圈白"和"体积雾"的差别）：

    1. **遮挡**：沿光线采样时，用深度判断采样点是否比光源近很多 ——
       近很多的物体挡在前面，光应该被挡住而不是穿过去。没有这一步，
       光会穿墙，看起来就是一团糊。
    2. **按场景距离衰减**：衰减用 ``|z_pixel - z_light|``（场景里的距离），
       不是屏幕上的像素距离。
    3. **散射带颜色**：体积光照亮的是空气，所以叠上去的应该是**光源的颜色**，
       不是白色。白色只会得到"一圈白"。
       ⚠️ 但散射色必须**低饱和** —— 见 :func:`auto_scatter_color`。
       实测 `ref10` 上光源中心落在草丛里，取样得到饱和黄绿，
       叠上去就是整幅图变绿。

    Args:
        far01: (H, W) 深度，0 = 近、1 = 远。
        light_xy: 光源位置，归一化 (x, y)。
        air_color: 散射色；``None`` 时走 :func:`auto_scatter_color` 自动估。
        occlude_gain: 遮挡强度。越大，挡在前面的物体遮得越死。
        falloff_gain: 随场景距离的衰减速度。
        air_sat_max: 自动散射色的线性饱和度上限（``air_color`` 显式给定时不生效）。
        t: 时间，[0, 1) 为一个循环。
        flicker: 光源闪烁幅度。0 = 完全静止（静态图必须用 0）。
            >0 时强度按 :func:`animate.flicker_gain` 起伏（整数频率正弦叠加），
            严格以 1 为周期 —— 循环首尾无缝。
        screen_falloff: **屏幕空间**衰减指数，0 = 关掉（旧行为）。

            ⚠️ 这是一个**结构性**的修正，不是调味。

            原来的衰减是 ``1/(1 + gain·|z − z_light|·4)`` —— 只按**场景深度差**
            衰减，**完全不看屏幕距离**。后果：只要画面里大部分像素与光源深度相近，
            它们就都会均匀地吃到同一层带色的光。实测 ref10 上：
            屏幕距离 0.5 以外的区域（占画面 58%）仍贡献了 **21%** 的注入量，
            观感就是"整幅图被往某个色相上拉，像蒙了一层滤镜"（用户报障）。

            加上屏幕空间衰减后，近处（<0.25）占比从 53% 提到 **69%**，
            全图注入量减半 —— 光更像"从那个位置发出来的"，而不是全局染色。

            取 1.0 是比较克制的；2.0 会更"聚"（占比 79%）但光晕也会变小。
    """
    h, w = rgb01.shape[:2]
    bright = bright_pass(rgb01, 0.48, knee=0.3)
    # ⚠️ 先降到亮度再进循环。原来在循环里对 (H,W,3) 做 .mean(axis=2)，
    #    28 次采样就白做了 28 遍归约（实测 33 ms/帧，占本函数四分之一）。
    bright_lum = bright @ LUMA

    lx, ly = int(np.clip(light_xy[0], 0, 1) * (w - 1)), int(np.clip(light_xy[1], 0, 1) * (h - 1))
    z_light = float(far01[ly, lx])

    # ── 锥顶点（M5.4）：可与光源解耦 ──
    #   cone_xy=None（默认）= 跟随光源（数值与旧行为逐位一致）；
    #   给定归一化坐标 = 锥形与光柱的几何锚点，**允许出画**（不 clip ——
    #   如太阳在画外的穿窗光束）。遮挡、深度衰减、散射色采样仍以**光源**为准。
    #   ⚠️ 解析舍入必须与两侧 packer 同一字：floor(x·(w−1)+0.5)。
    #      用 int() 截断会差 1px（numpy 228 vs packer 229，探针已抓到）。
    if cone_xy is not None:
        ax = int(cone_xy[0] * (w - 1) + 0.5)
        ay = int(cone_xy[1] * (h - 1) + 0.5)
    else:
        ax, ay = lx, ly

    if air_color is None:
        air = auto_scatter_color(rgb01, lx, ly, radius=3, sat_max=air_sat_max)
    else:
        air = np.clip(np.asarray(air_color, dtype=np.float32).ravel(), 0.0, 1.0)
    # ── 散射色同样受 fog_tint：光柱的"上色"也归这个旋钮管 ──
    if fog_tint < 0.999:
        la = float(air[0] * 0.2126 + air[1] * 0.7152 + air[2] * 0.0722)
        air = np.float32([la, la, la]) + (air - np.float32([la, la, la])) * np.float32(fog_tint)

    far_flat = far01.reshape(-1)
    bright_flat = bright_lum.reshape(-1)
    # 采样点的**像素下标**只跟几何有关（h/w/光源位置/采样数/跨度），与画面内容无关，
    # 所以缓存起来跨帧复用 —— 这是"每帧"路径上最值得省的一笔。
    offsets = _ray_offsets(h, w, lx, ly, int(samples), float(span))

    acc = np.zeros(h * w, dtype=np.float32)
    # 屏幕空间阴影量：沿"像素→光源"的射线，若某采样点比**本像素**更近，
    # 说明有东西挡在中间 → 这一格处于阴影里（光柱被切断）。
    shade = np.zeros(h * w, dtype=np.float32)
    illum = 1.0
    for off in offsets:
        zq = far_flat[off]
        # 遮挡：采样点比光源近很多 → 它在光源前面，挡住光
        occl = np.exp(-occlude_gain * np.clip(z_light - zq, 0.0, None))
        acc += bright_flat[off] * (illum * occl)
        shade = np.maximum(shade, far_flat - zq - 0.02)
        illum *= decay
    acc = (acc / max(1, len(offsets))).reshape(h, w)

    # 距离衰减按**场景深度差**算，而不是屏幕像素距离
    dist = np.abs(far01 - z_light)
    falloff = 1.0 / (1.0 + falloff_gain * dist * 4.0)

    # 光源闪烁：乘性增益，严格以 1 为周期（互质整数频率正弦叠加）
    gain = flicker_gain(t, depth=flicker) if flicker > 0 else 1.0

    out = acc * falloff

    # 屏幕空间衰减：让光"属于那个位置"，而不是均匀铺满全图
    if screen_falloff > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d_screen = np.hypot((xx / max(w, 1)) - lx / max(w, 1),
                            (yy / max(h, 1)) - ly / max(h, 1))
        out = out / (1.0 + (d_screen * 4.0) ** screen_falloff)

    # ── 光锥：把"径向均匀的光晕"塑成形（直边 + 锥内加权）──
    # ⚠️ cone_angle <= 0 时**连乘法都不做** —— 这是"默认行为逐位不变"的保证
    #    （乘 1.0 也会让浮点结果在最后一位上分叉）。
    if cone_angle > 0:
        cw = cone_weight(w, h, ax, ay, cone_angle, cone_dir_deg,
                         cone_reach, cone_gain)
        out = out * cw

        # ── ⭐ 光柱：**独立加性层**（M5.2）──
        #
        # ⚠️⚠️ 为什么不能只乘权重：实测（ref01/04/10）体积光整层只有
        #     **1~3/255** —— `falloff`（按场景深度差）与 `screen_falloff`
        #     把能量压掉了两个数量级（当年为了治"全局滤镜感"刻意调弱的）。
        #     给这么弱的层塑形，塑不出可见的光柱（几何对，但看不出来）。
        #
        # 做法（★ 这里是第二次修正）：光柱**不能**乘 ``acc`` ——
        # 实测那样做锥内平均只有 2/255，因为 ``acc`` 是"沿射线能否看见**光源**"
        # （点源可见性）：锥内大多数像素的射线并没打到那个亮点。
        # 光柱要表达的是"**锥内的空气被照亮**"，所以亮度沿光柱基本均匀、
        # 只随距离衰减，再被**屏幕空间阴影**切断（有东西挡在中间就暗）。
        if shaft > 0:
            # 沿轴衰减：近光源亮、远处淡（锚点 = 锥顶点，解耦时与光源不同）
            axial = 1.0 / (1.0 + 1.6 * np.hypot(
                (np.mgrid[0:h, 0:w][1].astype(np.float32) - ax) / max(h - 1, 1),
                (np.mgrid[0:h, 0:w][0].astype(np.float32) - ay) / max(h - 1, 1)))
            beam = cw * axial * np.exp(-4.0 * np.clip(shade, 0.0, None)).reshape(h, w)
            out = out + beam * float(shaft)

    return np.clip(out * strength * gain, 0.0, 1.0)[..., None] * air[None, None, :]


def brightest_center(rgb01: np.ndarray, blur_radius: float = 8.0) -> tuple[float, float]:
    """找出画面最亮区域的重心，作为光轴中心。返回归一化 (x, y)。"""
    lum = blur((rgb01 @ LUMA)[..., None], blur_radius)[..., 0]
    idx = int(np.argmax(lum))
    h, w = lum.shape
    return ((idx % w) + 0.5) / w, ((idx // w) + 0.5) / h


# ------------------------------------------------------------------ 辉光
def bloom(
    rgb01: np.ndarray,
    threshold: float = 0.55,
    strength: float = 0.75,
    radii=(3.0, 7.0, 15.0),
) -> np.ndarray:
    """多尺度高斯辉光。

    先做高光提取，再在几个不同半径上模糊并加权相加 —— 小半径给出锐利的边缘光，
    大半径给出弥散的空气感。这正是"贵"的来源之一，而且在像素化之后依然有效。
    """
    bright = bright_pass(rgb01, threshold, knee=0.25)
    acc = np.zeros_like(rgb01)
    wsum = 0.0
    for i, r in enumerate(radii):
        w = 1.0 / (i + 1)
        acc += blur(bright, r) * w
        wsum += w
    return np.clip(rgb01 + acc / max(wsum, 1e-6) * strength, 0.0, 1.0)
