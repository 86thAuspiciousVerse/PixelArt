"""降采样与影调映射 —— 像素尾巴的前两步。

朴素缩放（nearest / 单纯 lanczos）是"廉价滤镜感"的头号来源：
nearest 会把细节碎成噪点，lanczos 会产生振铃并糊掉细线。
结构感知降采样用局部方差在两者之间加权，既保轮廓又不引入噪点。
"""

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "structure_aware_downsample",
    "auto_levels",
    "unsharp",
    "tone_map",
    "fit_to_aspect",
    "fit_native",
    "upscale_nearest",
]


def structure_aware_downsample(
    img: Image.Image,
    size: tuple[int, int],
    var_gain: float = 45.0,
    detail: float = 0.0,
) -> np.ndarray:
    """把图降到 ``size`` (W, H)，返回 (H, W, 3) float32 in [0, 1]。

    做法：``box``（区域均值）与 ``lanczos``（保局部细节）按**局部方差**加权混合。
    方差大 = 边缘/纹理 → 偏向 lanczos 保住轮廓；方差小 = 平坦区 → 用均值避免噪点。

    这是 Gerstner 2012《Pixelated Image Abstraction》的廉价近似。
    升级路径见 docs/research-01-prior-art.md §2。

    Args:
        detail: **细节预算**（M5.8，0~1，默认 0 = 原行为逐位不变）。
            低网格下均匀降采样会把细线/五官抹掉——均值对"深底上的 1px 亮线"
            的答案是"灰"。detail > 0 时，**忙碌格子不再取均值，而是偏向格内
            距均值最远的"代表性细节像素"**（那条亮线自己），偏离越大偏得越狠。
            平坦格子不受影响（本来就该是均值）。预算 spent 在这里才有意义：
            降采样时丢掉的细节，事后的锐化救不回来。
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("size 必须为正")

    a = np.asarray(img, dtype=np.float32) / 255.0
    sq = Image.fromarray(np.clip(a * a * 255.0, 0.0, 255.0).astype(np.uint8))

    box = np.asarray(img.resize(size, Image.Resampling.BOX), dtype=np.float32) / 255.0
    sqb = np.asarray(sq.resize(size, Image.Resampling.BOX), dtype=np.float32) / 255.0
    lan = np.asarray(img.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    var = np.clip(sqb - box * box, 0.0, None).mean(axis=2, keepdims=True)
    wgt = np.clip(var * var_gain, 0.0, 1.0)
    if detail <= 0.0:
        return (box * (1.0 - wgt) + lan * wgt).astype(np.float32)

    # ── 细节预算（M5.8）：忙碌格子偏向"代表性细节像素" ──
    #   work 分辨率恒为 grid × 整数倍（prepare_scene 的约定），可精确分格。
    H, W = a.shape[:2]
    sy, sx = max(1, H // h), max(1, W // w)
    a_c = a[: h * sy, : w * sx]
    # 每格内像素到**格均值**的偏差平方（对"细线/点"最敏感——它们偏离均值最远）
    dev = ((a_c.reshape(h, sy, w, sx, 3) - box.reshape(h, 1, w, 1, 3)) ** 2).sum(-1)
    dev2 = dev.transpose(0, 2, 1, 3).reshape(h, w, sy * sx)             # (h, w, 格内像素)
    am = dev2.argmax(axis=2)                                            # 每格最偏像素
    iy, ix = am // sx, am % sx
    Y = np.arange(h)[:, None] * sy + iy
    X = np.arange(w)[None, :] * sx + ix
    detailpix = a_c[Y, X]                                               # (h, w, 3)
    # 偏差归一化（98 分位定标，防止极端噪点绑架刻度）
    dev_max = dev2.max(axis=2)                                          # (h, w)
    dn = np.clip(dev_max / max(float(np.percentile(dev_max, 98)), 1e-6), 0.0, 1.0)[..., None]
    w2 = np.clip(wgt + float(detail) * (1.0 - wgt) * dn, 0.0, 1.0)
    return (box * (1.0 - w2) + detailpix * w2).astype(np.float32)


def auto_levels(
    x: np.ndarray,
    clip: tuple[float, float] = (1.0, 99.0),
    strength: float = 1.0,
    target_span: float = 0.92,
) -> np.ndarray:
    """按分位数自动重设黑白场（直方图拉伸），**按需触发**。

    ⚠️ 这一步对"糊"的观感影响极大，但**绝不能无条件全开**。

    - 低对比素材（夜间动画、雾天、阴天照片）的直方图只占中间一小段，
      这时 median cut 会把宝贵的色阶大量浪费在**几乎相同的几个暗部颜色**上，
      量化后就变得又灰又糊。拉伸之后色阶才花在真正有差异的地方。
    - 但**本身动态范围就够**的素材（干净的写实图、已经调过色的图）
      再拉伸只会过饱和、丢层次 —— 实测一张绿色悬崖图被拉成荧光绿。

    所以这里会先量一下现有动态范围：已经接近 ``target_span`` 就自动不拉。

    Args:
        strength: 上限强度。
        clip: 用于估现有范围的分位数。
        target_span: 期望的黑白场跨度；现有跨度已 ≥ 此值则不再拉伸。
    """
    lo, hi = np.percentile(x, clip)
    span = float(hi - lo)
    need = float(np.clip((target_span - span) / max(1e-6, target_span), 0.0, 1.0))
    s = float(np.clip(strength, 0.0, 1.0)) * need
    if s <= 0.0:
        return x.astype(np.float32, copy=True)
    y = np.clip((x - lo) / max(1e-6, float(hi - lo)), 0.0, 1.0)
    return (x * (1.0 - s) + y * s).astype(np.float32)


def unsharp(x: np.ndarray, amount: float = 0.55, radius: float = 1.0) -> np.ndarray:
    """在**像素网格分辨率**上做局部对比增强（USM）。

    像素画"利不利"，很大程度取决于方块之间有没有硬朗的明暗差。
    降采样天然会削弱相邻方块的反差（即使是结构感知降采样也一样），
    在量化前补一次轻度 USM，方块边缘会明显"立"起来。

    ⚠️ 要在量化**之前**做：让色板本身也反映增强后的对比。
    过度增强会在边缘产生振铃，``amount`` 建议不超过 0.8。
    """
    if amount <= 0:
        return x
    img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    low = np.asarray(img.filter(ImageFilter.GaussianBlur(max(radius, 0.3))),
                     dtype=np.float32) / 255.0
    return np.clip(x + amount * (x - low), 0.0, 1.0).astype(np.float32)


def tone_map(
    x: np.ndarray,
    black: float = 0.0,
    white: float = 0.95,
    scurve: float = 0.16,
) -> np.ndarray:
    """影调映射：重设黑白场 + 轻度 S 曲线。

    夜景素材需要压住暗部、保住高光，否则量化后暗部会变成一片死黑。
    ``scurve=0`` 时退化为纯线性重映射。
    """
    y = np.clip((x - black) / max(white - black, 1e-6), 0.0, 1.0)
    s = y * y * (3.0 - 2.0 * y)
    return (y * (1.0 - scurve) + s * scurve).astype(np.float32)


def fit_to_aspect(img: Image.Image, aspect: float = 16 / 9) -> Image.Image:
    """居中裁切到指定宽高比（不缩放）。"""
    w, h = img.size
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return img.crop((x0, 0, x0 + nw, h))
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return img.crop((0, y0, w, y0 + nh))


def fit_native(img: Image.Image, long_edge: int = 1920) -> Image.Image:
    """保持原始宽高比，把长边缩放到 ``long_edge``。

    竖构图（手机壁纸比例）必须走这条 —— 强行裁成 16:9 会毁掉大半画面。
    """
    w, h = img.size
    k = long_edge / max(w, h)
    if k >= 1.0:
        return img
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.Resampling.LANCZOS)


def upscale_nearest(img: Image.Image, width: int) -> Image.Image:
    """整数倍最近邻放大；若 width 不是原宽整数倍则抛出（像素画必须整数倍）。"""
    w, h = img.size
    if width % w:
        raise ValueError(f"{width} 不是 {w} 的整数倍，像素画必须整数倍放大")
    k = width // w
    return img.resize((w * k, h * k), Image.Resampling.NEAREST)
