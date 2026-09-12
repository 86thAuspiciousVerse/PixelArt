"""辉光模糊算子的差异量化 —— 决定 WGSL 侧该怎么实现。

═══ 要回答的问题 ═══

``compose.blur`` 走的是 PIL：
``float → uint8（截断）→ ImageFilter.GaussianBlur → float/255``

这条路有两处不能照搬到 GPU：
  1. **中间降到 uint8**。量化误差会被后面的锐化/量化放大，而且暗部最吃亏
     （辉光恰恰在暗部最可见）。
  2. **GaussianBlur 是三次盒式模糊的近似**，且具体盒宽由 Pillow 版本决定
     （``BoxBlur.c`` 里的实现细节）→ 逐位复现既不现实也不稳。

所以本工具量三件事，用数据决定路线：

  A. PIL(radius) vs **真高斯**（σ=radius）——差多少？
  B. PIL(radius) vs **cv2 的 float 高斯** ——差多少？
  C. 差异经过完整 bloom → 色板吸附之后，还剩多少？

如果 A/C 都很小，那"改用真高斯"就是安全的，而且顺手修掉 uint8 量化。
如果 C 很大，就必须老老实实保留 PIL 路径、并把 bloom 明确标为"不追求逐位"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pixelart.compose import LUMA, bloom, bright_pass, to_u8  # noqa: E402

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def blur_pil(a: np.ndarray, radius: float) -> np.ndarray:
    """当前实现（compose.blur 的做法）。"""
    squeeze = a.ndim == 3 and a.shape[2] == 1
    src = a[..., 0] if squeeze else a
    out = np.asarray(
        Image.fromarray(to_u8(src)).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32) / 255.0
    if a.ndim == 3:
        return out[..., None] if squeeze else out
    return out


def blur_cv2(a: np.ndarray, radius: float) -> np.ndarray:
    """cv2 的 float32 高斯（不降到 uint8）。k 取 2*ceil(3σ)+1，与真高斯一致。"""
    sigma = float(radius)
    k = int(2 * np.ceil(3 * sigma)) + 1
    return np.asarray(cv2.GaussianBlur(a.astype(np.float32), (k, k), sigma,
                                       borderType=cv2.BORDER_REFLECT),
                      dtype=np.float32)


def blur_true(a: np.ndarray, radius: float) -> np.ndarray:
    """完全用 numpy 实现的可分离真高斯（作为"真值"参照，慢但可信）。

    ⚠️ 不能用 ``np.convolve(..., mode="same")``：当卷积核**比信号长**时
    （h=64 而核 67），``mode="same"`` 返回的是**核的长度**（67），
    形状就变了。第一版就是这么写的，直接 broadcast 报错。
    手写一个带反射填充的卷积，长度永远不变。
    """
    sigma = float(radius)
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    # 用于 np.pad 的宽度必须是对称的 (r, r)
    pad = ((r, r), (0, 0)) if a.ndim == 2 else ((r, r), (0, 0), (0, 0))
    kernel = k.reshape((-1,) + (1,) * (a.ndim - 1))

    def conv_axis(arr, axis):
        # 沿 axis 做一维卷积：先反射填充，再用滑动窗口求和
        widths = [(0, 0)] * arr.ndim
        widths[axis] = (r, r)
        p = np.pad(arr.astype(np.float64), widths, mode="reflect")
        out = np.zeros_like(arr, dtype=np.float64)
        for i, kv in enumerate(k):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + arr.shape[axis])
            out += p[tuple(sl)] * kv
        return out

    out = a.astype(np.float64)
    out = conv_axis(out, 1)
    out = conv_axis(out, 0)
    return out.astype(np.float32)


def _test_image(w: int, h: int, seed: int = 7) -> np.ndarray:
    """带高光块的测试图 —— 辉光只在有高光时才显形。"""
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lum = 0.10 + 0.25 * (xx / max(w - 1, 1)) * (yy / max(h - 1, 1))
    rgb = np.stack([lum * 1.0, lum * 0.95, lum * 1.15], -1)
    # 几个高亮点（辉光的来源）
    for (cy, cx, rad, val) in ((h // 4, w // 3, 3, 1.0),
                               (h // 2, 2 * w // 3, 2, 0.9),
                               (3 * h // 4, w // 5, 2, 0.8)):
        yy2, xx2 = np.mgrid[0:h, 0:w]
        m = (yy2 - cy) ** 2 + (xx2 - cx) ** 2 <= rad * rad
        rgb[m] = val
    rgb += rs.uniform(-0.01, 0.01, rgb.shape).astype(np.float32)
    return np.clip(rgb, 0, 1).astype(np.float32)


def main() -> int:
    w, h = 96, 64
    img = _test_image(w, h)

    print("辉光模糊算子差异量化")
    print("=" * 76)
    print(f"  cv2 可用: {HAVE_CV2}")
    print()

    # ── 单算子层面 ──
    print("① 单算子层面（作用于 bright_pass 的输出，即真正被模糊的东西）")
    bright = bright_pass(img, 0.55, knee=0.25)
    print(f"   {'半径':>5s} {'PIL vs 真高斯':>16s} {'PIL vs cv2':>14s} "
          f"{'cv2 vs 真高斯':>14s}")
    for radius in (2.0, 5.0, 11.0):
        a = blur_pil(bright, radius)
        b = blur_true(bright, radius)
        c = blur_cv2(bright, radius) if HAVE_CV2 else b
        d_ab = float(np.abs(a - b).max())
        d_ac = float(np.abs(a - c).max())
        d_cb = float(np.abs(c - b).max())
        print(f"   {radius:5.1f} {d_ab:16.3e} {d_ac:14.3e} {d_cb:14.3e}")

    print()
    print("   参考尺度：1 个 uint8 色阶 = 1/255 ≈ 3.9e-3")

    # ── 算子对最终 uint8 的影响 ──
    print()
    print("② 算子差异放大到 uint8 后的像素级影响")
    for radius in (2.0, 5.0, 11.0):
        a = np.clip(blur_pil(bright, radius) * 255, 0, 255).astype(np.uint8)
        b = np.clip(blur_true(bright, radius) * 255, 0, 255).astype(np.uint8)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        n = int((d > 0).sum())
        print(f"   半径 {radius:4.1f}: 不一致像素 {n:5d}/{a.size} "
              f"({100.0 * n / a.size:5.2f}%)  最大差 {int(d.max())} 色阶  "
              f"均值 {d.mean():.3f}")

    # ── 端到端：bloom → 色板吸附 ──
    print()
    print("③ 端到端：完整 bloom 之后（含高光提取 + 三尺度加权 + 加回原图）")
    pal = _test_palette(32)
    from pixelart.palette import snap_exact, to_perceptual

    ref = bloom(img, threshold=0.55, strength=0.85 * 0.5, radii=(2, 5, 11))

    # 用真高斯替换 blur，其余一致 —— 模拟"GPU 用真高斯"的结果
    def bloom_true(rgb01, threshold, strength, radii):
        b = bright_pass(rgb01, threshold, knee=0.25)
        acc = np.zeros_like(rgb01)
        wsum = 0.0
        for i, r in enumerate(radii):
            ww = 1.0 / (i + 1)
            acc += blur_true(b, r) * ww
            wsum += ww
        return np.clip(rgb01 + acc / max(wsum, 1e-6) * strength, 0.0, 1.0)

    alt = bloom_true(img, 0.55, 0.85 * 0.5, (2, 5, 11))
    d = np.abs(ref - alt)
    print(f"   bloom 输出最大差 {d.max():.3e}  均值 {d.mean():.3e}")

    # 通过色板吸附后
    sr = snap_exact(to_perceptual(ref), pal)
    sa = snap_exact(to_perceptual(alt), pal)
    n_snap = int((np.abs(sr - sa).max(axis=2) > 0).sum())
    print(f"   经色板吸附后不一致像素 {n_snap}/{w * h} "
          f"({100.0 * n_snap / (w * h):.2f}%)")

    print()
    print("=" * 76)
    print("  判读：")
    print("   · ① 若 PIL vs 真高斯在 uint8 尺度（3.9e-3）以内 → 换真高斯是安全的")
    print("   · ③ 若吸附后不一致像素 <2% → 影响可忽略，bloom 用容差验收即可")
    print("   · 若 ③ 很大 → 必须保留 PIL 路径，并把 bloom 标为'不追求逐位'")
    return 0


def _test_palette(n: int = 32, seed: int = 3) -> np.ndarray:
    rs = np.random.RandomState(seed)
    greens = np.stack([rs.uniform(0.1, 0.5, n // 2),
                       rs.uniform(0.5, 0.9, n // 2),
                       rs.uniform(0.1, 0.4, n // 2)], -1)
    rest = rs.uniform(0.05, 0.95, (n - n // 2, 3))
    return np.clip(np.concatenate([greens, rest], 0), 0, 1).astype(np.float32)


if __name__ == "__main__":
    sys.exit(main())
