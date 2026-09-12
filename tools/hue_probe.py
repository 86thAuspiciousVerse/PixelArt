"""色相探针 —— 定位"某一块颜色被染成别的颜色"的问题。

⚠️ 存在的理由：`step_probe.py` 只打整体亮度与分位，**色相偏移在里面完全看不出来**。
   一次真实事故：ref10 的白色/灰色建筑在像素化后被涂成绿色，和草地一个色，
   而亮度统计完全正常。所以必须**逐通道 R/G/B** 地看。

做两件事：

1. 逐阶段打印 R/G/B 均值 + "偏绿/偏蓝"指数（用色板/区域统计）。
2. **区域追踪**：自动找出一块「明亮低饱和」区域（建筑/白墙），
   分别报告它在 原图 / 预处理后 / 合成后 / 量化后 的平均颜色，
   以及量化时它被吸附到了哪个色板项 —— 一眼看出是谁把它染绿的。

用法::

    python tools/hue_probe.py assets/input/ref10_green_cliff.jpg
    python tools/hue_probe.py assets/input/ref10_green_cliff.jpg --density 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.compose import (  # noqa: E402
    bloom,
    brightest_center,
    depth_equalize,
    depth_fog,
    volumetric_light,
)
from pixelart.palette import (  # noqa: E402
    LUMA,
    from_perceptual,
    palette_from_image,
    snap_exact,
    to_perceptual,
)
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.resample import (  # noqa: E402
    auto_levels,
    structure_aware_downsample,
    tone_map,
    unsharp,
)

#: 屏幕显示用的 R/G/B 分别是什么——探针里始终按这个顺序打印
CH = "RGB"


def hexs(c: np.ndarray) -> str:
    v = np.rint(np.clip(c, 0, 1) * 255).astype(int)
    return "#%02x%02x%02x" % tuple(v)


def rgbline(tag: str, a: np.ndarray) -> None:
    """打印一个 (H,W,3) 或 (k,3) 的逐通道均值。"""
    flat = a.reshape(-1, 3)
    m = flat.mean(axis=0)
    print(f"  {tag:26s} R={m[0]:.3f} G={m[1]:.3f} B={m[2]:.3f}   "
          f"ratio G/R={m[1] / max(m[0], 1e-6):.2f} B/R={m[2] / max(m[0], 1e-6):.2f}   "
          f"mean={m.mean():.3f}")


def neutral_mask(rgb01: np.ndarray, luma_q: float = 75.0, sat_max: float = 0.28) -> np.ndarray:
    """明亮 + 低饱和 = 建筑/白墙那类中性色区域。"""
    lum = rgb01 @ LUMA
    mx = rgb01.max(axis=-1)
    mn = rgb01.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1e-6, None)
    return (lum >= np.percentile(lum, luma_q)) & (sat <= sat_max)


def green_index(rgb01: np.ndarray) -> float:
    """偏绿指数：G 相对 R/B 的超出量。>0 = 偏绿，<0 = 偏品红。"""
    m = rgb01.reshape(-1, 3).mean(axis=0)
    return float(m[1] - 0.5 * (m[0] + m[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--grid-long", type=int, default=480)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--levels", type=float, default=0.9)
    ap.add_argument("--sharpen", type=float, default=0.55)
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--power", type=float, default=3.0)
    ap.add_argument("--rays", type=float, default=0.55)
    ap.add_argument("--bloom", type=float, default=0.85)
    ap.add_argument("--no-depth", action="store_true")
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGB")
    gw = args.grid_long if src.width >= src.height else round(src.width * args.grid_long / src.height)
    gh = args.grid_long if src.height > src.width else round(src.height * args.grid_long / src.width)
    grid = (max(2, gw // 2 * 2), max(2, gh // 2 * 2))
    work = (grid[0] * args.scale, grid[1] * args.scale)
    ref = src.resize(work, Image.Resampling.LANCZOS)

    print(f"{Path(args.src).name}  原图 {src.size}  网格 {grid}  x{args.scale}")
    print("\n=== 逐阶段逐通道 ===")
    rgbline("ref(工作分辨率)", np.asarray(ref, np.float32) / 255.0)

    if args.no_depth:
        far = np.linspace(0.0, 1.0, ref.height, dtype=np.float32)[:, None].repeat(ref.width, 1)
    else:
        far = DepthEstimator().predict_far(ref)
    far_grid = depth_equalize(
        np.asarray(Image.fromarray(far.astype(np.float32), mode="F").resize(grid, Image.Resampling.BOX),
                   dtype=np.float32),
        strength=0.5,
    )

    p = PixelTail(n_colors=args.colors)
    base = structure_aware_downsample(ref, grid, p.var_gain)
    rgbline("① 结构感知降采样", base)
    base = auto_levels(base, (1.0, 99.0), args.levels)
    rgbline("② +auto_levels", base)
    base = tone_map(base, p.tone_black, p.tone_white, p.tone_scurve)
    rgbline("③ +tone_map", base)
    base = unsharp(base, args.sharpen)
    rgbline("④ +unsharp", base)

    ref_g = structure_aware_downsample(ref, grid, p.var_gain)
    ref_g = auto_levels(ref_g, (1.0, 99.0), args.levels)
    ref_g = tone_map(ref_g, p.tone_black, p.tone_white, p.tone_scurve)
    ref_g = unsharp(ref_g, args.sharpen)

    fogged = depth_fog(base, far_grid, color=None, density=args.density, power=args.power)
    rgbline(f"⑤ +depth_fog({args.density})", fogged)
    cx, cy = brightest_center(fogged, 3.0)
    vol = volumetric_light(fogged, far_grid, (cx, cy), strength=args.rays)
    lit = bloom(np.clip(fogged + vol, 0, 1), threshold=0.55, strength=args.bloom * 0.5, radii=(2, 5, 11))
    rgbline("⑥ +体积光+辉光", lit)

    print("\n=== 中性亮区（建筑/白墙）平均色追踪 ===")
    # 用**原始参考图的网格版**定掩膜，保证追踪的是同一批像素
    m = neutral_mask(ref_g)
    print(f"  掩膜占比 {m.mean() * 100:.1f}%   （明亮且低饱和的像素）")
    for tag, arr in (("ref 网格版", ref_g), ("预处理后 base", base),
                     ("合成后 lit", lit)):
        c = arr[m].mean(axis=0)
        print(f"  {tag:16s} {hexs(c)}  R={c[0]:.3f} G={c[1]:.3f} B={c[2]:.3f}  "
              f"偏绿={green_index(arr[m]):+.3f}")

    print("\n=== 色板组成 ===")
    from pixelart.palette import ensure_neutral_highlight  # noqa: E402

    pal_used = palette_from_image(base, args.colors)
    pal_full = ensure_neutral_highlight(base, pal_used)

    lum = pal_full @ LUMA
    order = np.argsort(-lum)
    mx = pal_full.max(axis=1)
    mn = pal_full.min(axis=1)
    sat = (mx - mn) / np.clip(mx, 1e-6, None)
    print(f"  共 {len(pal_full)} 项（median cut {len(pal_used)} + 补项 {len(pal_full) - len(pal_used)}）")
    print("  最亮的 12 项（感知空间→线性的 hex）：")
    for i in order[:12]:
        c = from_perceptual(pal_full[i])
        is_neutral = "中性" if sat[i] <= 0.30 else ("偏绿" if c[1] > max(c[0], c[2]) else "偏蓝/其他")
        print(f"    idx{int(i):3d}  {hexs(c)}  lum={lum[i]:.3f} sat={sat[i]:.2f}  {is_neutral}")

    print("\n=== 建筑像素量化后被吸附到哪 ===")
    if m.any():
        qb = to_perceptual(base[m])
        d = ((qb[:, None, :] - pal_full[None, :, :]) ** 2).sum(axis=2)
        idx = d.argmin(axis=1)
        uniq, cnt = np.unique(idx, return_counts=True)
        for k in np.argsort(-cnt)[:6]:
            j = int(uniq[k])
            c = from_perceptual(pal_full[j])
            print(f"    色板 idx{j:3d} {hexs(c)} 占建筑区 {cnt[k] / cnt.sum() * 100:.1f}%  "
                  f"偏绿={green_index(c[None, :]):+.3f}")

    print("\n=== 量化后各区域偏绿指数 ===")
    q = to_perceptual(lit)
    u8, _ = pixelate(lit, p, palette=None)
    qlin = np.asarray(u8, np.float32) / 255.0
    print(f"  全图 偏绿={green_index(qlin):+.3f}")
    if m.any():
        print(f"  建筑区 偏绿={green_index(qlin[m]):+.3f}  （>0 说明建筑被染绿）")
        mc = qlin[m].mean(axis=0)
        print(f"  建筑区平均色 {hexs(mc)}  R={mc[0]:.3f} G={mc[1]:.3f} B={mc[2]:.3f}")
    gm = ~m & (qlin[:, :, 1] > qlin[:, :, 0]) & (qlin[:, :, 1] > qlin[:, :, 2])
    if gm.any():
        print(f"  绿地区 平均色 {hexs(qlin[gm].mean(axis=0))}  占比 {gm.mean() * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
