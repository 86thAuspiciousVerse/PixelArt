"""M1 冒烟预览 —— 把「分析 → 预处理 → 合成 → 像素尾巴」整条链路串起来看一眼。

**这不是成品**，用来回答两个问题：

1. 加上深度雾 / 光轴 / 辉光之后，有没有「2.5D 体积雾的光线感」？
2. 像素化之后**利不利**（会不会发糊）？

输出对比：

    A  原图（未像素化）
    B  只做像素尾巴            —— 相当于市面上的「像素滤镜」
    C  + 深度雾
    D  + 光轴 + 辉光           —— 完整合成，然后才像素化   ← 目标形态

`--sweep` 会额外给出雾浓度的参数扫描（调参用）。

⚠️ 三条踩坑得来的经验（详见 docs/spike-log.md）：

- 单目深度是**排名不是距离**，分布严重偏斜，必须按排名重映射（`depth_equalize`），
  否则雾会把整幅吞黑、连色板都跟着变暗。
- 雾必须**只在远端发力**（`power ≈ 3`），否则中景就糊。
- 雾色要用远景的**中位数**（不是均值，会被光轴这类亮块带跑）。
- 低对比素材必须先过 `auto_levels`，否则色板被浪费在同色暗部上，
  量化后就是"又灰又糊"。再补一次 `unsharp` 让方块边缘立起来。

用法::

    python tools/m1_preview.py --src assets/input/ref04_lain_room.jpg
    python tools/m1_preview.py --src assets/input/ref05_veil_city.jpg --aspect native
    python tools/m1_preview.py --src assets/input/ref05_veil_city.jpg --aspect native --sweep 0.4,0.7,1.1
    python tools/m1_preview.py --levels 0 --sharpen 0        # 关掉预处理做对照
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.compose import (  # noqa: E402
    bloom,
    brightest_center,
    bright_pass,
    depth_equalize,
    depth_fog,

    volumetric_light,
)
from pixelart.palette import palette_from_image  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.resample import (  # noqa: E402
    auto_levels,
    fit_native,
    fit_to_aspect,
    structure_aware_downsample,
    tone_map,
    unsharp,
)


def parse_aspect(s: str) -> float:
    s = s.strip().lower()
    if s in ("native", "keep", "none"):
        return 0.0
    if ":" in s:
        a, b = s.split(":", 1)
        return float(a) / float(b)
    return float(s)


def parse_color(s: str):
    if not s or s.strip().lower() in ("auto", "none"):
        return None
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 3:
        raise SystemExit("--fog-color 需要形如 0.05,0.07,0.12，或者写 auto")
    return tuple(parts)


def grid_from_aspect(w: int, h: int, long_edge: int) -> tuple[int, int]:
    """按长边定像素网格，保持原始宽高比，并取偶数。"""
    if w >= h:
        gw, gh = long_edge, round(h * long_edge / w)
    else:
        gw, gh = round(w * long_edge / h), long_edge
    return max(2, gw // 2 * 2), max(2, gh // 2 * 2)


def downsample_scalar(field: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(field.astype(np.float32), mode="F").resize(grid, Image.Resampling.BOX),
        dtype=np.float32,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--aspect", default="16:9", help="'native' 保持原比例，或 16:9 / 2.16 等")
    ap.add_argument("--grid-long", type=int, default=480, help="像素网格长边")
    ap.add_argument("--work-long", type=int, default=1920, help="工作分辨率长边")
    # 预处理
    ap.add_argument("--levels", type=float, default=0.9, help="自动色阶强度 0~1")
    ap.add_argument("--sharpen", type=float, default=0.55, help="像素网格上的 USM 强度")
    # 合成
    ap.add_argument("--density", type=float, default=0.7, help="雾浓度")
    ap.add_argument("--power", type=float, default=3.0, help="雾的距离幂次")
    ap.add_argument("--fog-color", default="auto", help="'auto' 或 r,g,b (0-1)")
    ap.add_argument("--rays", type=float, default=0.55, help="光轴强度")
    ap.add_argument("--rays-center", default="auto", help="'auto' 或 归一化 x,y")
    ap.add_argument("--rays-color", default="auto",
                    help="散射色 'auto' 或 r,g,b (0-1)。测到草丛一类非光源时会偏色。")
    ap.add_argument("--rays-sat", type=float, default=0.25,
                    help="自动散射色的饱和度上限（空气散射近乎无色；0 直接关掉着色）")
    ap.add_argument("--bloom", type=float, default=0.85, help="辉光强度")
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--sweep", default="", help="雾浓度扫描列表，逗号分隔")
    ap.add_argument("--no-depth", action="store_true", help="跳过深度推理（调试用）")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # ---- 读图 + 定网格 + 定工作分辨率 -------------------------------- #
    # ⚠️ 顺序很重要：先定网格，再让工作分辨率 = 网格 × 整数倍。
    #    否则放大时不是整数倍，像素方块会变得参差不齐，直接毁掉"利"的感觉。
    src = Image.open(args.src).convert("RGB")
    a = parse_aspect(args.aspect)
    grid = grid_from_aspect(src.width, src.height, args.grid_long)
    k = max(2, round(args.work_long / args.grid_long))
    work = (grid[0] * k, grid[1] * k)

    if a <= 0:
        ref = fit_native(src, max(work)).resize(work, Image.Resampling.LANCZOS)
    else:
        ref = fit_to_aspect(src, a).resize(work, Image.Resampling.LANCZOS)

    key = args.tag or Path(args.src).stem
    print(f"原图 {src.width}x{src.height}  网格 {grid[0]}x{grid[1]}  放大 x{k}  工作 {work[0]}x{work[1]}")

    # ---- ① 分析 ------------------------------------------------------ #
    if args.no_depth:
        far01 = np.linspace(0.0, 1.0, ref.height, dtype=np.float32)[:, None].repeat(ref.width, 1)
        print("depth: 跳过（竖直渐变代替）")
    else:
        far01 = DepthEstimator().predict_far(ref)
    far_grid = depth_equalize(downsample_scalar(far01, grid), strength=0.5)
    print("  深度(按排名重映射后)分位 5/25/50/75/95: " +
          "  ".join(f"{v:.2f}" for v in np.percentile(far_grid, [5, 25, 50, 75, 95])))

    # ---- ② 预处理：降采样 → 自动色阶 → 影调 → 锐化 -------------------- #
    p = PixelTail(n_colors=args.colors)
    base = structure_aware_downsample(ref, grid, p.var_gain)
    base = auto_levels(base, clip=(1.0, 99.0), strength=args.levels)
    base = tone_map(base, p.tone_black, p.tone_white, p.tone_scurve)
    base = unsharp(base, amount=args.sharpen)

    # ---- ③ 合成 ------------------------------------------------------ #
    def composite(density: float):
        fogged = depth_fog(base, far_grid, color=parse_color(args.fog_color),
                           density=density, power=args.power)
        cx, cy = (brightest_center(fogged, 3.0)
                  if args.rays_center.strip().lower() in ("auto", "")
                  else tuple(float(v) for v in args.rays_center.split(",")))
        # 体积光（深度感知：有遮挡、按场景距离衰减、散射带光源颜色）
        # 散射色默认自动估，且强制低饱和 —— 光源定位错时（比如落在草丛上）
        # 不加这道限制会给整幅图叠一层绿光，见 compose.auto_scatter_color。
        vol = volumetric_light(fogged, far_grid, (cx, cy), strength=args.rays,
                               air_color=parse_color(args.rays_color),
                               air_sat_max=args.rays_sat)
        # bloom 只做小半径的"发亮"，大范围的光晕交给上面的体积光
        return fogged, bloom(np.clip(fogged + vol, 0, 1), threshold=0.55,
                             strength=args.bloom * 0.5, radii=(2, 5, 11)), (cx, cy)

    fogged, lit, center = composite(args.density)
    print(f"  光轴中心 ({center[0]:.2f}, {center[1]:.2f})   雾色(自动)=" +
          "  ".join(f"{v:.2f}" for v in (np.median(base[far_grid >= np.quantile(far_grid, 0.85)], axis=0)
                                         if args.fog_color.strip().lower() in ("auto", "none", "")
                                         else parse_color(args.fog_color))))

    def px(arr: np.ndarray) -> Image.Image:
        # palette=None -> 走完整自动路径（median cut + 中性亮色补项）
        u8, _ = pixelate(arr, p, palette=None)
        return Image.fromarray(u8)

    def up(img: Image.Image) -> Image.Image:
        """整数倍最近邻放大 —— 像素画必须整数倍，否则方块参差。"""
        return img.resize((grid[0] * k, grid[1] * k), Image.Resampling.NEAREST)

    def cell(im: Image.Image, w: int, h: int) -> Image.Image:
        """等比缩放塞进 w×h 的格子（不拉伸）。"""
        t = im.copy()
        t.thumbnail((w, h), Image.Resampling.LANCZOS)
        return t

    # ---- ④ 输出 ------------------------------------------------------ #
    if args.sweep:
        dens = [float(v) for v in args.sweep.split(",") if v.strip()]
        cells = [("A  原图", ref), ("B  只做像素尾巴", up(px(base)))]
        for dd in dens:
            f_, _, _ = composite(dd)
            cells.append((f"雾 density={dd}", up(px(f_))))
        cells.append(("D  雾 + 光轴 + 辉光", up(px(lit))))

        cols, cw, ch = 3, 560, 560
        rows = (len(cells) + cols - 1) // cols
        sw = Image.new("RGB", (cols * cw, rows * ch), (8, 9, 13))
        sd = ImageDraw.Draw(sw)
        for i, (title, im) in enumerate(cells):
            sx, sy = (i % cols) * cw, (i // cols) * ch
            t = cell(im, cw - 16, ch - 46)
            sw.paste(t, (sx + (cw - t.width) // 2, sy + 40))
            sd.text((sx + 10, sy + 10), title, font=C.font(22), fill=C.LABEL_FG)
        C.save(sw, out("m1", f"sweep_{key}.png"))
        print(f"[ok] out/m1/sweep_{key}.png")
        return 0

    panels = [("A  原图（未像素化）", ref), ("B  只做像素尾巴", up(px(base))),
              (f"C  + 深度雾（{args.density}）", up(px(fogged))), ("D  + 光轴 + 辉光", up(px(lit)))]
    cw, ch = 960, 600
    sheet = Image.new("RGB", (cw * 2, ch * 2), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(panels):
        bx, by = (i % 2) * cw, (i // 2) * ch
        t = cell(im, cw - 20, ch - 50)
        sheet.paste(t, (bx + (cw - t.width) // 2, by + 44))
        d.text((bx + 12, by + 12), title, font=C.font(26), fill=C.LABEL_FG)
    C.save(sheet, out("m1", f"{key}_fog{args.density}.png"))
    C.save(up(px(lit)), out("m1", f"{key}_full.png"))
    print(f"[ok] out/m1/{key}_fog{args.density}.png")
    print(f"[ok] out/m1/{key}_full.png    ({grid[0] * k}x{grid[1] * k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
