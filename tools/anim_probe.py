"""「动画幅度够不够」检测 —— 数值上在动 ≠ 肉眼看得到。

相邻帧平均差 0.26/255 这种量级，可能完全看不出来。
这里逐像素统计**整段视频的极差**（max−min），并把它放大成预览图，
直接看"哪些地方在动、动了几级色阶"。

判据参考：
  - 极差 0        → 那一块完全静止（合理，比如近景建筑）
  - 极差 1~2      → 基本看不出
  - 极差 ≥ 4      → 看得出在动
  - 极差 ≥ 10     → 明显

对壁纸来说，**"一部分区域明显在动 + 大部分区域安静"** 才是最舒服的 ——
全画面一起闪就是廉价效果。
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
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.pipeline import AnimParams, prepare_scene, render_video, upscale  # noqa: E402
from pixelart.pixelate import PixelTail  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--aspect", default="native")
    ap.add_argument("--n", type=int, default=60, help="统计用帧数（不必等于成片帧数）")
    ap.add_argument("--tag", default="anim")
    ap.add_argument("--fog-drift", type=float, default=0.30)
    ap.add_argument("--flicker", type=float, default=0.16)
    ap.add_argument("--dust", type=int, default=200)
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()

    aspect = 0.0 if args.aspect.strip().lower() in ("native", "keep", "none") else float(args.aspect)
    anim = (AnimParams(fog_drift=0.0, light_flicker=0.0, dust_count=0) if args.no_anim
            else AnimParams(fog_drift=args.fog_drift, light_flicker=args.flicker, dust_count=args.dust))

    print(f"渲染 {args.n} 帧做幅度统计（{Path(args.src).name}）...")
    scene = prepare_scene(Image.open(args.src), aspect=aspect,
                          tail=PixelTail(), depth_estimator=DepthEstimator())
    frames, pal, _ = render_video(scene, n_frames=args.n, anim=anim, include_last=False)

    stack = np.stack(frames).astype(np.int16)          # (N,H,W,3)
    rng_px = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2)   # (H,W) 通道最大极差

    print(f"\n网格 {scene.grid[0]}x{scene.grid[1]}   输出 {scene.out_size[0]}x{scene.out_size[1]}")
    print(f"整段极差（色阶 0~255）：")
    print(f"  全图最大 {int(rng_px.max())}   均值 {rng_px.mean():.2f}   中位 {int(np.median(rng_px))}")
    for lo, hi, label in ((1, 3, "1~2（看不出）"), (3, 5, "3~4（略微）"),
                          (5, 11, "5~10（看得出）"), (11, 256, ">=11（明显）")):
        frac = float(((rng_px >= lo) & (rng_px < hi)).mean()) * 100
        print(f"  极差 {label:14s} 占画面 {frac:5.1f}%")
    print(f"  极差 = 0（完全静止）占画面 {float((rng_px == 0).mean()) * 100:5.1f}%")

    # 可视化：极差放大 18 倍，热色 = 动得多
    vis = np.clip(rng_px.astype(np.float32) * 18.0, 0, 255).astype(np.uint8)
    heat = np.stack([vis, np.clip(vis * 0.55, 0, 255).astype(np.uint8), np.zeros_like(vis)], -1)

    k = scene.scale
    a = Image.fromarray(np.asarray(Image.fromarray(frames[0]).resize(
        (frames[0].shape[1] * k, frames[0].shape[0] * k), Image.Resampling.NEAREST)))
    b = Image.fromarray(np.asarray(Image.fromarray(heat).resize(
        (heat.shape[1] * k, heat.shape[0] * k), Image.Resampling.NEAREST)))
    cw = a.width + 20
    sheet = Image.new("RGB", (cw * 2, a.height + 50), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    sheet.paste(a, (10, 42))
    sheet.paste(b, (cw + 10, 42))
    d.text((12, 12), "第 0 帧", font=C.font(24), fill=C.LABEL_FG)
    d.text((cw + 12, 12), f"整段极差 x18（黄=动得多）max={int(rng_px.max())}",
           font=C.font(24), fill=C.LABEL_FG)
    C.save(sheet, out("m2", f"{args.tag}_amplitude.png"))
    print(f"\n[ok] out/m2/{args.tag}_amplitude.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
