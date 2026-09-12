"""视差演示 —— 生成**能看见**的证据：循环动图 + 差分图。

为什么需要它：视差是**时间上的**效果，单帧看不出来；而且 t=0 处位移
恰好为零（循环闭合的要求），所以"盯着首帧找变化"永远找不到。

产物（落 out/m5/）：
  · parallax_demo_<asset>.gif   30 帧循环，×3 整数放大 —— 动的部分一眼可见
  · parallax_diff_<asset>.png   左：t=0 帧；中：t=0.5 帧；右：差分×N（放大看差异位置）

用法： python tools/parallax_demo.py [--asset 名字] [--amp 0.5] [--grid 240]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from pixelart.pipeline import AnimParams, compose_frame, finish_frame, upscale
from pixelart.tune import TuneParams, build_scene


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="ref04_lain_room.jpg")
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--grid", type=int, default=240)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--zoom", type=int, default=3)
    args = ap.parse_args()

    params = TuneParams.from_query({
        "asset": [args.asset], "grid_long": [str(args.grid)],
        "work_long": [str(args.grid * 4)], "aspect": ["native"],
    })
    img = Image.open(ROOT / "assets" / "input" / args.asset)
    scene = build_scene(img, params)
    base = params.to_anim()
    kw = params.compose_kwargs(scene)

    def frame(t, amp):
        anim = AnimParams(**{**base.to_dict(), "parallax": amp})
        u8, _ = finish_frame(compose_frame(scene, t=t, anim=anim), scene.tail)
        return u8

    out_dir = ROOT / "out" / "m5"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.asset.replace(".jpg", "").replace(".png", "")

    # ① 循环动图（t=1 与 t=0 逐位相同，所以最后一帧不重复放）
    frames = [upscale(frame(i / args.frames, args.amp), args.zoom)
              for i in range(args.frames)]
    gif = out_dir / f"parallax_demo_{stem}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / 12), loop=0, optimize=False)
    print(f"① 循环动图（{args.frames} 帧，×{args.zoom}）: {gif}")

    # ② 差分图：t=0 / t=0.5 / 差分×6
    f0 = frame(0.0, args.amp)
    f5 = frame(0.5, args.amp)
    d = np.abs(f5.astype(np.int16) - f0.astype(np.int16)) * 6
    diff = np.clip(d, 0, 255).astype(np.uint8)
    gap = 6
    panels = [f0, f5, diff]
    h = f0.shape[0]
    W = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((h, W, 3), dtype=np.uint8)
    x = 0
    for p_ in panels:
        canvas[:, x:x + p_.shape[1]] = p_
        x += p_.shape[1] + gap
    png = out_dir / f"parallax_diff_{stem}.png"
    upscale(canvas, 2).save(png)
    changed = float((np.abs(f5.astype(int) - f0.astype(int)).max(axis=2) > 4).mean())
    print(f"② 差分图（左 t=0 / 中 t=0.5 / 右 差分×6）: {png}")
    print(f"   明显变化的像素占比: {changed*100:.1f}%"
          f"（>4/255 视作肉眼可见）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
