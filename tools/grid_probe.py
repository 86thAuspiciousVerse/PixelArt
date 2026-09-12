"""像素网格探测 —— 同一张图在多档网格下走一遍像素尾巴，输出对比图。

用途：用真实数据回答「目标像素网格该定多大」，而不是拍脑袋。
结论见 docs/research-01-prior-art.md §4：
480x270（x4）是密集动画场景「结构可读」的下限；640x360 保细节但像素感变弱。

用法::

    python tools/grid_probe.py
    python tools/grid_probe.py --src assets/input/ref04_lain_room.jpg
    python tools/grid_probe.py --grids 320x180 480x270 640x360 --colors 24
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
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.pixelate import PixelTail, render_frame  # noqa: E402
from pixelart.resample import fit_to_aspect  # noqa: E402

DEFAULT_GRIDS = [(320, 180), (384, 216), (480, 270), (640, 360)]
DETAIL_BOX = (760, 280, 1380, 900)      # 1:1 裁切区（主角 + 显示器 + 线缆）


def parse_grid(s: str) -> tuple[int, int]:
    w, _, h = s.partition("x")
    return int(w), int(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--grids", nargs="*", default=None, help="如 480x270 640x360")
    ap.add_argument("--colors", type=int, default=32)
    ap.add_argument("--target-width", type=int, default=1920)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    grids = [parse_grid(g) for g in args.grids] if args.grids else DEFAULT_GRIDS
    src = Image.open(args.src).convert("RGB")
    tag = args.tag or Path(args.src).stem
    ref = fit_to_aspect(src, 16 / 9).resize((1920, 1080), Image.Resampling.LANCZOS)

    p = PixelTail(n_colors=args.colors)
    panels: list[tuple[str, np.ndarray]] = []
    bigs: dict[tuple[int, int], Image.Image] = {}

    for (w, h) in grids:
        small, _ = render_frame(ref, (w, h), p)
        small.save(out("grid", f"{tag}_px_{w}x{h}.png"))
        k = args.target_width // w
        if args.target_width % w:
            print(f"  [跳过] {w}x{h}: {args.target_width} 不是 {w} 的整数倍")
            continue
        big = small.resize((w * k, h * k), Image.Resampling.NEAREST)
        big.save(out("grid", f"{tag}_up_{w}x{h}_x{k}_1920x1080.png"))
        bigs[(w, h)] = big
        panels.append((f"{w}x{h}  放大 {k} 倍", np.asarray(big.resize((960, 540), Image.Resampling.NEAREST))))
        print(f"[ok] {w}x{h}  x{k}  colors={args.colors}")

    # 全景 2x2
    sheet = Image.new("RGB", (1920, 1080), (8, 9, 13))
    for i, (title, arr) in enumerate(panels[:4]):
        cx, cy = (i % 2) * 960, (i // 2) * 540
        sheet.paste(Image.fromarray(arr), (cx, cy))
        d = ImageDraw.Draw(sheet)
        d.rectangle([cx + 6, cy + 6, cx + 320, cy + 50], fill=(8, 9, 13))
        d.text((cx + 14, cy + 12), title, font=C.font(26), fill=C.LABEL_FG)
    C.save(sheet, out("grid", f"{tag}_A_grids.png"))

    # 细节 1:1
    x0, y0, x1, y1 = DETAIL_BOX
    pw, ph = x1 - x0, y1 - y0
    strip = Image.new("RGB", (pw * len(bigs), ph + 48), (8, 9, 13))
    ds = ImageDraw.Draw(strip)
    for i, ((w, h), big) in enumerate(bigs.items()):
        strip.paste(big.crop((x0, y0, x1, y1)), (i * pw, 48))
        ds.text((i * pw + 14, 10), f"{w}x{h}", font=C.font(26), fill=C.LABEL_FG)
    C.save(strip, out("grid", f"{tag}_B_detail.png"))

    print(f"\n产物目录: out/grid/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
