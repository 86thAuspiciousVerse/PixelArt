"""单图深度查看 —— 输出大尺寸的「原图 / 深度」对照，用于判断**物体级**层次。

grid/depth_probe 出的是四联小图（每格 640px），细节看不清。
判断"服务器机架、显示器、人物、窗台玩偶是不是分得开"需要更大的画面。

用法::

    python tools/depth_view.py
    python tools/depth_view.py --src assets/input/ref02_replace_coffeeshop.jpg
    python tools/depth_view.py --mode overlay          # 深度叠加在原图上
    python tools/depth_view.py --mode both             # 上下三格
    python tools/depth_view.py --width 1200
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
from pixelart.analyze import DepthEstimator, colorize_depth  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.resample import fit_to_aspect  # noqa: E402


def overlay(rgb01: np.ndarray, far01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把深度伪彩半透明叠回原图：一眼看出哪个物体对应哪块深度。"""
    tint = colorize_depth(1.0 - far01).astype(np.float32) / 255.0    # 暖 = 近
    return np.clip(rgb01 * (1.0 - alpha) + tint * alpha, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--mode", default="both", choices=["stack", "overlay", "both"])
    ap.add_argument("--width", type=int, default=1100)
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGB")
    ref = fit_to_aspect(src, 16 / 9)
    w = min(args.width, ref.width)
    h = int(round(ref.height * w / ref.width))
    ref = ref.resize((w, h), Image.Resampling.LANCZOS)

    est = DepthEstimator()
    near01 = est.predict(ref)                 # 1 = 近
    far01 = 1.0 - near01

    rgb01 = np.asarray(ref, np.float32) / 255.0
    depth_rgb = colorize_depth(near01)        # 暖 = 近
    ov = overlay(rgb01, far01)

    rows = [("原图", np.asarray(ref))]
    if args.mode in ("stack", "both"):
        rows.append(("深度伪彩（暖=近，冷=远）", depth_rgb))
    if args.mode in ("overlay", "both"):
        rows.append(("深度叠在原图上", (ov * 255).astype(np.uint8)))

    label_h, gap = 44, 6
    sheet = Image.new("RGB", (w, len(rows) * (h + label_h + gap)), (8, 9, 13))
    draw = ImageDraw.Draw(sheet)
    for i, (title, arr) in enumerate(rows):
        y = i * (h + label_h + gap)
        sheet.paste(Image.fromarray(arr.astype(np.uint8)), (0, y + label_h))
        draw.text((12, y + 10), title, font=C.font(30), fill=C.LABEL_FG)

    key = Path(args.src).stem
    dst = out("probe", f"view_{key}.png")
    C.save(sheet, dst)
    print(f"[ok] {dst}   ({sheet.width}x{sheet.height})")

    # 顺带打印分位，判断深度是否被压缩
    q = np.percentile(far01, [1, 5, 25, 50, 75, 95, 99])
    print("  深度分位(1/5/25/50/75/95/99): " + "  ".join(f"{v:.3f}" for v in q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
