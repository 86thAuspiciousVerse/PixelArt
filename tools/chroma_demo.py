"""色度抽样（chroma subsampling）破坏性实测。

绝大多数 MP4 默认使用 ``yuv420p``：色度（Cb/Cr）在水平和垂直方向各减半。
像素画每个方块边缘都是硬色变 —— 4:2:0 会在每条边界上糊出错误颜色，
并且**解码后画面不再是色板精确的**。

本脚本在 numpy 里复现这一步（不需要 ffmpeg），输出：
    - 近单色素材（Lain 像素画）的误差
    - 高饱和测试卡（最坏情况）的误差
以及上下两排的 1:1 裁切对比图。

结论：输出循环视频请用**无损动画 WebP**；需要 MP4 时必须 ``-pix_fmt yuv444p``。

用法::

    python tools/chroma_demo.py
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
from pixelart.dither import BAYER8 as _BAYER  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402

BAYER8 = _BAYER + 0.5                     # -> [0, 1)
R709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DETAIL_BOX = (760, 280, 1380, 900)

#: PICO-8 调色板，用于合成最坏情况测试卡
PICO8 = [(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81),
         (171, 82, 54), (95, 87, 79), (194, 195, 199), (255, 241, 232),
         (255, 0, 77), (255, 163, 0), (255, 236, 39), (0, 228, 54),
         (41, 173, 255), (131, 118, 156), (255, 119, 168), (255, 204, 170)]


# --------------------------------------------------------------------- #
def rgb_to_ycbcr(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = a @ R709
    return y, (a[..., 2] - y) / 1.8556, (a[..., 0] - y) / 1.5748


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.5748 * cr
    g = y - 0.1873 * cb - 0.4681 * cr
    b = y + 1.8556 * cb
    return np.clip(np.stack([r, g, b], -1), 0.0, 1.0)


def _resize_plane(p: np.ndarray, size: tuple[int, int], resample) -> np.ndarray:
    return np.asarray(Image.fromarray(p, mode="F").resize(size, resample), dtype=np.float32)


def chroma_420(a: np.ndarray) -> np.ndarray:
    """模拟 yuv420p：色度 2x2 平均下采样 + 双线性上采样。"""
    h, w, _ = a.shape
    y, cb, cr = rgb_to_ycbcr(a)
    half = (w // 2, h // 2)
    cb = _resize_plane(_resize_plane(cb, half, Image.Resampling.BOX), (w, h), Image.Resampling.BILINEAR)
    cr = _resize_plane(_resize_plane(cr, half, Image.Resampling.BOX), (w, h), Image.Resampling.BILINEAR)
    return ycbcr_to_rgb(y, cb, cr)


def make_saturated_card() -> np.ndarray:
    """合成高饱和像素画测试卡（4:2:0 的最坏情况）。"""
    w, h = 480, 270
    card = np.zeros((h, w, 3), dtype=np.float32)
    for i, c in enumerate(PICO8):                       # 上半：16 条饱和竖条
        card[0:120, i * 30:(i + 1) * 30] = np.array(c, np.float32) / 255.0
    tile = (np.tile(BAYER8, (150 // 8 + 2, 240 // 8 + 2))[:150, :240] / 8.0)[..., None]
    grad = np.zeros((150, 240, 3), dtype=np.float32)
    for y in range(150):
        t = y / 149.0
        for k in range(3):
            a = np.array(PICO8[8 + k * 3], np.float32) / 255.0
            b = np.array(PICO8[8 + k * 3 + 1], np.float32) / 255.0
            grad[y, :, k] = a[k] * (1 - t) + b[k] * t
    card[120:270, 0:240] = np.where(tile > 0.5, grad, np.array(PICO8[1], np.float32) / 255.0)
    for by in range(150 // 4):                          # 右下：互补饱和色棋盘
        for bx in range(240 // 4):
            c = PICO8[8 + ((bx + by) % 8)]
            card[120 + by * 4:124 + by * 4, 240 + bx * 4:244 + bx * 4] = np.array(c, np.float32) / 255.0
    return card


# --------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(out("grid", "ref04_lain_room_up_480x270_x4_1920x1080.png")))
    args = ap.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        src_path = INPUT / "ref04_lain_room.jpg"
    img = Image.open(src_path).convert("RGB")
    if img.size != (1920, 1080):
        img = img.resize((1920, 1080), Image.Resampling.NEAREST)

    card = make_saturated_card()
    card_up = Image.fromarray(np.clip(card * 255, 0, 255).astype(np.uint8)).resize(
        (1920, 1080), Image.Resampling.NEAREST)

    cases = [
        ("Lain 像素画（近单色）", np.asarray(img, np.float32) / 255.0),
        ("高饱和测试卡（最坏情况）", np.asarray(card_up, np.float32) / 255.0),
    ]

    print("=" * 60)
    rows = []
    for name, a in cases:
        sub = chroma_420(a)
        err = np.abs(a - sub)
        print(f"{name}\n    平均误差 {err.mean() * 255:6.2f}/255    最大误差 {err.max() * 255:6.1f}/255")
        rows.append((name, a, sub))
    print("=" * 60)

    x0, y0, x1, y1 = DETAIL_BOX
    pw, ph = x1 - x0, y1 - y0
    sheet = Image.new("RGB", (pw * 2, ph * len(rows) + 28 * len(rows)), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    for r, (name, a, sub) in enumerate(rows):
        for c, (tag, arr) in enumerate((("原图 4:4:4", a), ("yuv420p", sub))):
            im = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
            top = r * (ph + 28)
            sheet.paste(im.crop((x0, y0, x1, y1)), (c * pw, top + 28))
            d.text((c * pw + 12, top + 2), f"{name} — {tag}", font=C.font(22), fill=C.LABEL_FG)

    dst = out("encode", "chroma_420_vs_444.png")
    C.save(sheet, dst)
    print(f"[ok] {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
