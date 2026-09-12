"""tools/ 下脚本共用的杂项：中文字体、贴标签、拼图。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)

LABEL_BG = (10, 12, 18)
LABEL_FG = (255, 196, 92)


def font(size: int = 24) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


def to_pil(a: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(a, Image.Image):
        return a.convert("RGB")
    x = np.asarray(a)
    if x.dtype != np.uint8:
        x = np.clip(x * (255.0 if x.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return Image.fromarray(x).convert("RGB")


def strip(panels: Sequence[tuple[str, np.ndarray | Image.Image]],
          panel_w: int = 640, label_h: int = 44, gap: int = 4,
          bg: tuple[int, int, int] = (8, 9, 13)) -> Image.Image:
    """把若干 (标题, 图) 横向拼成一条对比图，等比缩放到统一宽度。"""
    ims = []
    for title, arr in panels:
        im = to_pil(arr)
        k = panel_w / im.width
        ims.append((title, im.resize((panel_w, max(1, round(im.height * k))), Image.Resampling.LANCZOS)))
    h = max(im.height for _, im in ims)
    total_w = len(ims) * panel_w + (len(ims) - 1) * gap
    out = Image.new("RGB", (total_w, h + label_h), bg)
    d = ImageDraw.Draw(out)
    x = 0
    for title, im in ims:
        out.paste(im, (x, label_h))
        d.text((x + 12, 8), title, font=font(26), fill=LABEL_FG)
        x += panel_w + gap
    return out


def save(img: Image.Image, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p
