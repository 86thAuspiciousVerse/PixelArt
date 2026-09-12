"""输出编码的单元测试。

重点固化 §7 的结论：动画 WebP / APNG 必须能写、能被解码、帧数正确。
"""

import numpy as np
import pytest
from PIL import Image

from pixelart.encode import (
    count_frames,
    save_animation,
    save_sequence,
    save_still,
    save_mp4_hint,
    to_pil,
)


def _frames(n: int = 5, size: tuple[int, int] = (64, 36)) -> list[Image.Image]:
    """造 n 张**结构上互不相同**的帧。

    刻意避免"纯色帧"：纯色帧在 GIF/WebP 这类会做帧合并的编码器里可能被吃掉，
    那样断言帧数就会莫名其妙地失败。
    """
    w, h = size
    out = []
    for i in range(n):
        a = np.zeros((h, w, 3), dtype=np.uint8)
        a[..., 0] = 24
        a[..., 2] = 72
        x0 = int(i * (w - 8) / max(n - 1, 1))
        a[4:14, x0:x0 + 8] = (255, 200, 40)
        a[20:28, w - 12 - x0:w - 4 - x0] = (60, 220, 180)
        a[:, i % w] = (200 - i * 23) % 256
        out.append(Image.fromarray(a))
    return out


# --------------------------------------------------------------------- #


def test_save_still(tmp_path):
    p = save_still(np.zeros((8, 12, 3), dtype=np.uint8), tmp_path / "a.png")
    assert p.exists()
    assert Image.open(p).size == (12, 8)


def test_to_pil_accepts_float_and_rounds():
    a = np.full((4, 4, 3), 0.5, dtype=np.float32)
    assert np.asarray(to_pil(a))[0, 0, 0] == 128
    b = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert np.asarray(to_pil(b))[0, 0, 0] == 255


def test_save_webp_animation_lossless(tmp_path):
    p = save_animation(_frames(5), tmp_path / "m.webp", duration_ms=40, lossless=True)
    assert p.exists() and p.stat().st_size > 0
    im = Image.open(p)
    assert im.format == "WEBP"
    assert count_frames(p) >= 2


def test_save_apng_animation(tmp_path):
    p = save_animation(_frames(5), tmp_path / "m.png", duration_ms=40)
    im = Image.open(p)
    assert im.format == "PNG"
    assert count_frames(p) == 5


def test_gif_fallback(tmp_path):
    p = save_animation(_frames(4), tmp_path / "m.gif", duration_ms=50)
    im = Image.open(p)
    assert im.format == "GIF"
    assert count_frames(p) == 4


def test_save_animation_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError):
        save_animation(_frames(2), tmp_path / "m.avi")


def test_save_animation_rejects_empty():
    with pytest.raises(ValueError):
        save_animation([], "x.webp")


def test_save_sequence(tmp_path):
    paths = save_sequence(_frames(3), tmp_path / "seq", prefix="f")
    assert len(paths) == 3
    assert all(p.exists() for p in paths)
    assert paths[0].name == "f_0000.png"


def test_webp_is_actually_lossless(tmp_path):
    """无损 WebP 解码回来应当与源帧逐像素一致。"""
    frames = _frames(3)
    p = save_animation(frames, tmp_path / "l.webp", duration_ms=40, lossless=True)
    im = Image.open(p)
    im.seek(0)
    a = np.asarray(im.convert("RGB"))
    b = np.asarray(frames[0].convert("RGB"))
    assert np.array_equal(a, b), "无损 WebP 应当逐像素一致"


def test_mp4_hint_mentions_444():
    hint = save_mp4_hint()
    assert "yuv444p" in hint
