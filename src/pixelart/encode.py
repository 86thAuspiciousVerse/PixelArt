"""输出编码。

⚠️ **像素画不能直接用默认设置的 MP4。**
绝大多数 MP4 用 ``yuv420p``：色度（Cb/Cr）在水平和垂直方向各减半。
像素画的方块边缘都是硬色变，这会在每条边界糊出错误颜色，
更要命的是**解码后画面不再是色板精确的** —— 而"有限色板"正是像素画的定义。
实测（tools/chroma_demo.py）：高饱和素材最大误差可达 127/255。

因此默认输出 **无损动画 WebP**（Pillow 原生支持，无需 ffmpeg）。
需要 MP4 时请显式指定 ``-pix_fmt yuv444p``，见 :func:`save_mp4_hint`。
"""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "to_pil",
    "save_still",
    "save_animation",
    "save_sequence",
    "count_frames",
    "save_mp4_hint",
]


def to_pil(frame: Image.Image | np.ndarray) -> Image.Image:
    """接受 PIL 图或 uint8/float 数组，统一成 RGB PIL 图。

    float 输入按 [0,1] 解释并**四舍五入**到 uint8（不是截断）。
    """
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        scale = 255.0 if a.max() <= 1.0 else 1.0
        a = np.rint(np.clip(a * scale, 0, 255)).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


def save_still(img: Image.Image | np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    to_pil(img).save(p, optimize=True)
    return p


def save_animation(
    frames: Sequence[Image.Image | np.ndarray],
    path: str | Path,
    duration_ms: int = 33,
    lossless: bool = True,
    quality: int = 100,
    method: int = 3,
) -> Path:
    """保存无缝循环动画。

    扩展名决定格式：
      ``.webp`` → 动画 WebP（推荐，可无损）
      ``.png``  → APNG（无损，体积更大）
      ``.gif``  → GIF（256 色上限，仅小尺寸时考虑）

    Args:
        duration_ms: 每帧显示时长。33ms ≈ 30fps。
        lossless: 仅 WebP 有效。像素画强烈建议 True。
        method: WebP 压缩**努力程度** 0~6。

            ⚠️ 默认取 3 而不是 Pillow 的最高档 6 —— 这是实测标定的：

            | method | 耗时 | 体积 |
            |---|---|---|
            | 0 | 0.29 s | 0.256 MB |
            | **3** | **0.60 s** | **0.238 MB** |
            | 6 | **30.5 s** | 0.237 MB |

            （1920x1016、6 帧、真实像素画帧）

            **method 6 慢 50 倍，只换来 0.4% 的体积**。而且 ``lossless=True``
            时 method **完全不影响画质** —— 它只决定压缩器花多少力气，
            无损就是无损。对 180 帧的视频，30 秒/6 帧意味着要跑 15 分钟，
            换 0.4% 体积，完全不划算。
    """
    if not frames:
        raise ValueError("frames 为空")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    pil = [to_pil(f) for f in frames]
    ext = p.suffix.lower()

    if ext == ".webp":
        pil[0].save(p, save_all=True, append_images=pil[1:], duration=duration_ms,
                    loop=0, lossless=bool(lossless), quality=int(quality), method=int(method))
    elif ext in (".png", ".apng"):
        pil[0].save(p, save_all=True, append_images=pil[1:], duration=duration_ms, loop=0)
    elif ext == ".gif":
        # GIF 只有 256 色且有 1-bit 透明，仅适合小尺寸 / 低色数内容。
        # 这里把 RGB 帧直接交给 Pillow，由它做全局量化 —— 不要只拿第一帧的调色板
        # 去量化其余帧，那样会把后面的颜色全部映射到第一帧的色域里。
        pil[0].save(p, save_all=True, append_images=pil[1:], duration=duration_ms,
                    loop=0, disposal=2, optimize=False)
    else:
        raise ValueError(f"不支持的扩展名: {ext}（用 .webp / .png / .gif）")
    return p


def save_sequence(frames: Iterable[Image.Image | np.ndarray], directory: str | Path,
                  prefix: str = "frame", digits: int = 4) -> list[Path]:
    """导出 PNG 序列（仅在需要二次编辑时用，体积很大）。"""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for i, f in enumerate(frames):
        p = d / f"{prefix}_{i:0{digits}d}.png"
        to_pil(f).save(p, optimize=True)
        out.append(p)
    return out


def count_frames(path: str | Path) -> int:
    """数一个动画文件有多少帧（也用于校验输出可被解码）。"""
    im = Image.open(path)
    n = 1
    try:
        while True:
            im.seek(n)
            n += 1
    except EOFError:
        return n


def save_mp4_hint() -> str:
    """需要 MP4 时的正确命令（色度必须 4:4:4）。"""
    return (
        "ffmpeg -framerate 30 -i frame_%04d.png "
        "-c:v libx264 -pix_fmt yuv444p -crf 14 -movflags +faststart out.mp4\n"
        "（关键：-pix_fmt yuv444p。默认的 yuv420p 会破坏像素画的色板。）"
    )
