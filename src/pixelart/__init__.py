"""pixelart —— 图片 → 2.5D 像素风（静态图 / 无缝循环视频）转换器。

管线（顺序不可颠倒）::

    分析(一次) → 结构感知降采样(一次) → 场景合成(每帧) → 像素尾巴(每帧) → 编码

设计要点见 docs/arch-01-design.md
"""

from .pixelate import PixelTail, pixelate
from .resample import structure_aware_downsample, tone_map

__version__ = "0.1.0"
__all__ = ["PixelTail", "pixelate", "structure_aware_downsample", "tone_map", "__version__"]
