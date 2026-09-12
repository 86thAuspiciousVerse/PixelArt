"""有序抖动（ordered dithering）。

用 Bayer 矩阵把量化误差在空间上打散，让有限的色板能表达连续渐变。
这是"海报化照片"和"像素画"之间最关键的差别之一。

⚠️ 时序一致性约束：Bayer 相位必须只由**方块坐标**决定。
   任何形如 ``(x + frame) & 7`` 的写法都会导致整屏闪烁。见 docs/arch-01-design.md §6。
"""

import numpy as np

__all__ = ["bayer_matrix", "BAYER8", "bayer_tile", "bayer_signed"]


def bayer_matrix(n: int = 8) -> np.ndarray:
    """递归构造 n×n Bayer 矩阵（n 为 2 的幂），归一化到 [0, 1)。"""
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("n 必须是 >=2 的 2 的幂")
    m = np.zeros((1, 1), dtype=np.int64)
    while m.shape[0] < n:
        m = np.block([
            [4 * m + 0, 4 * m + 2],
            [4 * m + 3, 4 * m + 1],
        ])
    return m.astype(np.float32) / float(m.size)


BAYER8: np.ndarray = bayer_matrix(8)


def bayer_tile(h: int, w: int, matrix: np.ndarray = BAYER8) -> np.ndarray:
    """平铺到 (h, w, 1)，可直接与 (h, w, 3) 广播。值域 [0, 1)。"""
    n = matrix.shape[0]
    reps = (h // n + 1, w // n + 1)
    return np.tile(matrix, reps)[:h, :w][..., None]


def bayer_signed(h: int, w: int, matrix: np.ndarray = BAYER8) -> np.ndarray:
    """居中到 [-0.5, 0.5)，方便直接叠加到颜色上。"""
    return bayer_tile(h, w, matrix) - 0.5
