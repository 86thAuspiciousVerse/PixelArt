"""语义掩膜（启发式 v1，零新依赖）+ 掩膜过滤的光源检测。

═══ 为什么先做启发式 ═══

语义分割模型（SegFormer 等）要加权重下载 / 缓存 / provider 处理一整套，
而 v1 要解决的两个问题——"别把天空当光源"与"别把大片反光当点源"——
用深度 + 等距坐标 + DoG（高斯差）就能判。先落地、用 10 张参考素材
验收，不够准再上模型。

═══ 三类掩膜 ═══

· **天空**：``far ≥ 0.985``（HANDOFF 实测星空深度饱和在 0.995）。
  深度是"无限远"的强信号，比"画面上部"更可靠（仰拍时天空在画面中央）。
· **地面反光带**：画面底部 12% 且亮度分位高 —— 湿路面反光是"大面积平滑亮"。
· **植被**：绿色占优（G 显著高于 R/B）**且**局部高频能量高（草叶的碎质感）。

═══ 光源检测 v2（detect_light_source）═══

判别"点源"与"反光面"的关键量是 **DoG 峰值**（小而亮 → 大响应；
大面积平滑 → 响应低），再排除天空。候选非空时用它，否则回退
:func:`brightest_center`（旧行为）—— 保证总有结果。
"""

from __future__ import annotations

import numpy as np

from .compose import blur, brightest_center

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: 天空的深度阈值（HANDOFF：星空深度饱和在 0.995 附近）
SKY_DEPTH = 0.985

__all__ = ["sky_mask", "vegetation_mask", "detect_light_source"]


def sky_mask(far: np.ndarray) -> np.ndarray:
    """天空掩膜：深度 ≥ SKY_DEPTH 的像素（布尔）。

    ⚠️⚠️ 历史教训：**固定绝对阈值 0.985 在真实深度输出上基本失效** ——
    天空的原始深度是**逐图不同的平台值**（实测 ref01≈0.953 / ref10≈0.965 /
    ref06≈0.987），0.985 对 ref01/ref10 的命中率是 0%。天空排除请改用
    :func:`pixelart.sky.detect_sky`（自适应平台值 + 逐列顶部连通）。
    本函数保留仅为兼容。
    """
    return np.asarray(far) >= SKY_DEPTH


def _adaptive_sky(far_raw: np.ndarray) -> tuple[np.ndarray, float]:
    """自适应天空检测（避免 masks ↔ sky 循环导入的薄封装）。"""
    from .sky import detect_sky
    return detect_sky(far_raw)


def vegetation_mask(rgb01: np.ndarray, thresh: float = 0.04) -> np.ndarray:
    """植被掩膜：绿色显著占优 + 高频碎质感（草叶）。

    绿色占优 = G − max(R, B) > thresh；高频 = 与 3px 模糊的差的绝对值均值。
    两者同时满足才判植被 —— 单看绿色会把青色墙面误判进来。
    """
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    green_dom = g - np.maximum(r, b)
    hi = np.abs(rgb01 - blur(rgb01, 3.0))
    texture = hi.mean(axis=-1)
    return (green_dom > thresh) & (texture > thresh * 0.6)


def detect_light_source(
    rgb01: np.ndarray,
    far01: np.ndarray,
    blur_radius: float = 3.0,
) -> tuple[float, float]:
    """**排除天空后**的最亮区域重心。返回归一化 (x, y)。

    与 :func:`brightest_center` 的唯一差别：**排除天空**（深度 ≥ SKY_DEPTH）。
    这是掩膜里唯一有把握的确定性收益 —— ref01 那类画面里星空/天空是
    最亮区域，但它是背景不是光源。

    ⚠️⚠️ 实验记录：DoG（小核减大核找"小而亮的点源"）被**证据否决** ——
    ref10 从白色楼体（合理）移到了剪影边缘（更差）、ref02 从招牌移到反光带。
    高斯差响应的是高频亮**边缘**（剪影），不是光源。git 历史里有完整记录。
    """
    h, w = far01.shape
    lum = np.clip(rgb01, 0.0, 1.0) @ LUMA
    lum_s = blur(lum, blur_radius)

    sky, _ = _adaptive_sky(far01)
    score = np.where(sky, np.float32(-1.0), lum_s.astype(np.float32))

    idx = int(np.argmax(score))
    if score.ravel()[idx] < 0.0:
        # 全是天空（极端退化）→ 回退旧检测
        from .compose import brightest_center
        return brightest_center(rgb01, blur_radius)
    return ((idx % w) + 0.5) / w, ((idx // w) + 0.5) / h
