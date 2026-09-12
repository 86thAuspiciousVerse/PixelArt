"""天空层替换（M5.7）：把深度模型搞不定的天空，换成程序化渐变 + 星场。

**为什么**：天空在单目深度模型里是"无限远的平面"，输出噪且平；量化后
发脏，还白白占用自动色板的槽位。画面里**真的有天空**时（双保守门：
天空连通区占比 ≥ 2%），把天空区替换成程序化的竖直渐变，夜景/暮色再叠加
**确定性星场**（`animate.hash01`，绝不逐帧重采样）。

**接入位置**：在 ``prepare_scene`` 里替换 **base**（场景构建阶段）——
下游的雾/体积光/辉光/色板/量化全部自动吃到新天空，CPU 与 GPU 两条路径
无需任何改动（base 是打包进场景包的）。

**天空检测的实测教训**：天空的原始深度是一个**逐图不同的平台值**
（实测 ref01≈0.953 / ref10≈0.965 / ref06≈0.987），固定绝对阈值
（0.985）在 ref01/ref10 上命中率 0%。因此用**自适应**：顶部 10% 行的
中位深度 = 平台值，天空 = 每列从顶往下、深度不跌破（平台 − 0.02）的
连通段。**适用范围**：室内图没有"无限远"区域，双保守门恒不触发，
逐位不变。
⚠️ v1 星场是**静态**的（烘焙在 base 里）；"星星闪烁"需要合成阶段的
逐帧通路（CPU+GPU 各自实现且保持 parity），留作 v2。

渐变与星星的配色直接以 sRGB 写在 ``base`` 上（base 尚未进色板/感知空间），
预设值即最终视觉值，不要在这里做任何空间变换。
"""

from __future__ import annotations

import numpy as np

from .animate import hash01

#: 预设：顶部色 / 地平线色 / 星星基准密度（占候选格点的比例）
SKY_PRESETS: dict[str, dict] = {
    "night": {"top": (11, 16, 38), "hor": (30, 44, 84), "star_base": 0.012},
    "dusk": {"top": (43, 29, 58), "hor": (224, 148, 92), "star_base": 0.004},
    "day": {"top": (126, 195, 232), "hor": (209, 233, 247), "star_base": 0.0},
}

#: 天空占比低于这个值就不动画面（双保守门之二）
MIN_FRACTION = 0.02

#: 渐变色带数。⚠️ 必须色带化：连续渐变在像素画里读作"HD 贴图"，
#: 与全图的方块语言割裂（用户实测退回）；色带 + 抖动才是像素画天空
#: 的经典语言。9 段 = 明显但柔和的色带。
SKY_BANDS = 9

#: 天空检测的深度容差（相对平台值的绝对量）
PLATEAU_TOL = 0.02


def detect_sky(far_raw: np.ndarray, tol: float = PLATEAU_TOL) -> tuple[np.ndarray, float]:
    """自适应天空检测。返回 ``(掩膜, 平台值)``。

    平台值 = 顶部 10% 行的中位深度（有天空时顶部几乎必然是天空）。
    平台 < 0.5 视为"顶部不是远处"（室内/俯拍），返回空掩膜。
    天空 = 每列从图像顶往下、深度不跌破（平台 − tol）的连续段 ——
    天然满足"与顶部连通"，悬檐/桥洞不会被误接。

    ⚠️ **地平线截断**：海面/大水面与天空同处"远"深度档，逐列连通会把
    海一起划进天空（ref10 实测：海被涂成渐变 + 阶梯边）。因此再取
    「首个天空列占比 ≤ 50% 的行」为地平线，只保留其以上部分 ——
    海面留在画面里（暮色天空 + 蓝海反而是好看的搭配）。
    """
    h, w = far_raw.shape
    top_rows = max(1, int(h * 0.10))
    plateau = float(np.median(far_raw[:top_rows]))
    if plateau < 0.5:
        return np.zeros((h, w), dtype=bool), plateau
    thr = plateau - tol
    below = far_raw < thr
    first_below = np.where(below.any(axis=0), below.argmax(axis=0), h)
    sky = np.arange(h, dtype=np.int64)[:, None] < first_below[None, :]

    # 地平线截断：逐行统计天空列占比，占比跌破 50% 的那一行 = 地平线
    row_frac = sky.mean(axis=1)
    hor_rows = np.nonzero(row_frac <= 0.5)[0]
    if hor_rows.size:
        sky[hor_rows[0]:, :] = False
    return sky, plateau


def sky_fraction(far_raw: np.ndarray) -> float:
    """天空占比（0~1）。供调用方决定是否触发。"""
    return float(detect_sky(far_raw)[0].mean())


def sky_replace(
    base: np.ndarray,
    far_raw: np.ndarray,
    mode: str = "night",
    stars: float = 1.0,
    feather: float = 1.2,
    seed: int = 7,
) -> tuple[np.ndarray, dict | None]:
    """把天空区替换为程序化渐变（+ 星场）。返回 ``(base', info)``。

    未触发双保守门时原样返回 ``(base, None)`` —— **逐位不变**。

    Args:
        base: 场景基础色 (H, W, 3) float32，sRGB 0~1（prepare_scene 的 ② 之后）。
        far_raw: **未均衡化**的原始深度（均衡化会抹平深度层次，检测失效 ——
            与 detect_light_source 的天空排除同一约定）。
        mode: "night" / "dusk" / "day"。未知值按 "night"。
        stars: 星密度倍率 0~2（0 = 无星；只有 night/dusk 有星）。
        feather: 掩膜羽化 sigma（px）。边界硬会锯齿，太软会往建筑上渗色。
        seed: 星点哈希种子（确定性：同参数同星图）。

    Returns:
        (替换后的 base, info)；info 含 fraction/stars/anchor 供诊断与前端展示。
    """
    preset = SKY_PRESETS.get(mode, SKY_PRESETS["night"])
    m, plateau = detect_sky(far_raw)
    frac = float(m.mean())
    if frac < MIN_FRACTION or not m.any():
        return base, None

    h, w = base.shape[:2]
    top = np.array(preset["top"], dtype=np.float32) / 255.0
    hor = np.array(preset["hor"], dtype=np.float32) / 255.0

    # 竖直渐变：t = 行在天空纵向范围 [y0, y1] 内的位置
    rows = np.nonzero(m.any(axis=1))[0]
    y0, y1 = int(rows[0]), int(rows[-1])
    t = np.clip((np.arange(h, dtype=np.float32) - y0) / max(y1 - y0, 1), 0.0, 1.0)
    # ⭐ 色带化（banding）：连续渐变会被读成"高清贴图"，与全图的像素块
    #    语言割裂（用户实测退回）。折成 SKY_BANDS 段阶梯 —— 色带 + 抖动
    #    是像素画天空的经典语言。量化后整片天空只剩 ≤ SKY_BANDS 种颜色。
    t = np.round(t * (SKY_BANDS - 1)) / (SKY_BANDS - 1)
    grad = top[None, None, :] * (1.0 - t[:, None, None]) + hor[None, None, :] * t[:, None, None]
    sky = np.broadcast_to(grad, base.shape).copy()

    # ── 星场（确定性：格点候选 + hash01 筛选，绝不逐帧重采样）──
    star_count = 0
    star_base = float(preset["star_base"]) * float(np.clip(stars, 0.0, 2.0))
    if star_base > 0.0:
        for cy in range(1, h - 1, 3):                      # 3px 格点候选
            for cx in range(1, w - 1, 3):
                if not (m[cy, cx] and m[cy - 1, cx] and m[cy + 1, cx]
                        and m[cy, cx - 1] and m[cy, cx + 1]):
                    continue                               # 只放远离边界的位置
                if hash01(cx, cy, seed) > star_base * 9.0:
                    continue
                b = 0.35 + 0.65 * hash01(cx, cy, seed + 1)  # 亮度 0.35~1
                warm = hash01(cx, cy, seed + 2) > 0.7
                col = np.array([255, 236, 200] if warm else [214, 226, 255],
                               dtype=np.float32) / 255.0
                sky[cy, cx] = sky[cy, cx] * (1.0 - b) + col * b
                star_count += 1

    # 羽化混合：边界 1~2px 过渡，避免掩膜锯齿
    from .compose import blur
    alpha = blur(m.astype(np.float32), float(feather))[..., None]
    out = base * (1.0 - alpha) + sky * alpha
    info = {"fraction": frac, "stars": star_count, "y0": y0, "y1": y1,
            "mode": mode, "plateau": round(plateau, 3)}
    return out.astype(np.float32), info
