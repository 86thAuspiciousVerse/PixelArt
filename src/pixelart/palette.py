"""调色板：取色、吸附、全局共享。

三个关键点：

1. **在感知空间（sqrt 近似）取色板**，而不是线性 RGB。
   夜景素材里暗部像素数量占压倒多数，直接在线性空间做 median cut 会让色板被暗部垄断，
   高光只有一两个色可用。开方后暗部被拉开，色板分配明显更合理。
2. **全局色板**：视频模式下所有帧必须共用同一块色板，否则逐帧闪烁。
   用 :func:`palette_from_frames` 一次性求出。
3. 吸附有两种实现：精确最近色（默认，稳）与 3D LUT（快，近似）。
"""

from typing import Iterable, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "to_perceptual",
    "from_perceptual",
    "palette_from_image",
    "palette_from_frames",
    "ensure_neutral_highlight",
    "refine_palette",
    "parse_hex_palette",
    "palette_to_hex",
    "build_lut",
    "snap_exact",
    "snap_lut",
]
#: ⚠️ 颜色空间约定
#: 本模块以及 ``pixelate()`` 内部的一切**色板都存放在"感知空间"**（线性 sRGB 开方后的值）。
#: 这样做是因为 median cut 在感知空间里对暗部友好得多（见模块文档）。
#: - 从图取色 → :func:`palette_from_image` / :func:`palette_from_frames`，返回的就是感知空间
#: - 从 hex 取色 → :func:`parse_hex_palette` 默认帮你转换到感知空间
#: - 要把色板展示/导出成 hex → :func:`palette_to_hex`
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def to_perceptual(x: np.ndarray) -> np.ndarray:
    """线性 [0,1] → 感知空间（开方近似）。"""
    return np.sqrt(np.clip(x, 0.0, 1.0))


def from_perceptual(x: np.ndarray) -> np.ndarray:
    """感知空间 → 线性 [0,1]。"""
    return np.clip(x, 0.0, 1.0) ** 2


def _sq_dist(x: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """``(N,3)`` 到 ``(k,3)`` 的平方距离矩阵 ``(N,k)``，float64。

    用展开式 ``|x-p|² = |x|² - 2·x·p + |p|²`` 走 BLAS，
    避免朴素广播分配 ``(N,k,3)`` 临时数组。
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(palette, dtype=np.float64)
    x2 = np.einsum("ij,ij->i", x, x)[:, None]
    p2 = np.einsum("ij,ij->i", p, p)[None, :]
    return p2 - 2.0 * (x @ p.T) + x2


def _median_cut_u8(u8: np.ndarray, n: int) -> np.ndarray:
    """对 uint8 RGB 图做 median cut，返回 (k,3) float32 in [0,1]。"""
    img = Image.fromarray(np.ascontiguousarray(u8))
    q = img.quantize(colors=int(n), method=Image.Quantize.MEDIANCUT)
    pal = np.asarray(q.getpalette(), dtype=np.float32)[: int(n) * 3].reshape(-1, 3) / 255.0
    return np.unique(pal, axis=0)


def palette_from_image(rgb01: np.ndarray, n_colors: int = 32) -> np.ndarray:
    """从单帧（感知空间输入）取色板。"""
    q = to_perceptual(rgb01)
    return _median_cut_u8(np.clip(q * 255.0, 0, 255).astype(np.uint8), n_colors)


def palette_from_frames(frames01: Iterable[np.ndarray], n_colors: int = 32,
                        max_samples: int = 32) -> np.ndarray:
    """从多帧求**全局共用**色板（时序一致性铁律 1）。

    为避免显存/内存爆炸，对帧做等间隔抽样；再把抽样帧的像素拼在一起做一次 median cut。
    """
    frames = list(frames01)
    if not frames:
        raise ValueError("frames 为空")
    if len(frames) > max_samples:
        idx = np.linspace(0, len(frames) - 1, max_samples).round().astype(int)
        frames = [frames[i] for i in idx]
    ch = [np.clip(to_perceptual(f).reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8) for f in frames]
    return _median_cut_u8(np.concatenate(ch, axis=0).reshape(-1, 1, 3), n_colors)


def _linear_saturation(palette: np.ndarray) -> np.ndarray:
    """色板各项在**线性空间**的饱和度 ``(max-min)/max``。

    ⚠️ 为什么必须在线性空间量：色板存的是感知空间（sqrt），
    而 sqrt 会把**高亮度处的色相差异压扁**。一块偏绿的白
    （``#c1d7aa``，线性饱和度 0.21，肉眼一看就是淡绿）
    在感知空间里的饱和度只有 0.11 —— 用一个统一的阈值去判它"算不算中性"，
    感知空间会把它判成中性，线性空间才判得对。
    """
    c = from_perceptual(palette)
    mx = c.max(axis=1)
    mn = c.min(axis=1)
    return (mx - mn) / np.clip(mx, 1e-6, None)


def ensure_neutral_highlight(
    rgb01: np.ndarray,
    palette: np.ndarray,
    min_fraction: float = 0.004,
    sat_max: float = 0.30,
    luma_q: float = 90.0,
    pal_sat_max: float = 0.10,
    luma_tol: float = 0.06,
) -> np.ndarray:
    """确保「明亮的低饱和区域」在色板里有对应项。

    ⚠️ median cut 是按**像素数量**分配色板的。当画面同时存在
    大面积的绿/蓝 和 一小块明亮的中性色（白墙、白雪、白衣服）时，
    那块中性色常会被并进某个绿/蓝的盒子里，代表色就变成"发绿的浅色"。
    结果就是**白墙被涂成绿墙**（实测发生过）。

    这里检查：如果画面里确实存在一块明亮的低饱和像素、
    而色板里**没有一个真正低饱和的项**能表示它，就补一个（长度 +1，不影响既有项）。

    ⚠️ **判据必须问「色板里有没有中性项」，而不是问「有没有距离近的项」。**
    曾经用「感知空间欧氏距离 < 0.07」判定"已经有中性色了"，
    结果在 `ref10_green_cliff` 上失效：色板里最亮的项是 ``#c1d7aa``（偏绿），
    但它在感知空间里离真中性只有 0.053 < 0.07 → 判定"已有" → 不补项
    → 白色建筑被整体涂成绿色，和草地一个色（用户报障）。
    原因还是 sqrt 把高亮度的色相差异压扁了：距离近 ≠ 色相中性。

    Args:
        min_fraction: 该区域至少占画面多少比例才值得管。
        sat_max: 判定"低饱和"的阈值 ``(max-min)/max``（**源像素**，线性空间）。
        luma_q: 亮度分位阈值，高于它才算"明亮"。
        pal_sat_max: 色板项算得上"中性"的线性饱和度上限。
        luma_tol: 色板项亮度需要落在种子亮度的这个邻域内，才算"能顶替"。
    """
    lum = rgb01 @ LUMA
    mx = rgb01.max(axis=-1)
    mn = rgb01.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1e-6, None)
    mask = (lum >= np.percentile(lum, luma_q)) & (sat <= sat_max)
    if mask.mean() < min_fraction:
        return palette

    seed = np.sqrt(np.median(rgb01[mask], axis=0)).astype(np.float32)   # 感知空间
    seed_lum = float(seed @ LUMA)

    # 只有「真正低饱和 **且** 亮度够近」的色板项，才算它能顶替这块中性区。
    # 偏绿的白（线性 sat 0.21）会被 pal_sat_max=0.10 排除，于是正确补项。
    pal_sat = _linear_saturation(palette)
    pal_lum = palette @ LUMA
    if bool(((pal_sat <= pal_sat_max) & (np.abs(pal_lum - seed_lum) <= luma_tol)).any()):
        return palette

    return np.vstack([palette, seed[None, :]])


def refine_palette(
    rgb01: np.ndarray,
    palette: np.ndarray,
    redundancy_tol: float = 0.010,
    cap_fraction: float = 0.35,
    luma_q: float = 85.0,
    chroma_max: float = 0.30,
    served_ok: float = 0.50,
    sat_slack: float = 0.02,
    pool_mode: str = "unserved",
    min_pixels: int = 32,
    sample: int = 50000,
    seed: int = 0,
) -> np.ndarray:
    """赎回**冗余槽位**，把它们投给「明亮低饱和」区域。色板尺寸保持不变。

    ⚠️ 这个函数存在的理由，是两次踩坑之后才想明白的：

    **坑一：median cut 按像素数量分槽位，于是"少数派色区"永远拿不到自己的色阶。**
    `ref10_green_cliff` 里植被占像素多数，32 个槽位里所有**亮部**槽位都给了绿，
    明亮的灰白建筑一个槽位都没有 → 只能吸附到偏绿的 ``#c1d7aa`` 上，
    **白楼被涂成和草地一个色**，建筑上的色彩分层（玻璃、混凝土、阴影）全丢。

    **坑二：光看"利用率"会得出错误结论。**
    我量过 `ref10` 的色板，32/32 项**全部被用到**，于是以为没有浪费。
    但"被用到"≠"不可替代"：有 13 个蓝项挤在同一个窄亮度带（0.45~0.60）里，
    逐项算**移除代价**才发现其中 6 个的代价 ≈ 0.00000~0.00018（几乎白占）。
    所以判据必须是移除代价，不是利用率。

    **坑三：按"最小化量化误差"重分配也修不了。**
    欧氏误差对**色相是盲的**：把一个白像素换成淡绿，误差只涨一点点，
    所以误差驱动的 split-merge 会把槽位继续留给绿色（实测降了 2.36% 误差，
    但亮部仍然全是绿）。要修它必须**定向**：专门找"被表示错了"的像素单独切槽位。

    算法：``median cut → 算每项移除代价 → 赎回最冗余的若干项 →
    对被表示错的像素做 median cut 得到同样多的新槽位``。
    **删除数恒等于新增数**，所以色板尺寸不变；实测整体误差基本不涨（+0.3%）。

    Args:
        redundancy_tol: 相对移除代价阈值（相对全图平均误差）。低于它才算冗余。
        cap_fraction: 最多轮换色板的这个比例。实测 0.35 是拐点：
            0.25 改善不足，0.45 起误差跳到 +5%（把好槽位也删了）。
        luma_q: 亮度分位阈值，高于它才算"明亮"。
        chroma_max: 线性饱和度上限，低于它才算"低饱和"（**源像素**侧）。
        pal_neutral_max: 色板项算得上"中性"的线性饱和度上限（**色板**侧，须比源侧更严）。
        served_ok: 如果该区域已有这个比例的像素被中性项接住，就不为它动手。
        min_pixels: 目标像素太少就不值得管。
        sample: 移除代价评估用的抽样像素数（固定 seed，保证确定性）。
    """
    palette = np.asarray(palette, dtype=np.float32)
    k = len(palette)
    if k < 4 or rgb01.ndim != 3:
        return palette

    flat = rgb01.reshape(-1, 3)
    if flat.shape[0] > sample:
        idx = np.random.default_rng(seed).choice(flat.shape[0], sample, replace=False)
        sub = flat[idx]
    else:
        sub = flat

    # ---- 1. 逐项移除代价（相对全图平均误差）----
    #
    # ⚠️ 用「一次算完距离矩阵 + 取最近与次近」代替「逐项把某一列删掉重算」。
    #    朴素写法是 O(N·k²)（k 次重算 min），在 480x254 网格上要 570 ms；
    #    这里距离矩阵只算一次，之后每项只是查表 —— O(N·k)。
    qs = to_perceptual(sub)
    D = _sq_dist(qs, palette)                       # (N, k)
    order = np.argpartition(D, 1, axis=1)[:, :2]    # 每行最近、次近（顺序未定）
    d0 = np.take_along_axis(D, order[:, :1], axis=1)[:, 0]
    d1 = np.take_along_axis(D, order[:, 1:2], axis=1)[:, 0]
    first = np.where(D[np.arange(D.shape[0]), order[:, 0]] <=
                     D[np.arange(D.shape[0]), order[:, 1]], order[:, 0], order[:, 1])
    second = np.where(first == order[:, 0], order[:, 1], order[:, 0])
    near, next_ = np.sqrt(np.minimum(d0, d1)), np.sqrt(np.maximum(d0, d1))

    err0 = float(near.mean())
    if err0 <= 1e-9:
        return palette
    # 拿掉第 i 项后：原本吸附到 i 的像素改用次近项，其余不变
    hit = first[:, None] == np.arange(k)[None, :]                  # (N, k) bool
    cost_after = np.where(hit, next_[:, None], near[:, None]).mean(axis=0)
    costs = ((cost_after - err0) / err0).astype(np.float32)

    # ---- 2. 找需要自己槽位的像素（两类）----
    #
    #   (a) **明亮且低饱和** —— 本应是灰白，却被彩色项表示 → 饱和度被夸大。
    #       典型：白墙吸到淡绿上。
    #   (b) **主色通道被翻转** —— 源色偏蓝，却被表示成绿色 → 色相方向被扭转。
    #       典型：`ref10` 的蓝灰玻璃楼被涂成绿色，和草地一个色（用户报障）。
    #
    # 两类必须都管：只做 (a) 能压低饱和度统计，但建筑的**色相**照样是绿的
    # （实测色相背叛率只从 18.0% 降到 17.0%，肉眼看还是绿）；
    # 只做 (b) 则色相好了、饱和度仍在。并集才同时改善两者。
    lum = flat @ LUMA
    mx = flat.max(axis=-1)
    mn = flat.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1e-6, None)
    faint = (lum >= np.percentile(lum, luma_q)) & (sat <= chroma_max)

    q = to_perceptual(flat)
    asn = palette[((q[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)]
    flipped = flat.argmax(axis=-1) != asn.argmax(axis=-1)

    # 只有 (a) 类需要豁免：已经被中性项接住就无需为它补槽位。
    # ⚠️⚠️ 「接住」的判据 = **吸光项不比像素本身更饱和**（线性空间，逐像素比，
    #    sat_slack 容忍数值噪声）。不能用固定阈值（旧版 pal_neutral_max=0.12）：
    #    ref10 的源中性区饱和度中位数只有 0.068，色板亮项饱和 0.08~0.12 全都
    #    "≤0.12 达标" → 91% 被判已接住 → 赎回池里没有中性像素 → 赎回出来的
    #    亮部项照样饱和 0.08~0.12 → Δsat 停在 +0.096 再也下不去（实测）。
    #    逐像素判据下，饱和 0.12 的项对饱和 0.07 的像素就是"染色"。
    sat_asn = _linear_saturation(asn)
    tinted = sat_asn > sat + sat_slack                    # 被染色的像素（逐像素）
    # ⚠️⚠️ 「接住率」= faint 中**不被染色**的比例 —— 这里曾是 tinted[faint].mean()
    #    （布尔反了：90.5% 被染色 ≥ 0.5 → 判"已接住"→ 中性赎回被永久跳过，
    #    三轮调参全部无效的根因）。教训：判定变量名必须与判定语义同向，
    #    否则 检查等于没有。
    served = (~tinted)[faint] if faint.any() else np.ones(0, dtype=bool)
    neutral_ok = bool(served.mean() >= served_ok) if served.size else True
    # 赎回池的中性成分 = **未被接住的那部分中性像素本身** ——
    # 它们才是"被表示错了"的人，对它们 median cut 出来的项才会落在
    # 真正的低饱和处。用宽口径 faint（sat≤0.30）切槽会被池内较饱和的
    # 像素拉高（实测亮项停在 sat 0.14 下不去）。
    faint_unserved = faint & tinted
    if pool_mode == "faint":
        faint_unserved = faint
    sel = flipped | (faint_unserved if not neutral_ok else np.zeros_like(faint_unserved))
    if int(sel.sum()) < min_pixels:
        return palette

    # ---- 4. 赎回最冗余的槽位，换取等量的新槽位 ----
    order = np.argsort(costs)
    redundant = [int(i) for i in order if costs[i] <= redundancy_tol]
    wish = min(len(redundant), max(1, int(round(k * cap_fraction))))
    if wish <= 0:
        return palette

    # ---- 5. 赎回池**按人群分池**切槽位 ──
    #   中性池单独 median cut → 亮部项真正落在低饱和处（混切会被翻转池的
    #   彩色像素拉高饱和，实测亮项停在 0.08~0.12 下不去）；翻转池单独切 →
    #   保住色相方向（蓝灰玻璃要有蓝灰项）。槽位按两池像素量分配。
    q_u8 = np.clip(q * 255.0, 0, 255).astype(np.uint8)
    f_m = sel & faint_unserved
    r_m = sel & ~faint_unserved
    parts = []
    if f_m.any():
        wish_f = int(round(wish * int(f_m.sum()) / int(sel.sum()))) if r_m.any() else wish
        wish_f = min(max(wish_f, 1), wish)
        parts.append(_median_cut_u8(q_u8[f_m].reshape(-1, 1, 3), wish_f))
    if r_m.any():
        wish_r = wish - sum(p.shape[0] for p in parts)
        if wish_r > 0:
            parts.append(_median_cut_u8(q_u8[r_m].reshape(-1, 1, 3), wish_r))
    if not parts:
        return palette
    add = np.unique(np.vstack(parts), axis=0)
    drop = redundant[: len(add)]
    if not drop:
        return palette
    return np.unique(np.vstack([np.delete(palette, drop, axis=0), add]), axis=0)


def parse_hex_palette(colors: Sequence[str], perceptual: bool = True) -> np.ndarray:
    """``['#0a0f1c', ...]`` → (k,3) float32。

    ``perceptual=True``（默认）会转换到**感知空间**，与 :func:`palette_from_image`
    的返回值保持同一约定，可直接传给 ``pixelate(palette=...)``。
    """
    out = []
    for c in colors:
        c = c.lstrip("#")
        if len(c) == 3:
            c = "".join(ch * 2 for ch in c)
        if len(c) != 6:
            raise ValueError(f"非法颜色: {c}")
        out.append([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    arr = np.asarray(out, dtype=np.float32)
    return to_perceptual(arr) if perceptual else arr


def palette_to_hex(palette: np.ndarray) -> list[str]:
    """感知空间色板 → sRGB hex 列表（展示 / 导出 / 写 params.json 用）。"""
    srgb = from_perceptual(np.asarray(palette, dtype=np.float32))
    return ["#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int)) for c in srgb]


def snap_exact(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """精确最近色吸附（欧氏距离）。arr: (...,3) float [0,1]。

    ⚠️ 实现上用**展开式 + 矩阵乘**，不是朴素的 ``(N,k,3)`` 广播：

    ``|x-p|² = |x|² - 2·x·p + |p|²``

    对 ``argmin`` 而言 ``|x|²`` 是常数，所以只需 ``argmin(|p|² - 2·x·p)``，
    也就是一次 ``(N,3) @ (3,k)`` 的矩阵乘 —— 走 BLAS，**结果与朴素实现等价**
    但快约 10 倍，且不再分配 ``(N,k,3)`` 那个巨大的临时数组
    （480x254 网格 × 32 色时是 48 MB）。

    实测（480x254，32 色）：96 ms → 9 ms。
    """
    shape = arr.shape
    flat = arr.reshape(-1, 3)
    pal = np.asarray(palette, dtype=np.float32)
    if flat.shape[0] == 0:
        return np.empty(shape, dtype=np.float32)
    if pal.shape[0] == 1:
        return np.broadcast_to(pal[0], shape).astype(np.float32)

    # 用 float64 做点积再比较：float32 下 |p|² 与 2·x·p 可能接近
    # （两者的差就是我们要比较的量），会因抵消而选错。
    return pal[_sq_dist(flat, pal).argmin(axis=1)].reshape(shape)


def build_lut(palette: np.ndarray, levels: int = 64) -> np.ndarray:
    """预计算 3D LUT：每个 RGB 网格中心 → 最近色板索引。(levels,levels,levels) int16。"""
    g = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    d = ((grid[:, None, :] - palette[None, :, :].astype(np.float32)) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(np.int16).reshape(levels, levels, levels)


def snap_lut(arr: np.ndarray, palette: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """用 LUT 近似吸附（比 exact 快一个量级，代价是网格边界处最多偏一个色阶）。"""
    lv = lut.shape[0]
    i = np.clip((np.clip(arr, 0.0, 1.0) * lv).astype(np.int32), 0, lv - 1)
    return palette[lut[i[..., 0], i[..., 1], i[..., 2]]]
