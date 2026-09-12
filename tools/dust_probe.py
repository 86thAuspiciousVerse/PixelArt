"""尘埃粒子诊断 —— 为什么会出现"黑色小点在移动"？

用户报障：光照在动是对的，但画面上有**黑点**在移动，像蚊子/蚂蚁在直线爬行。

⚠️ 关键疑点：`dust_layer` 返回的图层是**加性**的（只会让画面变亮），
   白色粒子**不可能**直接产生黑点。所以黑点一定是**别的步骤被粒子触发了**。

按嫌疑顺序逐个证伪/证实：

  H1  边缘压暗误触发（首要嫌疑）
      粒子是"单个亮像素 + 周围暗像素"，在边缘检测器眼里就是**极强边界**。
      边缘压暗会把 `g` 饱和到 1.0，于是粒子**自己与相邻像素**被乘上
      (1 - 0.35) → 变暗。这和约束 5（从抖动结果求梯度导致整幅压暗 32%）
      是**同一类机制的另一种表现**。
  H2  bloom（只会变亮，理论上不可能产生黑点，但要排除）
  H3  色板被粒子污染（`palette_from_frames` 把粒子的亮色当成重要色 → 挤掉别的色阶）
  H4  量化把"亮了一点"的像素映射到更暗的色板项（最近色不保证保序 → 可能）

量法：同一 t 下渲两遍（有粒子 / 无粒子），**用同一块色板**量化，逐像素比较。
哪些像素因为粒子而变暗、变暗几个色阶 —— 这是用户真正看得见的东西。
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
from pixelart.animate import dust_layer  # noqa: E402
from pixelart.pipeline import AnimParams  # noqa: E402
from pixelart.compose import bloom  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.palette import palette_from_frames  # noqa: E402
from pixelart.pipeline import (  # noqa: E402
    compose_frame,
    finish_frame,
    prepare_scene,
)
from pixelart.pixelate import PixelTail  # noqa: E402


def luma(a: np.ndarray) -> np.ndarray:
    return a @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--aspect", default="native")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dust", type=int, default=200)
    ap.add_argument("--tag", default="dust")
    args = ap.parse_args()

    aspect = 0.0 if args.aspect.strip().lower() in ("native", "keep", "none") else float(args.aspect)
    scene = prepare_scene(Image.open(args.src), aspect=aspect,
                          tail=PixelTail(), depth_estimator=None
                          if False else __import__("pixelart.analyze", fromlist=["DepthEstimator"]).DepthEstimator())
    gh, gw = scene.grid[1], scene.grid[0]

    a_on = AnimParams(fog_drift=0.30, light_flicker=0.16, dust_count=args.dust)
    a_off = AnimParams(fog_drift=0.30, light_flicker=0.16, dust_count=0)

    ts = [i / args.n for i in range(args.n)]
    on_c, on_ct = zip(*[compose_frame(scene, t=t, anim=a_on, return_content=True) for t in ts])
    off_c, off_ct = zip(*[compose_frame(scene, t=t, anim=a_off, return_content=True) for t in ts])
    on, off = list(on_c), list(off_c)
    on_ct, off_ct = list(on_ct), list(off_ct)

    # 生产路径的色板（含粒子）
    pal = palette_from_frames(on, n_colors=scene.tail.n_colors, max_samples=8)

    print(f"网格 {gw}x{gh}   粒子数 {args.dust}   帧数 {args.n}")
    print("\n=== 粒子的直接效果（合成阶段，量化前）===")
    d = on[0] - off[0]
    dl = luma(d)
    print(f"  合成阶段变暗的像素数: {int((dl < -1e-6).sum())}   "
          f"最暗 {float(dl.min()):+.4f}")
    print(f"  合成阶段变亮的像素数: {int((dl > 1e-6).sum())}   最亮 {float(dl.max()):+.4f}")
    print("  → 合成阶段粒子是**纯加性**的（不可能变暗）。")

    def scan(edge_ref_mode: str, tail: PixelTail):
        """edge_ref_mode: 'none'（旧行为，用含粒子的图）| 'content'（修复后）"""
        tot = {2: 0, 4: 0, 8: 0}
        worst = 0.0
        for i in range(args.n):
            er = None if edge_ref_mode == "none" else on_ct[i]
            ero = None if edge_ref_mode == "none" else off_ct[i]
            u_on, _ = finish_frame(on[i], tail, palette=pal, edge_ref=er)
            u_off, _ = finish_frame(off[i], tail, palette=pal, edge_ref=ero)
            dd = luma(u_on.astype(np.float32)) - luma(u_off.astype(np.float32))
            for k in tot:
                tot[k] += int((dd <= -k).sum())
            worst = min(worst, float(dd.min()))
        return tot, worst

    print("\n=== 量化之后：粒子是否让像素变暗（核心指标）===")
    for mode, label in (("none", "旧行为（边缘参考 = 含粒子的图）"),
                        ("content", "修复后（边缘参考 = 不含粒子的图）")):
        tot, worst = scan(mode, scene.tail)
        print(f"  {label}")
        for k in sorted(tot):
            print(f"      变暗 >= {k:2d} 色阶: {tot[k]:7d} 像素")
        print(f"      单帧最大变暗 {worst:+.1f} 色阶")

    # ---- H1：关掉边缘压暗再测（证伪用）----
    print("\n=== H1 验证：关掉边缘压暗（edge_strength=0）===")
    tot, worst = scan("none", PixelTail(edge_strength=0.0))
    for k in sorted(tot):
        print(f"  变暗 >= {k:2d} 色阶: {tot[k]:7d} 像素   （应≈0，证明是边缘压暗的锅）")

    # ---- H3：色板有没有被粒子污染 ----
    print("\n=== H3 验证：色板是否被粒子污染 ===")
    pal_off = palette_from_frames(off, n_colors=scene.tail.n_colors, max_samples=8)
    from pixelart.palette import from_perceptual  # noqa: E402

    def pal_lum(p):
        return float((from_perceptual(p) @ np.array([0.2126, 0.7152, 0.0722], np.float32)).max())

    print(f"  含粒子色板最亮 {pal_lum(pal):.3f}；无粒子色板最亮 {pal_lum(pal_off):.3f}"
          f"  → 差 {pal_lum(pal) - pal_lum(pal_off):+.4f}（≈0 说明没污染）")

    # ---- 可视化：把"因粒子变暗"的位置画出来 ----
    print("\n=== 可视化 ===")
    k = scene.scale
    base_u8, _ = finish_frame(on[0], scene.tail, palette=pal, edge_ref=on_ct[0])
    base_img = np.asarray(Image.fromarray(base_u8).resize((gw * k, gh * k), Image.Resampling.NEAREST))
    heat = np.zeros((gh, gw), np.float32)
    for i in range(args.n):
        u_on, _ = finish_frame(on[i], scene.tail, palette=pal, edge_ref=None)
        u_off, _ = finish_frame(off[i], scene.tail, palette=pal, edge_ref=None)
        dd = luma(u_off.astype(np.float32)) - luma(u_on.astype(np.float32))
        heat = np.maximum(heat, np.clip(dd, 0, None))
    vis = np.clip(heat * 30, 0, 255).astype(np.uint8)
    heat_img = np.stack([np.zeros_like(vis), vis, np.zeros_like(vis)], -1)   # 绿 = 变暗
    heat_up = Image.fromarray(heat_img).resize((gw * k, gh * k), Image.Resampling.NEAREST)

    sheet = Image.new("RGB", (base_img.shape[1] + heat_up.width + 30, base_img.shape[0] + 50), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    sheet.paste(Image.fromarray(base_img), (10, 42))
    sheet.paste(heat_up, (base_img.shape[1] + 20, 42))
    d.text((12, 12), f"修复后的成片（放大 x{k}）", font=C.font(22), fill=C.LABEL_FG)
    d.text((base_img.shape[1] + 22, 12),
           f"旧行为下因粒子变暗的量 x30（绿）max={heat.max():.0f} 色阶", font=C.font(22), fill=C.LABEL_FG)
    C.save(sheet, out("m2", f"{args.tag}_darkdiag.png"))
    print(f"[ok] out/m2/{args.tag}_darkdiag.png")
    return 0


def snap_idx(src: np.ndarray, pal: np.ndarray) -> np.ndarray:
    flat = src.reshape(-1, 3).astype(np.float64)
    p = np.asarray(pal, np.float64)
    d = (p * p).sum(1)[None, :] - 2.0 * (flat @ p.T) + (flat * flat).sum(1)[:, None]
    return d.argmin(axis=1).reshape(src.shape[:2])


if __name__ == "__main__":
    raise SystemExit(main())
