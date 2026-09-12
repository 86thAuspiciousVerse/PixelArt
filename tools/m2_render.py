"""M2 —— 渲染无缝循环像素视频（以及同源的静态图）。

这是产品定义里的「输出 B」。用它回答两个问题：

1. 跨帧色板共享、抖动屏幕空间锁定这两条铁律，在真实画面上到底成不成立？
2. 动起来的观感对不对（雾在漂、光在闪、尘埃在浮）？

⭐ **静态图与视频共用同一条管线**：``t`` 取一个值就是静态图，扫过 [0,1) 就是视频。
所以 ``--still`` 产出的 PNG 和视频的第 0 帧在**同色板**下是逐位一致的。

用法::

    # 6 秒 30fps 竖构图（默认无损动画 WebP）
    python tools/m2_render.py --src assets/input/ref04_lain_room.jpg --aspect native --seconds 6

    # 先只做循环自检 + 出一张帧条预览图，不编码视频（快）
    python tools/m2_render.py --src assets/input/ref04_lain_room.jpg --check-only

    # 关掉动画幅度做对照（应当退化成静态图）
    python tools/m2_render.py --src assets/input/ref04_lain_room.jpg --no-anim --check-only

    # 只出静态 PNG（和视频第 0 帧同色板）
    python tools/m2_render.py --src assets/input/ref04_lain_room.jpg --still-only

⚠️ 输出编码：默认**无损动画 WebP**。MP4 默认的 yuv420p 会把色度减半，
在像素方块边界糊出错误颜色（高饱和素材实测最大误差 127/255），
而且解码后不再是色板精确的 —— 而"有限色板"正是像素画的定义。
要 MP4 必须显式 ``-pix_fmt yuv444p``（见 encode.save_mp4_hint）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.encode import save_animation, save_still  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.pipeline import (  # noqa: E402
    AnimParams,
    check_loop,
    compose_frame,
    finish_frame,
    grid_from_aspect,
    mean_abs_diff,
    prepare_scene,
    render_video,
    upscale,
)
from pixelart.pixelate import PixelTail  # noqa: E402


def parse_aspect(s: str) -> float:
    s = s.strip().lower()
    if s in ("native", "keep", "none"):
        return 0.0
    if ":" in s:
        a, b = s.split(":", 1)
        return float(a) / float(b)
    return float(s)


def frame_strip(frames: list[np.ndarray], cols: int = 6, rows: int = 2,
                cell: int = 300) -> Image.Image:
    """把等间隔抽出的帧拼成一张对比图 —— 不用播放器也能看出"动没动"。"""
    n = cols * rows
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    tiles = []
    for i in idx:
        im = Image.fromarray(frames[int(i)])
        k = max(1, cell // max(im.width, 1))
        tiles.append(im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST))

    cw = max(t.width for t in tiles) + 16
    ch = max(t.height for t in tiles) + 40
    sheet = Image.new("RGB", (cw * cols, ch * rows), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    for j, (i, t) in enumerate(zip(idx, tiles)):
        bx, by = (j % cols) * cw, (j // cols) * ch
        sheet.paste(t, (bx + 8, by + 34))
        d.text((bx + 10, by + 8), f"frame {int(i)}", font=C.font(22), fill=C.LABEL_FG)
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INPUT / "ref04_lain_room.jpg"))
    ap.add_argument("--aspect", default="16:9", help="'native' 保持原比例，或 16:9 / 2.16")
    ap.add_argument("--grid-long", type=int, default=480)
    ap.add_argument("--work-long", type=int, default=1920)
    ap.add_argument("--colors", type=int, default=32)
    # 时长
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    # 合成
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--power", type=float, default=3.0)
    ap.add_argument("--rays", type=float, default=0.55)
    ap.add_argument("--rays-color", default="auto")
    ap.add_argument("--rays-sat", type=float, default=0.25)
    ap.add_argument("--bloom", type=float, default=0.85)
    # 动画幅度
    ap.add_argument("--fog-drift", type=float, default=0.30)
    ap.add_argument("--flicker", type=float, default=0.16)
    ap.add_argument("--dust", type=int, default=200)
    ap.add_argument("--dust-bright", type=float, default=0.55)
    ap.add_argument("--no-anim", action="store_true", help="动画幅度全关（应退化为静态图）")
    # 行为
    ap.add_argument("--check-only", action="store_true", help="只做自检 + 帧条预览，不编码")
    ap.add_argument("--still-only", action="store_true", help="只出静态 PNG")
    ap.add_argument("--fmt", default="webp", choices=["webp", "apng", "gif"])
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    src = Image.open(args.src).convert("RGB")
    grid = grid_from_aspect(src.width, src.height, args.grid_long)
    key = args.tag or Path(args.src).stem

    anim = (AnimParams(fog_drift=0.0, light_flicker=0.0, dust_count=0)
            if args.no_anim else
            AnimParams(fog_drift=args.fog_drift, light_flicker=args.flicker,
                       dust_count=args.dust, dust_bright=args.dust_bright))

    print(f"原图 {src.width}x{src.height}  网格 {grid[0]}x{grid[1]}  "
          f"aspect={args.aspect}  动画={'关' if args.no_anim else '开'}")

    t0 = time.perf_counter()
    scene = prepare_scene(
        src, grid_long=args.grid_long, work_long=args.work_long,
        aspect=parse_aspect(args.aspect), tail=PixelTail(n_colors=args.colors),
        depth_estimator=DepthEstimator(),
    )
    t_prep = time.perf_counter() - t0
    print(f"  [一次] 分析+预处理 {t_prep:.2f}s   输出 {scene.out_size[0]}x{scene.out_size[1]}"
          f"  x{scene.scale}")
    print(f"         深度分位 {[round(v, 2) for v in scene.stats['depth_q']]}"
          f"  光源 ({scene.light_xy[0]:.2f}, {scene.light_xy[1]:.2f})")

    rc = None if args.rays_color.strip().lower() in ("auto", "none", "") else \
        tuple(float(v) for v in args.rays_color.split(","))

    if args.still_only:
        c, ct = compose_frame(scene, t=0.0, density=args.density, power=args.power,
                              rays=args.rays, bloom_strength=args.bloom, anim=anim,
                              rays_color=rc, rays_sat=args.rays_sat, return_content=True)
        u8, pal = finish_frame(c, scene.tail, palette=None, edge_ref=ct)
        p = save_still(upscale(u8, scene.scale), out("m2", f"{key}_still.png"))
        print(f"[ok] {p}   ({scene.out_size[0]}x{scene.out_size[1]}, 色板 {len(pal)} 色)")
        return 0

    n_frames = max(2, int(round(args.seconds * args.fps)))
    print(f"  [每帧] 合成 {n_frames} 帧 + 1 帧自检 ...")

    def prog(i, n):
        if i == n or i % max(1, n // 8) == 0:
            print(f"         {i}/{n}", flush=True)

    t0 = time.perf_counter()
    frames, palette, stats = render_video(
        scene, n_frames=n_frames, anim=anim, density=args.density, power=args.power,
        rays=args.rays, bloom_strength=args.bloom, rays_color=rc, rays_sat=args.rays_sat,
        include_last=True, on_progress=prog,
    )
    t_frames = time.perf_counter() - t0
    print(f"  [每帧] {t_frames:.1f}s  ({t_frames / n_frames * 1000:.0f} ms/帧)   "
          f"色板 {stats['palette_size']} 色")

    # ---- 自检 ----
    print("\n── 自检 ──")
    loop = stats["loop_max_diff"]
    print(f"  循环无缝 frame(0)==frame(N): {'✅ 通过' if stats['loop_ok'] else '❌ 未通过'}"
          f"  最大通道差 {loop}  （0 = 逐位相同）")
    moves = [mean_abs_diff(frames[i], frames[(i + 1) % len(frames)])
             for i in range(0, len(frames), max(1, len(frames) // 8))]
    print(f"  really moving 相邻帧平均差 min/mean/max = "
          f"{min(moves):.3f}/{np.mean(moves):.3f}/{max(moves):.3f}")
    far_apart = mean_abs_diff(frames[0], frames[len(frames) // 2])
    print(f"  半周期帧差 {far_apart:.3f}  （太小说明动画幅度不够）")

    body = frames[:n_frames]
    uniq = set()
    for f in body[::max(1, len(body) // 12)]:
        uniq |= set(map(tuple, f.reshape(-1, 3)[::53].tolist()))
    # ⚠️ 必须用和 pixelate 完全相同的换算（astype 是**截断**不是四舍五入），
    #    否则会误报"越界"。
    from pixelart.palette import from_perceptual  # noqa: E402
    pal_u8 = {tuple(int(v) for v in row) for row in
              np.clip(from_perceptual(palette) * 255.0, 0, 255).astype(np.uint8)}
    outside = uniq - pal_u8
    print(f"  全片颜色全部来自色板: {'✅' if not outside else '❌'}  "
          f"（画面 {len(uniq)} 色，色板 {len(pal_u8)} 色，越界 {len(outside)}）")

    strip = frame_strip(body)
    C.save(strip, out("m2", f"{key}_strip.png"))
    print(f"  [ok] out/m2/{key}_strip.png  （帧条预览）")

    if args.check_only:
        print("\n（--check-only：跳过编码）")
        return 0

    # ---- 编码 ----
    big = [upscale(f, scene.scale) for f in body]
    p = save_animation(big, out("m2", f"{key}_loop.{args.fmt}"),
                       duration_ms=int(round(1000 / args.fps)), lossless=True)
    size_mb = p.stat().st_size / 1e6
    print(f"\n[ok] {p}")
    print(f"     {scene.out_size[0]}x{scene.out_size[1]}  {args.fps}fps  "
          f"{args.seconds:.1f}s  {n_frames} 帧  无损   {size_mb:.1f} MB")

    # 静态图与视频第 0 帧：同色板 + 同 edge_ref 下必须逐位一致
    c0, ct0 = compose_frame(scene, t=0.0, density=args.density, power=args.power,
                            rays=args.rays, bloom_strength=args.bloom, anim=anim,
                            rays_color=rc, rays_sat=args.rays_sat, return_content=True)
    u8, _ = finish_frame(c0, scene.tail, palette=palette, edge_ref=ct0)
    same = np.array_equal(u8, body[0])
    ps = save_still(upscale(u8, scene.scale), out("m2", f"{key}_still.png"))
    print(f"[ok] {ps}")
    print(f"     静态图 == 视频第 0 帧: {'✅ 逐位相同' if same else '❌ 不一致'}")
    print(f"\n提示：{Path(p).name} 是**无损动画 WebP**，可直接在浏览器/图片查看器里循环播放。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
