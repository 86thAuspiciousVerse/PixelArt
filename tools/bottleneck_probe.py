"""瓶颈归因：把"预览很卡"拆成【渲染】/【编码】/【传输】/【往返】四块。

为什么要先量：用户的感受是"卡"，但"卡"可能是四件完全不同的事，
而它们对应的解法完全不同 ——

  渲染慢  → 换 GPU（但 240 网格只有 3 万像素，GPU 的收益可能被启动开销吃掉）
  编码慢  → 每帧 PNG 压缩，纯 CPU，与 GPU 无关
  传输慢  → 每帧几十 KB 过 HTTP，是"往返次数 × 单帧成本"，与像素量弱相关
  往返慢  → 每次拖滑杆都要等服务端回应，是**延迟**问题不是吞吐问题

⚠️ 如果是后三者占大头，那"上显卡"就不是主要解法 ——
   真正该做的是**把逐帧运算放到浏览器里**（浏览器自带 GPU，
   而且省掉了编码与传输）。这和"装个 CUDA 版 onnxruntime"是两回事。

给每个阶段标注它能不能上 GPU：
  G  完全可并行，适合 shader（逐像素 / 逐通道）
  P  部分可并行（有小半径卷积、归约）
  S  难并行（全局排序、中位数、median cut 这类）

用法::

    python tools/bottleneck_probe.py
    python tools/bottleneck_probe.py --grid 480
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelart.palette import palette_from_frames  # noqa: E402
from pixelart.paths import INPUT  # noqa: E402
from pixelart.pipeline import (  # noqa: E402
    AnimParams,
    compose_frame,
    finish_frame,
    prepare_scene,
)
from pixelart.pixelate import PixelTail  # noqa: E402
from pixelart.tune import TuneParams, build_scene, display_scale, render_still  # noqa: E402

ASSET = "ref04_lain_room.jpg"


def tm(fn, n=5, warm=1):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def hr(title: str):
    print(f"\n{'─' * 74}\n{title}\n{'─' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--grid", type=int, default=0, help="0 = 自动（预览 240 / 成片 480）")
    args = ap.parse_args()

    img = Image.open(INPUT / ASSET)
    p = TuneParams()
    if args.grid:
        p.grid_long = int(args.grid)
        p.work_long = int(args.grid) * 4

    t0 = time.perf_counter()
    sc = build_scene(img, p)
    t_build = time.perf_counter() - t0
    gh, gw = sc.grid[1], sc.grid[0]
    lum_px = gw * gh
    n = p.preview_frames(sc, 10)

    print(f"素材 {ASSET}  网格 {gw}x{gh}（{lum_px:,} 像素）  ×{sc.scale}  "
          f"预览 {n} 帧 @10fps（{p.seconds:g}s）")
    print(f"build_scene（一次性：深度推理 + 预处理）  {t_build:.2f} s")

    # ── 一次性 vs 每帧 ──
    hr("① 「一次」与「每帧」的成本结构（网格长边 %d）" % p.grid_long)
    print(f"{'阶段':34s} {'耗时':>9s}  {'×帧数':>10s}  GPU")
    print(f"{'一 次 性（与帧数无关）':34s}")
    print(f"  {'深度推理（ONNX）':32s} {t_build * 1000:8.0f}ms {'—':>10s}  P")
    print(f"  {'预处理（降采样/色阶/锐化）':32s} {'（含在上面）':>9s} {'—':>10s}  G")
    print(f"{'每 帧（× ' + str(n) + ' 帧）':34s}")
    stages = []

    def add(tag, ms, gpu):
        stages.append((tag, ms))
        print(f"  {tag:32s} {ms:8.1f}ms {ms * n:9.0f}ms  {gpu}")

    a = AnimParams()
    base = sc.base
    fogged, content = compose_frame(sc, t=0.3, anim=a, return_content=True)
    add("depth_fog", tm(lambda: __import__("pixelart.compose", fromlist=["depth_fog"]).depth_fog(
        base, sc.far, color=sc.fog_color, density=p.density, power=p.power,
        t=0.3, drift=a.fog_drift)), "G")
    from pixelart.compose import bloom, volumetric_light
    add("volumetric_light", tm(lambda: volumetric_light(
        fogged, sc.far, sc.light_xy, strength=p.rays, t=0.3, flicker=a.light_flicker)), "P")
    add("bloom", tm(lambda: bloom(content, threshold=0.55, strength=0.425, radii=(2, 5, 11))), "P")
    add("compose_frame 合计", tm(lambda: compose_frame(sc, t=0.3, anim=a), 3), "G")

    pal = palette_from_frames([compose_frame(sc, t=i / 6, anim=a) for i in range(6)],
                              n_colors=p.colors, max_samples=6)
    add("finish_frame（像素尾巴）", tm(lambda: finish_frame(fogged, p.to_tail(), palette=pal,
                                                          edge_ref=content)), "G")
    add("render_still 合计", tm(lambda: render_still(sc, p, palette=pal), 3), "G")

    total_frame = sum(ms for _, ms in stages) - stages[-1][1] + stages[-1][1]
    real = tm(lambda: render_still(sc, p, palette=pal), 3)
    print(f"\n  单帧真实耗时（render_still）  {real:.0f} ms   = 每帧 GPU 可并行占比约 "
          f"{sum(ms for t, ms in stages if t not in ('render_still 合计',)) / max(real, 1e-9) * 100:.0f}%")

    # ── 编码与传输 ──
    hr("② 每帧的**编码 + 传输**成本（这部分与 GPU 无关，但浏览器方案能直接省掉）")
    scale = display_scale(sc)
    u8, _ = render_still(sc, p, palette=pal)
    im = Image.fromarray(np.ascontiguousarray(u8))

    t_enc = None
    for lvl, name in ((3, "compress_level=3（当前）"), (6, "compress_level=6")):
        def enc():
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=False, compress_level=lvl)
            return buf.getvalue()
        b = enc()
        ms = tm(enc, 5)
        if t_enc is None:
            t_enc, nbytes = ms, len(b)
        print(f"  服务端 PNG 编码 {name:24s} {ms:7.1f}ms   {len(b) / 1024:7.1f} KB（网格分辨率）")

    def enc_up():
        buf = io.BytesIO()
        up = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
        up.save(buf, format="PNG", optimize=False, compress_level=3)
        return buf.getvalue()
    b_up = enc_up()
    ms_up = tm(enc_up, 5)
    print(f"  ＋最近邻放大 ×{scale}（给浏览器显示）     {ms_up:7.1f}ms   {len(b_up) / 1024:7.1f} KB")
    print(f"  ＋HTTP 传输（本机回环，忽略）")
    print(f"  → 单帧「服务端到浏览器」的总成本 ≈ {t_enc + ms_up:.0f} ms + 传输")

    # ── HTTP 往返 ──
    hr("③ 拖一次滑杆的**往返延迟**（感知「卡」的主要来源）")
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_q = (f"asset={ASSET}&grid_long={p.grid_long}&work_long={p.work_long}"
              f"&aspect=native&seconds={p.seconds:g}&fps={p.fps}")
    try:
        url = f"http://127.0.0.1:{args.port}/api/still?{base_q}&t=0&scale={scale}"
        with op.open(url, timeout=60) as r:
            r.read()
        # 换 t 值绕过缓存，模拟"拖动中"
        ts = [0.11, 0.22, 0.33, 0.44, 0.55]
        lats = []
        for t in ts:
            t0 = time.perf_counter()
            with op.open(url.replace("t=0", f"t={t}"), timeout=120) as r:
                r.read()
            lats.append((time.perf_counter() - t0) * 1000)
        print(f"  未命中缓存的一次 /api/still 往返        {np.mean(lats):7.0f} ms   "
              f"（{min(lats):.0f}~{max(lats):.0f}）")
        print(f"  界面里滑杆节流间隔                     {60:7.0f} ms")
        print(f"  → **拖动时的刷新率 ≈ {1000 / max(np.mean(lats), 1e-9):.1f} fps**"
              f"（人眼要 24fps 以上才觉得跟手）")
    except Exception as e:                              # noqa: BLE001
        print(f"  服务未运行（{type(e).__name__}）—— 跳过。"
              f"先跑 tools/m3_server.py 再测这一节。")

    # ── 结论 ──
    hr("④ 结论")
    ret = np.mean(lats) if 'lats' in dir() and lats else None
    print(f"  单帧渲染（纯计算，可上 GPU）        {real:6.0f} ms")
    print(f"  单帧编码 + 放大（纯 CPU，与 GPU 无关）{t_enc + ms_up:6.0f} ms")
    if ret:
        print(f"  单次往返总延迟                       {ret:6.0f} ms")
        share = (t_enc + ms_up) / max(ret, 1e-9) * 100
        print(f"  其中「编码+放大」占比                 {share:5.0f}%")
        print(f"  其中「渲染」占比                       {real / max(ret, 1e-9) * 100:5.0f}%")
    print()
    print("  ⚠️ 判读要点：**240~480 网格只有 3~12 万像素**。")
    print("     这个规模上，GPU 的算力收益会被「内核启动 + 数据搬运」开销吃掉一大半，")
    print("     所以「换个 GPU 后端」的加速比不会像大图那样夸张。")
    print("     真正的收益来自**把整条每帧链路搬进浏览器**：")
    print("     ① 省掉每帧的服务端编码（上面这一项）")
    print("     ② 省掉每帧的 HTTP 往返与解码（往返延迟直接消失）")
    print("     ③ 逐帧在 GPU 上流水化，不再有 Python 侧的分配与拷贝")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
