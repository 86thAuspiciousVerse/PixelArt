"""诊断用户反馈的三个问题。

一、「一张照片渲染要 8~10 秒」—— 我的瓶颈报告说的是单帧 45ms，
    但界面实际要渲**整段 60 帧**，所以是两个不同的数。这里把两条路径都量出来：
      A. 拖一次滑杆（单帧）
      B. 点「渲染预览」（整段 60 帧 + 逐帧取图）
    并拆出"服务端渲染 / 编码 / 逐帧 HTTP"各占多少。

二、「ref10 房子变绿又出现了」—— 先确认是否复现，再定位是哪一步。
    重点怀疑：M3 的预览网格是 240（成片是 480），
    换分辨率会不会让色板修正失效？

三、「画面像被上了一层全局颜色滤镜」—— 量体积光图层的**空间分布**：
    如果它的贡献在远离光源处仍然可观，那"全局染色"就是结构性的。
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.compose import depth_fog, volumetric_light  # noqa: E402
from pixelart.palette import LUMA, from_perceptual, palette_from_image, refine_palette  # noqa: E402
from pixelart.pipeline import AnimParams, compose_frame, finish_frame, prepare_scene  # noqa: E402
from pixelart.pixelate import PixelTail  # noqa: E402
from pixelart.tune import TuneParams, build_scene  # noqa: E402



def hr(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def sat_of(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(axis=0)
    return float((c.max() - c.min()) / max(c.max(), 1e-6))


def green_of(c):
    c = np.asarray(c, np.float32).reshape(-1, 3).mean(axis=0)
    return float(c[1] - 0.5 * (c[0] + c[2]))


# ══════════════════════════════════════════════════════════════
# 一、8~10 秒花在哪
# ══════════════════════════════════════════════════════════════
hr("一、渲染耗时：单帧 vs 整段（用户说 8~10 秒）")

for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
    os.environ.pop(k, None)
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
B = "http://127.0.0.1:8770"
ASSET = "ref04_lain_room.jpg"


def http_json(path, timeout=300.0):
    with OP.open(B + path, timeout=timeout) as r:
        import json
        return json.loads(r.read())


def http_bytes(path, timeout=300.0):
    with OP.open(B + path, timeout=timeout) as r:
        return r.read()


try:
    http_json("/api/assets")
    online = True
except Exception as e:                                    # noqa: BLE001
    online = False
    print(f"  （服务未运行：{type(e).__name__}）")

P = TuneParams()
Q = (f"asset={ASSET}&grid_long={P.grid_long}&work_long={P.work_long}&aspect=native"
     f"&seconds={P.seconds:g}&fps={P.fps}")

if online:
    t0 = time.perf_counter()
    st = http_json(f"/api/preview?{Q}&preview_fps=10")
    t_preview = time.perf_counter() - t0
    n = st["n_frames"]
    print(f"  A. 点「渲染预览」：服务端渲 {n} 帧 + 1 自检帧 → {t_preview:.2f} s")

    # 逐帧取图（界面就是这样把帧拉到浏览器里的）
    t0 = time.perf_counter()
    tot = 0
    for i in range(n):
        tot += len(http_bytes(f"/api/frame?{Q}&preview_fps=10&i={i}&scale={st['display_scale']}"))
    t_frames = time.perf_counter() - t0
    print(f"  B. 逐帧取图 {n} 次 HTTP（界面预载全部帧）  → {t_frames:.2f} s   "
          f"（{tot / 1024 / 1024:.1f} MB，平均 {t_frames / n * 1000:.0f} ms/次）")
    t0 = time.perf_counter()
    http_bytes(f"/api/still?{Q}&t=0.5&scale={st['display_scale']}")
    t_one = time.perf_counter() - t0
    print(f"  C. 单帧（拖滑杆，未命中缓存）            → {t_one * 1000:.0f} ms")
    t0 = time.perf_counter()
    http_bytes(f"/api/still?{Q}&t=0.5&scale={st['display_scale']}")
    t_one2 = time.perf_counter() - t0
    print(f"  D. 单帧（命中缓存）                      → {t_one2 * 1000:.0f} ms")

    print(f"\n  → 用户体感的 8~10 秒 = A + B ≈ {t_preview + t_frames:.1f} s")
    print(f"    其中**逐帧 HTTP 占 {t_frames / (t_preview + t_frames) * 100:.0f}%**")
    print(f"    ⚠️ 我先前报告的「约 67ms/帧」是**单帧**的数，"
          f"而界面一次要拉 {n} 帧 —— 两个数不矛盾，但我没把整段成本说清楚。")
else:
    print("  服务未运行，跳过 HTTP 部分。先跑 tools/m3_server.py。")

# ══════════════════════════════════════════════════════════════
# 二、ref10 变绿是否复现
# ══════════════════════════════════════════════════════════════
hr("二、ref10 房子变绿：在不同网格下复现吗？")

img10 = Image.open(INPUT / "ref10_green_cliff.jpg")
print(f"{'网格':>8s} {'建筑区平均色':>14s} {'饱和度':>8s} {'偏绿':>8s} {'色板中性项':>12s}")
print("-" * 62)
for grid in (120, 240, 320, 480):
    p = TuneParams(grid_long=grid, work_long=grid * 4)
    sc = build_scene(img10, p)
    u8, pal = __import__("pixelart.tune", fromlist=["render_still"]).render_still(sc, p)
    lin = np.asarray(u8, np.float32) / 255.0

    lum = sc.base @ LUMA
    mx, mn = sc.base.max(-1), sc.base.min(-1)
    M = (lum >= np.percentile(lum, 88)) & ((mx - mn) / np.clip(mx, 1e-6, None) <= 0.20)
    if M.sum() < 20:
        print(f"{grid:8d}  (建筑区过小，跳过)")
        continue
    c = lin[M].mean(axis=0)
    cp = from_perceptual(pal)
    psat = (cp.max(1) - cp.min(1)) / np.clip(cp.max(1), 1e-6, None)
    hexs = "#%02x%02x%02x" % tuple(np.rint(c * 255).astype(int))
    print(f"{grid:8d} {hexs:>14s} {sat_of(lin[M]):8.3f} {green_of(lin[M]):+8.3f} "
          f"{int((psat <= 0.12).sum()):12d}")

print("\n对照：真实建筑区颜色（预处理后，未量化）")
for grid in (240, 480):
    p = TuneParams(grid_long=grid, work_long=grid * 4)
    sc = build_scene(img10, p)
    lum = sc.base @ LUMA
    mx, mn = sc.base.max(-1), sc.base.min(-1)
    M = (lum >= np.percentile(lum, 88)) & ((mx - mn) / np.clip(mx, 1e-6, None) <= 0.20)
    c = sc.base[M].mean(axis=0)
    hexs = "#%02x%02x%02x" % tuple(np.rint(np.clip(c, 0, 1) * 255).astype(int))
    print(f"  网格 {grid:3d}: {hexs}  饱和度 {sat_of(sc.base[M]):.3f}  "
          f"偏绿 {green_of(sc.base[M]):+.3f}   （这是「正确」的目标）")

# ══════════════════════════════════════════════════════════════
# 三、画面像「全局颜色滤镜」：体积光的空间分布
# ══════════════════════════════════════════════════════════════
hr("三、体积光到底是「局部光束」还是「全局染色」？")

p = TuneParams()
sc = build_scene(img10, p)
fogged = depth_fog(sc.base, sc.far, color=sc.fog_color, density=p.density, power=p.power)
vol = volumetric_light(fogged, sc.far, sc.light_xy, strength=p.rays,
                       air_sat_max=p.rays_sat)
gh, gw = sc.grid[1], sc.grid[0]
yy, xx = np.mgrid[0:gh, 0:gw]
d = np.hypot(xx / gw - sc.light_xy[0], yy / gh - sc.light_xy[1])

v = vol.sum(axis=2)
print(f"  光源位置 {[round(x, 2) for x in sc.light_xy]}   注入总量 {v.mean():.5f}")
print(f"\n  {'离光源距离':>12s} {'占画面':>8s} {'该区域平均注入':>14s} {'相对最近处':>12s}")
print("  " + "-" * 52)
near = v[d < 0.1].mean()
for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.5)):
    m = (d >= lo) & (d < hi)
    if m.sum() == 0:
        continue
    val = v[m].mean()
    print(f"  {f'{lo:.2f}~{hi:.2f}':>12s} {m.mean() * 100:7.1f}% {val:14.5f} "
          f"{val / max(near, 1e-9) * 100:11.0f}%")

print(f"\n  ⚠️ 判读：如果远处仍有最近处的 30% 以上，那「全局染色」就是**结构性的** ——")
print(f"     因为 falloff = 1/(1 + gain·|z−z_light|·4) 只按**深度差**衰减，")
print(f"     完全不看**屏幕距离**。所以只要画面里大部分像素与光源深度相近，")
print(f"     它们全都会均匀地吃到同一层带色的光 —— 观感就是蒙了一层滤镜。")
