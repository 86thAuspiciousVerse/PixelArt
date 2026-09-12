"""M2 性能剖析：每帧各阶段耗时，找热点。

目的：HANDOFF 里给的预算是「单帧约 150ms」，实测 477ms —— 慢 3 倍。
不先量就优化是瞎猜（本项目的老教训）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelart.animate import dust_layer  # noqa: E402
from pixelart.compose import bloom, depth_fog, volumetric_light  # noqa: E402
from pixelart.palette import palette_from_frames, to_perceptual  # noqa: E402
from pixelart.pipeline import AnimParams, compose_frame, prepare_scene  # noqa: E402
from pixelart.pixelate import pixelate  # noqa: E402


def tm(fn, n=5):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


img = Image.open("assets/input/ref04_lain_room.jpg")
sc = prepare_scene(img, grid_long=480, work_long=1920)
a = AnimParams()
gh, gw = sc.grid[1], sc.grid[0]
print(f"网格 {gw}x{gh}  输出 {sc.out_size}\n")

print(f"{'阶段':26s} {'耗时':>9s}  说明")
print("-" * 70)
fogged = depth_fog(sc.base, sc.far, color=sc.fog_color, density=0.7, power=3.0, t=0.3, drift=a.fog_drift)
print(f"{'depth_fog':26s} {tm(lambda: depth_fog(sc.base, sc.far, color=sc.fog_color, density=0.7, power=3.0, t=0.3, drift=a.drift if False else a.fog_drift)):8.1f} ms")

vol = volumetric_light(fogged, sc.far, sc.light_xy, strength=0.55, t=0.3, flicker=a.light_flicker)
print(f"{'volumetric_light':26s} {tm(lambda: volumetric_light(fogged, sc.far, sc.light_xy, strength=0.55, t=0.3, flicker=a.light_flicker)):8.1f} ms")

lit = np.clip(fogged + vol, 0, 1)
dust = dust_layer((gh, gw), sc.far, t=0.3, count=a.dust_count, seed=11, light_xy=sc.light_xy)
print(f"{'dust_layer':26s} {tm(lambda: dust_layer((gh, gw), sc.far, t=0.3, count=a.dust_count, seed=11, light_xy=sc.light_xy)):8.1f} ms")

lit2 = np.clip(lit + dust * a.dust_bright, 0, 1)
comp = bloom(lit2, threshold=0.55, strength=0.425, radii=(2, 5, 11))
print(f"{'bloom':26s} {tm(lambda: bloom(lit2, threshold=0.55, strength=0.425, radii=(2, 5, 11))):8.1f} ms")

print(f"{'pixelate(自动色板)':26s} {tm(lambda: pixelate(comp, sc.tail, palette=None), 3):8.1f} ms  ← 含 refine_palette")
pal = palette_from_frames([comp], n_colors=32)
print(f"{'pixelate(给定色板)':26s} {tm(lambda: pixelate(comp, sc.tail, palette=pal)):8.1f} ms  ← 视频走的路径")
print(f"{'to_perceptual':26s} {tm(lambda: to_perceptual(comp)):8.1f} ms")
print("-" * 70)
print(f"{'compose_frame 合计':26s} {tm(lambda: compose_frame(sc, t=0.3, anim=a), 3):8.1f} ms")
