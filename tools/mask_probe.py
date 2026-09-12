"""光源检测诊断图：看清"自动光源落在哪、哪里的亮是伪装"。

用途：设计语义掩膜前先看证据 —— ref10（落在草丛）与 ref02（落在地面反光）
是 HANDOFF 记录的两个已知失败；ref04 是正常对照。

产物（out/m5/）：每素材一张三联图
  左：base 画面 + 当前自动光源（十字）
  中：亮度图（blur 后）+ 90/95/99 分位等高线
  右：深度图（far，0 近 1 远）+ 自动光源位置

用法： python tools/mask_probe.py [素材名 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw

from pixelart.compose import brightest_center, blur
from pixelart.masks import detect_light_source
from pixelart.tune import build_scene

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def colorize(far: np.ndarray) -> np.ndarray:
    x = np.clip(far, 0, 1)
    return (np.stack([x * 0.2, x * 0.6, x], axis=-1) * 255).astype(np.uint8)


def main(argv: list[str]) -> int:
    names = argv[1:] or ["ref10_green_cliff.jpg", "ref02_replace_coffeeshop.jpg",
                         "ref04_lain_room.jpg"]
    out_dir = ROOT / "out" / "m5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        p = (ROOT / "assets" / "input" / name)
        if not p.exists():
            print(f"  跳过（无 {name}）")
            continue
        scene = build_scene(Image.open(p), __import__("pixelart.tune", fromlist=["TuneParams"]).TuneParams.from_query(
            {"asset": [name], "grid_long": ["240"], "work_long": ["960"],
             "aspect": ["native"]}))
        base = np.clip(scene.base, 0, 1)
        far = scene.far
        h, w = far.shape
        lum = blur((base @ LUMA)[..., None], 3.0)[..., 0]

        lx, ly = scene.light_xy
        lpx, lpy = int(lx * (w - 1)), int(ly * (h - 1))
        nx, ny = detect_light_source(base, far)      # 掩膜过滤的新检测
        npx, npy = int(nx * (w - 1)), int(ny * (h - 1))

        # 三联：base+光源 / 亮度+分位线 / 深度
        p90, p95, p99 = np.percentile(lum, [90, 95, 99])
        mid = (np.stack([lum, lum, lum], -1) * 255).astype(np.uint8)
        vis = mid.copy()
        vis[np.abs(lum - p90) < 0.004] = (40, 120, 250)
        vis[np.abs(lum - p95) < 0.004] = (250, 160, 40)
        vis[np.abs(lum - p99) < 0.004] = (250, 60, 60)
        dep = colorize(far)
        dep[lpy, lpx] = (255, 255, 0)
        dep[npy, npx] = (60, 255, 60)          # 新检测（绿）

        H, W = base.shape[:2]
        gap = 6
        canvas = np.zeros((H, W * 3 + gap * 2, 3), dtype=np.uint8)
        for i, im in enumerate((base, vis, dep)):
            x0 = i * (W + gap)
            canvas[:, x0:x0 + W] = (im * 255 if im.dtype == np.float32 else im)

        img = Image.fromarray(canvas)
        dr = ImageDraw.Draw(img)
        cs = 8
        dr.line([(lpx * 3 - cs, lpy), (lpx * 3 + cs, lpy)], fill=(255, 40, 40), width=2)
        dr.line([(lpx * 3, lpy - cs), (lpx * 3, lpy + cs)], fill=(255, 40, 40), width=2)
        # W*3+gap*2 宽度上 base 占前 W 列 —— 十字只画在 base 上
        dr.line([(lpx - cs, lpy), (lpx + cs, lpy)], fill=(255, 40, 40), width=2)
        dr.line([(lpx, lpy - cs), (lpx, lpy + cs)], fill=(255, 40, 40), width=2)
        # 新检测（绿十字，画在 base 与深度上）
        dr.line([(npx - cs, npy), (npx + cs, npy)], fill=(40, 255, 60), width=2)
        dr.line([(npx, npy - cs), (npx, npy + cs)], fill=(40, 255, 60), width=2)
        dr.line([(npx * 3 - cs, npy), (npx * 3 + cs, npy)], fill=(40, 255, 60), width=2)
        dr.line([(npx * 3, npy - cs), (npx * 3, npy + cs)], fill=(40, 255, 60), width=2)

        path = out_dir / f"mask_probe_{name.replace('.jpg', '')}.png"
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
        img.save(path)
        print(f"{name[:20]:22s} 旧 ({lx:.3f},{ly:.3f})  新 ({nx:.3f},{ny:.3f})  "
              f"亮度分位 90/95/99 = {p90:.3f}/{p95:.3f}/{p99:.3f}")
        print(f"  → {path.name}")

    print("\n（左：画面+光源十字；中：亮度+分位等高线；右：深度）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
