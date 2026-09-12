"""spike-1 —— 验证单目深度在**风格化素材**上是否可靠。

这是整个「分层 + 深度雾」路线的地基。深度模型训练集是真实照片，
动画截图 / 手绘风格上的表现必须实测，不能假设。

对 assets/input/ 下每张图输出四联图：
    [原图] [暖色=近] [暖色=远] [等深线叠加在原图]

并打印若干人工标注采样点的深度值（0=近, 1=远），便于客观核对前后关系。

用法::

    python tools/depth_probe.py                     # 跑 assets/input 下全部
    python tools/depth_probe.py --only ref04        # 只跑文件名含 ref04 的
    python tools/depth_probe.py --no-points         # 跳过采样点表
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
from pixelart.analyze import DepthEstimator, colorize_depth  # noqa: E402
from pixelart.paths import DOCS, INPUT, out  # noqa: E402

#: 人工标注的采样点（归一化 x, y），用于客观核对前后关系。
#: 只对 ref04（Lain 房间）标注，因为那是最有判断价值的一张。
#: 期望的前后顺序（地面真值，从左到右依次由近到远）::
#:     前景线缆 ≈ 右下桌面前缘  <  设备堆 ≈ 服务器机架  <  主角 ≈ CRT  <  窗台玩偶  <  窗玻璃
PROBE_POINTS: dict[str, list[tuple[str, float, float]]] = {
    "ref04": [
        ("前景线缆(最前)", 0.093, 0.857),
        ("右下桌面前缘", 0.648, 0.791),
        ("女孩前方设备堆", 0.519, 0.708),
        ("左侧服务器机架", 0.130, 0.330),
        ("主角躯干", 0.500, 0.643),
        ("CRT 屏幕", 0.556, 0.494),
        ("窗台玩偶(左1)", 0.727, 0.395),
        ("窗玻璃(熊之间)", 0.833, 0.330),
    ],
}


def contour_overlay(rgb: np.ndarray, depth_far: np.ndarray, levels: int = 12,
                    band: float = 0.0035, color=(255, 255, 255)) -> np.ndarray:
    """把等深线画在原图上，直观看层次。depth_far: 0=近 1=远。"""
    img = rgb.astype(np.float32).copy()
    for lv in np.linspace(0.0, 1.0, levels + 2)[1:-1]:
        m = np.abs(depth_far - lv) < band
        img[m] = color
    return img.astype(np.uint8)


def annotate_points(rgb: np.ndarray, points, depth_far: np.ndarray) -> np.ndarray:
    """在图上标出采样点并写上深度值。"""
    im = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(im)
    h, w = rgb.shape[:2]
    f = C.font(22)
    for name, nx, ny in points:
        x, y = int(nx * w), int(ny * h)
        v = float(depth_far[max(0, y - 4):y + 5, max(0, x - 4):x + 5].mean())
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(255, 80, 80), width=3)
        text = f"{name} {v:.2f}"
        d.rectangle([x + 10, y - 26, x + 34 + len(text) * 12, y + 6], fill=(10, 12, 18))
        d.text((x + 16, y - 24), text, font=f, fill=(255, 230, 180))
    return np.asarray(im)


def process(path: Path, est: DepthEstimator, panel_w: int = 640, points=True) -> Path:
    src = Image.open(path).convert("RGB")
    t0 = time.time()
    near01 = est.predict(src)          # 1 = 近
    dt = time.time() - t0
    far01 = 1.0 - near01               # 0 = 近, 1 = 远

    rgb = np.asarray(src)
    panels = [
        ("原图", rgb),
        ("深度伪彩 · 暖色=近（模型原生）", colorize_depth(near01, invert=False)),
        ("深度伪彩 · 暖色=远（反相）", colorize_depth(near01, invert=True)),
        ("等深线叠加原图（按「暖=近」的层次）", contour_overlay(rgb, far01)),
    ]
    sheet = C.strip(panels, panel_w=panel_w)
    key = path.stem
    dst = out("probe", f"depth_{key}.png")
    C.save(sheet, dst)
    print(f"[ok] {path.name}  推理 {dt:.1f}s  ->  out/probe/{dst.name}")

    hits = {k: v for k, v in PROBE_POINTS.items() if k in key}
    if points and hits:
        for _, plist in hits.items():
            marked = C.strip([("采样点（数值 = 0 近 → 1 远）", annotate_points(rgb, plist, far01))],
                             panel_w=panel_w * 2)
            C.save(marked, out("probe", f"depth_{key}_points.png"))
            print(f"     采样点（0 = 近, 1 = 远）:")
            for name, nx, ny in plist:
                h, w = far01.shape
                x, y = int(nx * w), int(ny * h)
                v = float(far01[max(0, y - 4):y + 5, max(0, x - 4):x + 5].mean())
                print(f"       {name:<16s} {v:.3f}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只处理文件名包含该子串的图")
    ap.add_argument("--panel-w", type=int, default=640)
    ap.add_argument("--no-points", action="store_true")
    args = ap.parse_args()

    files = sorted(INPUT.glob("*.jpg")) + sorted(INPUT.glob("*.png"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        print(f"assets/input 下没有找到图片: {INPUT}")
        return 1

    print(f"加载模型 ...")
    est = DepthEstimator()
    print(f"  {est!r}")
    for f in files:
        process(f, est, args.panel_w, points=not args.no_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
