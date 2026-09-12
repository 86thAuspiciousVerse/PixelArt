"""修复前后的 1:1 局部对照：让"黑点"这件事直接看得见。

取一块粒子密集的区域放大，并排：
  左  旧行为（边缘参考含粒子）→ 每颗粒子周围被压暗成小黑点
  右  修复后（边缘参考不含粒子）→ 粒子只是亮点，没有暗晕

上方标出"因粒子而变暗"的像素（旧行为），看它是不是正好围着粒子。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.paths import INPUT, out  # noqa: E402
from pixelart.palette import palette_from_frames  # noqa: E402
from pixelart.pipeline import AnimParams, compose_frame, finish_frame, prepare_scene  # noqa: E402
from pixelart.pixelate import PixelTail  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(x):
    return x.astype(np.float32) @ LUMA


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT / "ref04_lain_room.jpg"
    tag = sys.argv[2] if len(sys.argv) > 2 else "lain"

    sc = prepare_scene(Image.open(src), aspect=0.0, tail=PixelTail(n_colors=32),
                       depth_estimator=DepthEstimator())
    gh, gw = sc.grid[1], sc.grid[0]
    a_on = AnimParams(dust_count=200, dust_bright=0.55)
    a_off = AnimParams(dust_count=0)

    ts = [i / 12 for i in range(12)]
    on = [compose_frame(sc, t=t, anim=a_on, return_content=True) for t in ts]
    off = [compose_frame(sc, t=t, anim=a_off, return_content=True) for t in ts]
    pal = palette_from_frames([c for c, _ in on], n_colors=32, max_samples=12)

    # 统计"因粒子变暗"的累计热区（旧行为）
    heat = np.zeros((gh, gw), np.float32)
    for (con, ctn), (cof, cto) in zip(on, off):
        u_on, _ = finish_frame(con, sc.tail, palette=pal, edge_ref=None)
        u_off, _ = finish_frame(cof, sc.tail, palette=pal, edge_ref=None)
        heat = np.maximum(heat, np.clip(luma(u_off) - luma(u_on), 0, None))

    # 选一块"变暗最集中"的区域
    k = sc.scale
    ks = np.ones(9, np.float32)
    score = np.convolve(heat.sum(0), ks, "same")[:, None] + np.convolve(heat.sum(1), ks, "same")[None, :]
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    hh, hw = 70, 100
    y0, x0 = int(np.clip(cy - hh // 2, 0, gh - hh)), int(np.clip(cx - hw // 2, 0, gw - hw))
    print(f"网格 {gw}x{gh}  裁切区域 ({x0},{y0}) 起 {hw}x{hh}  →  放大 x{k}")

    a = finish_frame(on[0][0], sc.tail, palette=pal, edge_ref=None)[0]
    b = finish_frame(on[0][0], sc.tail, palette=pal, edge_ref=on[0][1])[0]
    ca = Image.fromarray(a[y0:y0 + hh, x0:x0 + hw]).resize((hw * k, hh * k), Image.Resampling.NEAREST)
    cb = Image.fromarray(b[y0:y0 + hh, x0:x0 + hw]).resize((hw * k, hh * k), Image.Resampling.NEAREST)
    heat_c = Image.fromarray((np.clip(heat[y0:y0 + hh, x0:x0 + hw] * 22, 0, 255)).astype(np.uint8))
    heat_c = heat_c.resize((hw * k, hh * k), Image.Resampling.NEAREST).convert("RGB")

    cw = ca.width + 20
    ch = ca.height + 52
    sheet = Image.new("RGB", (cw * 3, ch), (8, 9, 13))
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate((
        ("修复前（边缘参考含粒子）", ca),
        ("修复后（边缘参考不含粒子）", cb),
        ("因粒子而变暗的量 x22", heat_c),
    )):
        sheet.paste(im, (i * cw + 10, 44))
        d.text((i * cw + 12, 12), label, font=C.font(24), fill=C.LABEL_FG)
    C.save(sheet, out("m2", f"{tag}_dustfix.png"))
    print(f"[ok] out/m2/{tag}_dustfix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
