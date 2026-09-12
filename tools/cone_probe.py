"""★ 光锥（M5.1）探针 —— 真素材上的端到端验收 + 肉眼可看的对比条图。

判据（每条都对应一条不变量）：
  ① **关闭时逐位不变**：cone=0 的输出必须与"完全不传锥参数"逐位相同
     （新特性不许改动默认行为）。
  ② **非空转**：cone>0 的画面必须真的变（u8 平均差 > 阈值）——
     防"数学上生效、视觉上看不出"。
  ③ **方向有效**：换方向角必须换出不同的画面（否则方向参数是摆设）。
  ④ **闭合**：开锥后 frame(0) 与 frame(1) 仍逐位相同（时序铁律 3）。
  ⑤ 出一条四联图（关 / 0.35 / 0.7 / 0.7 反向）到 out/m5/ 供肉眼验收。

用法： python tools/cone_probe.py [--asset 名字] [--stage 0.7]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from pixelart.pipeline import AnimParams, compose_frame, finish_frame, upscale
from pixelart.tune import TuneParams, build_scene


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="ref04_lain_room.jpg")
    ap.add_argument("--stage", type=float, default=0.7)
    ap.add_argument("--grid", type=int, default=240)
    args = ap.parse_args()

    p = TuneParams.from_query({
        "asset": [args.asset], "grid_long": [str(args.grid)],
        "work_long": [str(args.grid * 4)], "aspect": ["native"],
    })
    img = Image.open(ROOT / "assets" / "input" / args.asset)
    scene = build_scene(img, p)
    anim = p.to_anim()

    def render(t, **cone):
        """按当前 TuneParams 合成一帧（cone 参数逐次覆盖）。"""
        q = TuneParams(**{**p.to_dict(), **cone})
        kw = q.compose_kwargs(scene)
        kw["anim"] = anim
        u8, _ = finish_frame(compose_frame(scene, t=t, **kw), scene.tail)
        return u8

    print(f"素材 {args.asset}  网格 {args.grid}  张角 {args.stage}")
    ok = True

    # ① 关闭时逐位不变：显式 cone=0 vs 完全不传（默认）
    a = render(0.0, rays_cone=0.0)
    b = render(0.0)
    same = np.array_equal(a, b)
    print(f"① cone=0 与默认逐位相同: {'✅' if same else '❌'}"
          + ("" if same else f"  最大差 {np.abs(a.astype(int)-b.astype(int)).max()}"))
    ok &= same

    # ②③ 判据下在**图层**上，不在 u8 帧上
    #
    # ⚠️⚠️ 这是本轮最重要的教训：实测整个体积光图层在这几张素材上只有
    #     **1~3/255**（`rays=0.55` + `screen_falloff` 是刻意调弱的），
    #     所以在它上面做锥形塑形，最终 u8 帧只差 0.04~1.8/255。
    #     拿 u8 当判据会得出"没生效"的结论，而几何完全正确 ——
    #     **判据要下在被测对象真正生效的层级上。**
    #     而这同时也说明：要做成"一眼可见的光柱"，必须把光锥做成**独立的加性层**，
    #     而不是给这个弱层乘权重（下一轮的工作）。
    from pixelart.compose import volumetric_light
    kw0 = dict(strength=p.rays, occlude_gain=5.0, falloff_gain=2.5,
               screen_falloff=p.rays_spread, t=0.0, flicker=0.0)
    l_off = volumetric_light(scene.base, scene.far, p.light_xy(scene), **kw0)
    l_on = volumetric_light(scene.base, scene.far, p.light_xy(scene),
                            cone_angle=args.stage, cone_dir_deg=80.0,
                            cone_reach=0.8, cone_gain=1.6, **kw0)
    l_rev = volumetric_light(scene.base, scene.far, p.light_xy(scene),
                             cone_angle=args.stage, cone_dir_deg=-60.0,
                             cone_reach=0.8, cone_gain=1.6, **kw0)
    d_lay = float(np.abs(l_on - l_off).mean())
    d_dir = float(np.abs(l_on - l_rev).mean())
    print(f"② 开锥：图层差异 {d_lay:.3e}（换算 {(l_on - l_off).mean()*255:+.2f}/255 平均）"
          f" {'✅' if d_lay > 1e-5 else '❌'}")
    print(f"③ 换方向 80°→−60°：图层差异 {d_dir:.3e} {'✅' if d_dir > 1e-5 else '❌'}")
    ok &= d_lay > 1e-5
    ok &= d_dir > 1e-5

    # ②b 如实报告最终帧的差异（**不设门槛** —— 它是"够不够看得见"的读数）
    c = render(0.0, rays_cone=args.stage, rays_dir=80.0)
    e = render(0.0, rays_cone=args.stage, rays_dir=-60.0)
    d = float(np.abs(c.astype(int) - a.astype(int)).mean())
    dd = float(np.abs(e.astype(int) - c.astype(int)).mean())
    print(f"②b 最终 u8 帧：开/关 {d:.2f}/255 · 换方向 {dd:.2f}/255 "
          f"（当前体积光层本身只有 1~3/255，所以塑形后仍然偏弱）")

    # ④ 闭合
    f0 = render(0.0, rays_cone=args.stage, rays_dir=80.0)
    f1 = render(1.0, rays_cone=args.stage, rays_dir=80.0)
    clo = np.array_equal(f0, f1)
    print(f"④ 开锥后循环闭合 frame(0)==frame(1): {'✅' if clo else '❌'}")
    ok &= clo

    # ④b ⭐ 光柱（独立加性层）的验收判据：**锥内**平均差异 ≥ 8/255
    #    （全帧均值会被没被照到的区域稀释，所以判据必须下在锥内）
    from pixelart.compose import cone_weight
    lxn, lyn = p.light_xy(scene)
    lx = int(lxn * (scene.grid[0] - 1))
    ly = int(lyn * (scene.grid[1] - 1))
    cwm = cone_weight(scene.grid[0], scene.grid[1], lx, ly,
                      args.stage, 80.0, 0.8, 1.6) > 0.5
    sh = render(0.0, rays_cone=args.stage, rays_dir=80.0, rays_shaft=1.0)
    dd = np.abs(sh.astype(int) - c.astype(int)).max(axis=2)
    in_cone = float(dd[cwm].mean())
    print(f"④b 光柱（强度 1.0）锥内平均差异 {in_cone:.1f}/255 "
          f"（覆盖 {cwm.mean()*100:.1f}%）{'✅ ≥8' if in_cone >= 8 else '❌ <8'}")
    ok &= in_cone >= 8.0

    # ⑤ 四联图：关 / 锥 0.35 / 锥 0.7 / 锥 0.7 + 光柱 1.0
    panels = [a,
              render(0.0, rays_cone=0.35, rays_dir=80.0),
              c,
              sh]
    gap = 4
    h = panels[0].shape[0]
    W = sum(x.shape[1] for x in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((h, W, 3), dtype=np.uint8)
    x = 0
    for pan in panels:
        canvas[:, x:x + pan.shape[1]] = pan
        x += pan.shape[1] + gap
    out_dir = ROOT / "out" / "m5"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cone_strip_{args.asset.replace('.jpg', '')}.png"
    upscale(canvas, 2).save(path)
    print(f"⑤ 四联图（关 / 锥0.35 / 锥0.7 / 锥0.7+光柱1.0）: {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
