"""★ 分层视差（M5）探针 —— 在**真实素材**上验证视差推拉的性质。

合成用例在 tests/test_parallax.py；这里用真图跑端到端：
  1. **循环闭合**：frame(0) 与 frame(1) 的 u8 输出**逐位相同**（时序铁律 3）。
  2. **非空转**：frame(0.5) 与 frame(0) 必须有肉眼可见的差异（u8 平均差 > 2）。
     —— 这条是防"测了个假东西"的关键：数学上在动、视觉上没动 = 白做。
  3. **默认路径不变**：parallax=0 时与旧参数（无 parallax 字段）逐位相同。
  4. **幅度分布**：四档占比要"大部分安静 + 一部分明显"（全画面均匀动 = 廉价）。
  5. 出一张 t=0 / 0.25 / 0.5 / 0.75 的四联条图到 out/m5/ 供肉眼验收。

用法：  python tools/parallax_probe.py [--asset 名字] [--amp 0.4]
"""

from __future__ import annotations

import argparse
import sys
import time
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
    ap.add_argument("--amp", type=float, default=0.4)
    ap.add_argument("--grid", type=int, default=240)
    args = ap.parse_args()

    params = TuneParams.from_query({
        "asset": [args.asset], "grid_long": [str(args.grid)],
        "work_long": [str(args.grid * 4)], "aspect": ["native"],
    })
    img = Image.open(ROOT / "assets" / "input" / args.asset)
    print(f"素材 {args.asset}  网格 {params.grid_long}  视差幅度 {args.amp}")

    scene = build_scene(img, params)
    base_anim = params.to_anim()
    anim = AnimParams(**{**base_anim.to_dict(), "parallax": args.amp})
    kw = params.compose_kwargs(scene)
    kw["anim"] = anim

    # ① 循环闭合（逐位）
    def _fin(t):
        # ⚠️ finish_frame 返回二元组 (u8 帧, 边缘图) —— 只取帧
        return finish_frame(compose_frame(scene, t=t, **kw), scene.tail)[0]
    f0 = _fin(0.0)
    f1 = _fin(1.0)
    ok_closure = np.array_equal(f0, f1)
    print(f"① 循环闭合 frame(0)==frame(1) 逐位: {'✅' if ok_closure else '❌'}")
    if not ok_closure:
        d = np.abs(f0.astype(int) - f1.astype(int))
        print(f"   不一致 {(d > 0).mean() * 100:.2f}%  最大 {d.max()}")

    # ② 非空转
    f5 = _fin(0.5)
    d5 = float(np.abs(f5.astype(int) - f0.astype(int)).mean())
    print(f"② 非空转 frame(0.5) vs frame(0): 平均差 {d5:.2f}/255 "
          f"{'✅' if d5 > 2.0 else '❌（视觉上没动 = 白做）'}")

    # ③ 默认路径不变：parallax=0 的 anim 与旧 anim（无该字段）输出一致
    old_anim = AnimParams(**{k: v for k, v in base_anim.to_dict().items()})
    kw_old = dict(kw); kw_old["anim"] = old_anim
    f_old = compose_frame(scene, t=0.3, **kw_old)
    kw_zero = dict(kw); kw_zero["anim"] = AnimParams(**{**base_anim.to_dict(), "parallax": 0.0})
    f_zero = compose_frame(scene, t=0.3, **kw_zero)
    ok_default = np.array_equal(f_old, f_zero)
    print(f"③ parallax=0 与旧参数逐位相同: {'✅' if ok_default else '❌'}")

    # ④ 幅度分布（沿用 M3 的四档口径，量的是 u8 帧差）
    f25 = _fin(0.25)
    d = np.abs(f5.astype(int) - f25.astype(int)).max(axis=2)
    quiet = float((d <= 1).mean()); faint = float(((d > 1) & (d <= 4)).mean())
    vis = float(((d > 4) & (d <= 12)).mean()); strong = float((d > 12).mean())
    print(f"④ 帧间幅度分布: 静止 {quiet*100:.0f}% · 微弱 {faint*100:.0f}% · "
          f"看得出 {vis*100:.0f}% · 明显 {strong*100:.0f}%")
    verdict = "✅" if (quiet > 0.3 and strong + vis > 0.05) else "⚠️（全静或全动都要警惕）"
    print(f"   {verdict}  理想形态是「大部分安静 + 一部分明显」")

    # ⑤ 性能
    t0 = time.perf_counter()
    for _ in range(3):
        compose_frame(scene, t=0.3, **kw)
    with_p = (time.perf_counter() - t0) / 3 * 1000
    kw_off = dict(kw); kw_off["anim"] = AnimParams(**{**base_anim.to_dict(), "parallax": 0.0})
    t0 = time.perf_counter()
    for _ in range(3):
        compose_frame(scene, t=0.3, **kw_off)
    without = (time.perf_counter() - t0) / 3 * 1000
    print(f"⑤ 每帧合成: 视差开 {with_p:.0f}ms / 关 {without:.0f}ms "
          f"(+{with_p - without:.0f}ms)")

    # ⑥ 四联条图（肉眼验收）
    strip_h = f0.shape[0]
    panels = [f0, _fin(0.25), f5, _fin(0.75)]
    labels = ["t=0.00", "t=0.25", "t=0.50", "t=0.75"]
    gap = 4
    W = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((strip_h, W, 3), dtype=np.uint8)
    x = 0
    for p_ in panels:
        canvas[:, x:x + p_.shape[1]] = p_
        x += p_.shape[1] + gap
    out_dir = ROOT / "out" / "m5"
    out_dir.mkdir(parents=True, exist_ok=True)
    strip = upscale(canvas, 2)
    strip_path = out_dir / f"parallax_strip_{args.asset.replace('.jpg', '')}.png"
    strip.save(strip_path)
    print(f"⑥ 四联条图: {strip_path}")

    bad = (not ok_closure) or d5 <= 2.0 or (not ok_default)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
