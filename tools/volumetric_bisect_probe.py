"""体积光差异的**逐项二分** —— 关掉每一项，看误差归谁。

前面已经排除的：
  · 采样下标不一致（实测 0/172032 —— 完全一致）
  · 加权方式（acc 标量 vs vec3 —— 修了，但幅度没变）

剩下的候选：遮挡 exp、屏幕空间衰减 pow、高光 mask、衰减累积的中间精度。
关掉一项跑一次，看最大差掉到哪 —— 掉到 1e-7 量级就说明那一项是主因。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pixelart.animate import flicker_gain  # noqa: E402
from pixelart.compose import auto_scatter_color, volumetric_light  # noqa: E402
from pixelart.webgpu import volumetric_uniforms  # noqa: E402
from wgsl_lab import WgslLab, _load_wgsl, _test_depth, _test_image  # noqa: E402


def main() -> int:
    w, h = 96, 64
    fogged = _test_image(w, h, seed=11)
    far = _test_depth(w, h)
    lxn, lyn = 0.62, 0.28
    lx, ly = int(lxn * (w - 1)), int(lyn * (h - 1))
    t, flicker = 0.41, 0.35
    gain = flicker_gain(t, depth=flicker)
    air = auto_scatter_color(fogged, lx, ly, radius=3, sat_max=0.25)

    lab = WgslLab()
    print("体积光差异的逐项二分（最大差）")
    print("=" * 64)

    base = dict(samples=28, span=0.85, decay=0.965, strength=0.55,
                occlude_gain=5.0, falloff_gain=2.5, screen_falloff=1.0,
                threshold=0.48, knee=0.3, gain=gain, lx=lx, ly=ly)

    variants = [
        ("全量（基线）", {}),
        ("关遮挡 occlude_gain=0", {"occlude_gain": 0.0}),
        ("关屏幕衰减 screen_falloff=0", {"screen_falloff": 0.0}),
        ("无衰减累积 decay=1.0", {"decay": 1.0}),
        # ⚠️ 刻意**不测** threshold/knee：numpy 的 volumetric_light 把
        #    `bright_pass(rgb01, 0.48, knee=0.3)` 写死在函数体里，不接受这两个参数。
        #    加了这一行只会拿 GPU 的 0.0 去比 numpy 的 0.48/0.3，
        #    得出的"1.7e-2 差异"是**测试自己的假象**，不是移植错误。
        #    （第一版就加了，还差点当成潜伏 bug 去追。）
        ("单采样 samples=1", {"samples": 1}),
        ("全关（遮挡+屏幕+衰减）", {"occlude_gain": 0.0, "screen_falloff": 0.0,
                                    "decay": 1.0}),
    ]

    for label, ov in variants:
        kw = dict(base)
        kw.update(ov)
        got = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                      inputs={"fogged": fogged.reshape(-1), "far": far.reshape(-1),
                              "air": np.asarray(air, np.float32)},
                      uniforms=volumetric_uniforms(w, h, **kw),
                      out_count=w * h * 3, out_shape_tail=(h, w, 3),
                      label="volumetric")
        want = np.clip(fogged + volumetric_light(
            fogged, far, (lxn, lyn),
            samples=kw["samples"], span=kw["span"], decay=kw["decay"],
            strength=kw["strength"], occlude_gain=kw["occlude_gain"],
            falloff_gain=kw["falloff_gain"], air_sat_max=0.25,
            t=t, flicker=flicker, screen_falloff=kw["screen_falloff"]), 0.0, 1.0)
        d = np.abs(got - want)
        n_big = int((d.max(axis=2) > 1e-3).sum())
        print(f"  {label:32s} 最大 {d.max():.3e}   >1e-3 的像素 {n_big:5d}")

    # ── 采样下标一致性：这是最容易出错的几何部分，单独守住 ──
    print()
    print("采样下标一致性（几何部分，必须完全一致）")
    samples, span = 28, 0.85
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (np.float32(lx) - xx) / np.float32(w)
    dy = (np.float32(ly) - yy) / np.float32(h)

    def positions(f64_step: bool):
        out = []
        for i in range(1, samples + 1):
            if f64_step:
                ss = i / samples * span            # Python float（f64）
            else:
                ss = np.float32(np.float32(i) / np.float32(samples) * np.float32(span))
            px = (xx + (dx * np.float32(w)) * ss).astype(np.int32)
            py = (yy + (dy * np.float32(h)) * ss).astype(np.int32)
            out.append(np.clip(py, 0, h - 1) * w + np.clip(px, 0, w - 1))
        return out

    a, b2 = positions(True), positions(False)
    mism = sum(int((u != v).sum()) for u, v in zip(a, b2))
    tot = samples * w * h
    print(f"  float64 与 float32 步长给出的下标：不一致 {mism}/{tot} "
          f"({100.0 * mism / tot:.5f}%)")
    print("  （步长 s 在 numpy 里是 Python float=f64、在 WGSL 里只有 f32；")
    print("    只要不为 0，采样点就没有歧义。）")

    print("=" * 64)
    print("  判读：哪一行掉到 1e-7 量级，主因就在那一项。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
