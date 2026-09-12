"""像素尾巴移植的**非空转检验** —— 确认验收台测的是真东西。

一个"逐位一致 0/6144"的结果有两种可能：
  (a) 移植正确；
  (b) 测试是空转的 —— 比如抖动/边缘/吸附根本没生效，两边都在算同一个恒等映射。

分辨两者只能靠**敏感性检验**：逐项开关，确认每一项都真的改变输出。
这与上一轮"变异测试"是同一条方法论：
**先证明检查会动，再相信检查的结果。**

顺带验一条硬约束：输出颜色必须**全部来自色板**（约束 3）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pixelart.pixelate import PixelTail  # noqa: E402
from wgsl_lab import WgslLab, _load_wgsl, _test_image, _test_palette  # noqa: E402


def _run(lab, w, h, composed, edge_ref, pal, tail) -> np.ndarray:
    n = len(pal)
    u = np.zeros((2, 4), dtype=np.float32)
    u[0] = [float(n), tail.dither, 1.0 if tail.dither_adaptive else 0.0, tail.edge_gain]
    u[1] = [tail.edge_strength, float(w), float(h), 0.0]
    g = lab.run(_load_wgsl("tail"), "main", (w, h),
                inputs={"src": composed.reshape(-1),
                        "edge_ref": edge_ref.reshape(-1),
                        "pal": pal.reshape(-1)},
                uniforms=u, out_count=w * h, out_dtype=np.uint32, label="tail")
    p = np.asarray(g, dtype=np.uint32).reshape(h, w)
    return np.stack([p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF], -1).astype(np.int32)


def main() -> int:
    w, h = 96, 64
    composed = _test_image(w, h, seed=11)
    edge_ref = _test_image(w, h, seed=12)
    pal = _test_palette(32)
    lab = WgslLab()

    full = _run(lab, w, h, composed, edge_ref, pal, PixelTail())
    nodith = _run(lab, w, h, composed, edge_ref, pal, PixelTail(dither=0.0))
    noedge = _run(lab, w, h, composed, edge_ref, pal, PixelTail(edge_strength=0.0))
    bare = _run(lab, w, h, composed, edge_ref, pal,
                PixelTail(dither=0.0, edge_strength=0.0))
    raw = np.clip(composed * 255, 0, 255).astype(np.int32)

    def nd(a, b):
        return int((np.abs(a - b).max(axis=2) > 0).sum())

    ok = True
    print("像素尾巴移植 —— 非空转检验")
    print("=" * 72)
    print("  每一项都必须真的改变输出，否则上面的\"逐位一致\"没有意义：")
    for label, d, thresh in (
        ("① 抖动生效", nd(full, nodith), 50),
        ("② 边缘压暗生效", nd(full, noedge), 50),
        ("③ 色板吸附生效（vs 原始量化）", nd(bare, raw), 50),
    ):
        good = d >= thresh
        ok = ok and good
        print(f"    {'✅' if good else '❌'} {label:28s} 差异像素 {d:5d}/{w * h}"
              f"（要求 >= {thresh}）")

    # 自适应抖动：中间调应比纯黑/纯白抖得多
    print()
    print("  自适应抖动的语义：中间调最强，纯黑/纯白关掉")
    luma = composed @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    dmap = (np.abs(full - nodith).max(axis=2) > 0)
    mid = float(dmap[(luma > 0.3) & (luma < 0.7)].mean()) if ((luma > 0.3) & (luma < 0.7)).any() else 0.0
    ext = float(dmap[(luma < 0.05) | (luma > 0.95)].mean()) if ((luma < 0.05) | (luma > 0.95)).any() else 0.0
    good = mid > ext
    ok = ok and good
    print(f"    {'✅' if good else '❌'} 中间调受影响比例 {mid:.3f} > 极端区 {ext:.3f}")

    # 约束 3：输出颜色必须全部来自色板
    print()
    print("  约束 3：输出颜色必须**全部来自色板**")
    pal_u8 = set(map(tuple, np.clip(pal ** 2 * 255, 0, 255).astype(np.int32).tolist()))
    uniq = np.unique(full.reshape(-1, 3), axis=0)
    stray = [c for c in uniq.tolist() if tuple(c) not in pal_u8]
    good = not stray
    ok = ok and good
    print(f"    {'✅' if good else '❌'} 输出用了 {len(uniq)} 种颜色 / 色板 {len(pal)} 种，"
          f"越界 {len(stray)} 种" + (f"  例：{stray[:3]}" if stray else ""))

    print("=" * 72)
    print(f"  {'全部通过' if ok else '存在问题'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
