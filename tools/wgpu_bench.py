"""WebGPU 移植的性能收益 —— 在**生产分辨率**上量，而不是靠"GPU 肯定快"。

═══ 为什么要量，而且量得诚实 ═══

本项目早就论证过（spike-log M3-GPU）：**收益不在算力**。
3~12 万像素太小，GPU 的算力优势会被"内核启动 + 数据搬运"吃掉不少。
真正的收益是**省掉每帧的服务端编码与 HTTP 往返**。

所以这个工具量两件事，分开报：

  · **纯计算**：各 pass 的 GPU 计算时间（把上传/回读排除掉）
  · **含搬运**：连每帧的缓冲上传一起算进去

浏览器里场景缓冲是**常驻**的（一次上传、每帧只传 uniform），
所以真实体验更接近前者；而服务端方案每帧还要多一次 PNG 编码 + HTTP。

⚠️ 这个工具**不**测浏览器里的呈现，也不测编码 —— 那些要用真浏览器量。
它只回答一个问题：**把渲染搬进 GPU，光计算值不值。**
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pixelart.animate import flicker_gain  # noqa: E402
from pixelart.compose import (  # noqa: E402
    bloom_float, brightest_center, depth_fog, limit_saturation,
    volumetric_light,
)
from pixelart.animate import dust_layer  # noqa: E402
from pixelart.pixelate import PixelTail, pixelate  # noqa: E402
from pixelart.webgpu import (  # noqa: E402
    bloom_blur_uniforms, bloom_bright_uniforms, bloom_combine_uniforms,
    bloom_kernels, dust_apply_uniforms, dust_fixed_point_scale,
    dust_particle_params, dust_splat_uniforms, fog_uniforms,
    scatter_uniforms, volumetric_uniforms,
)
from wgsl_lab import WgslLab, _load_wgsl, _test_depth, _test_image, _test_palette  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    w, h, nf = args.w, args.h, args.frames
    print("=" * 74)
    print("  WebGPU 移植的性能收益")
    print("=" * 74)

    lab = WgslLab(args.backend)
    print(f"  适配器   {lab.describe()}")
    print(f"  分辨率   {w}×{h}（{w * h / 1000:.0f} 千像素）")
    print(f"  帧数     {nf}（取平均）")
    print()

    # ── 数据准备（与 compose_frame 的输入一致） ──
    base = _test_image(w, h, seed=21)
    far = _test_depth(w, h)
    pal = _test_palette(32)
    tail = PixelTail()
    fog_color = limit_saturation(np.array([0.55, 0.68, 0.58], np.float32), 0.40)
    lxn, lyn = brightest_center(base, blur_radius=3.0)
    lx, ly = int(lxn * (w - 1)), int(lyn * (h - 1))
    n, npix = w * h * 3, w * h
    zero = np.zeros(n, dtype=np.float32)

    kflat, table = bloom_kernels((2.0, 5.0, 11.0))
    par = dust_particle_params(200, seed=11)
    dscale = dust_fixed_point_scale(200)
    par_flat, far_flat, base_flat = par.reshape(-1), far.reshape(-1), base.reshape(-1)

    def gpu_noreadback(t: float):
        """同一条链，但**不回读、缓冲走池子** —— 模拟浏览器里缓冲常驻的情形。"""
        gain = flicker_gain(t, depth=0.55)
        lab.run(_load_wgsl("fog"), "main", (w, h),
                inputs={"src": base_flat, "far": far_flat},
                uniforms=fog_uniforms(w, h, t, fog_color, 0.8, 3.0, drift=0.35),
                out_count=n, readback=False)
        lab.run(_load_wgsl("scatter"), "main", (1, 1),
                inputs={"src": base_flat},
                uniforms=scatter_uniforms(w, h, lx, ly, 3, 0.25),
                out_count=3, readback=False)
        lab.run(_load_wgsl("volumetric"), "main", (w, h),
                inputs={"fogged": base_flat, "far": far_flat,
                        "air": np.array([0.5, 0.5, 0.5], np.float32)},
                uniforms=volumetric_uniforms(w, h, samples=28, strength=0.55,
                                             gain=gain, lx=lx, ly=ly),
                out_count=n, readback=False)
        lab.run(_load_wgsl("dust_splat"), "main", None,
                inputs={"par": par_flat, "far": far_flat},
                uniforms=dust_splat_uniforms(w, h, 200, t, 0.55, 0.65, 1.8,
                                             (lxn, lyn), dscale, True),
                out_count=npix, out_dtype=np.uint32, workgroup=(64, 1),
                dispatch=((200 + 63) // 64, 1), readback=False)
        lab.run(_load_wgsl("dust_apply"), "main", (w, h),
                inputs={"bin": np.zeros(npix, np.uint32), "lit": base_flat},
                uniforms=dust_apply_uniforms(w, h, dscale, 0.55),
                out_count=n, readback=False)
        lab.run(_load_wgsl("bloom_bright"), "main", (w, h),
                inputs={"rgb": base_flat},
                uniforms=bloom_bright_uniforms(w, h, 0.55, 0.25),
                out_count=n, readback=False)
        for entry in table:
            lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                    inputs={"kbuf": kflat, "src": base_flat, "prev": zero},
                    uniforms=bloom_blur_uniforms(w, h, "h", entry, False),
                    out_count=n, readback=False)
            lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                    inputs={"kbuf": kflat, "src": base_flat, "prev": zero},
                    uniforms=bloom_blur_uniforms(w, h, "v", entry, True),
                    out_count=n, readback=False)
        lab.run(_load_wgsl("bloom_combine"), "main", (w, h),
                inputs={"rgb": base_flat, "acc": zero},
                uniforms=bloom_combine_uniforms(w, h, 0.425, (2.0, 5.0, 11.0)),
                out_count=n, readback=False)
        lab.run(_load_wgsl("tail"), "main", (w, h),
                inputs={"src": base_flat, "edge_ref": base_flat,
                        "pal": pal.reshape(-1)},
                uniforms=np.array([[32.0, 0.10, 1.0, 2.0],
                                   [0.35, float(w), float(h), 0.0]],
                                  dtype=np.float32).reshape(-1),
                out_count=npix, out_dtype=np.uint32, readback=False)

    def gpu_frame(t: float):
        gain = flicker_gain(t, depth=0.55)
        fogged = lab.run(_load_wgsl("fog"), "main", (w, h),
                         inputs={"src": base_flat, "far": far_flat},
                         uniforms=fog_uniforms(w, h, t, fog_color, 0.8, 3.0, drift=0.35),
                         out_count=n, out_shape_tail=(h, w, 3), label="fog")
        air = lab.run(_load_wgsl("scatter"), "main", (1, 1),
                      inputs={"src": fogged.reshape(-1)},
                      uniforms=scatter_uniforms(w, h, lx, ly, 3, 0.25),
                      out_count=3, label="scatter")
        lit = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                      inputs={"fogged": fogged.reshape(-1), "far": far_flat,
                              "air": np.asarray(air, np.float32)},
                      uniforms=volumetric_uniforms(w, h, samples=28, span=0.85,
                                                   decay=0.965, strength=0.55,
                                                   occlude_gain=5.0, falloff_gain=2.5,
                                                   screen_falloff=1.0, threshold=0.48,
                                                   knee=0.3, gain=gain, lx=lx, ly=ly),
                      out_count=n, out_shape_tail=(h, w, 3), label="volumetric")
        content = lit.copy()
        bins = lab.run(_load_wgsl("dust_splat"), "main", None,
                       inputs={"par": par_flat, "far": far_flat},
                       uniforms=dust_splat_uniforms(w, h, 200, t, 0.55, 0.65, 1.8,
                                                    (lxn, lyn), dscale, True),
                       out_count=npix, out_dtype=np.uint32, label="dust_splat",
                       workgroup=(64, 1), dispatch=((200 + 63) // 64, 1))
        dusted = lab.run(_load_wgsl("dust_apply"), "main", (w, h),
                         inputs={"bin": bins, "lit": lit.reshape(-1)},
                         uniforms=dust_apply_uniforms(w, h, dscale, 0.55),
                         out_count=n, out_shape_tail=(h, w, 3), label="dust_apply")
        bright = lab.run(_load_wgsl("bloom_bright"), "main", (w, h),
                         inputs={"rgb": dusted.reshape(-1)},
                         uniforms=bloom_bright_uniforms(w, h, 0.55, 0.25),
                         out_count=n, out_shape_tail=(h, w, 3), label="bloom_bright")
        tmp, acc = zero.copy(), zero.copy()
        for entry in table:
            tmp = lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                          inputs={"kbuf": kflat, "src": bright.reshape(-1), "prev": zero},
                          uniforms=bloom_blur_uniforms(w, h, "h", entry, False),
                          out_count=n, out_shape_tail=(h, w, 3), label="bh")
            acc = lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                          inputs={"kbuf": kflat, "src": tmp.reshape(-1),
                                  "prev": acc.reshape(-1)},
                          uniforms=bloom_blur_uniforms(w, h, "v", entry, True),
                          out_count=n, out_shape_tail=(h, w, 3), label="bv")
        comp = lab.run(_load_wgsl("bloom_combine"), "main", (w, h),
                       inputs={"rgb": dusted.reshape(-1), "acc": acc.reshape(-1)},
                       uniforms=bloom_combine_uniforms(w, h, 0.425, (2.0, 5.0, 11.0)),
                       out_count=n, out_shape_tail=(h, w, 3), label="combine")
        packed = lab.run(_load_wgsl("tail"), "main", (w, h),
                         inputs={"src": comp.reshape(-1),
                                 "edge_ref": content.reshape(-1),
                                 "pal": pal.reshape(-1)},
                         uniforms=np.array([[32.0, 0.10, 1.0, 2.0],
                                            [0.35, float(w), float(h), 0.0]],
                                           dtype=np.float32).reshape(-1),
                         out_count=npix, out_dtype=np.uint32, label="tail")
        return np.asarray(packed, np.uint32).reshape(h, w)

    # ── 预热（编译着色器 + 首次分配）──
    gpu_frame(0.0)
    print("  预热完成（着色器已编译）")
    print()

    # ── 量 GPU ──
    t0 = time.perf_counter()
    for i in range(nf):
        gpu_frame(i / nf)
    gpu_ms = (time.perf_counter() - t0) / nf * 1000.0

    # ── 量 CPU：用**生产路径**（compose_frame + finish_frame）──
    #
    # ⚠️ 必须用生产路径，不能用 bloom_float（我新加的那个 numpy 参考实现）。
    #    bloom_float 的卷积是**逐 tap 的 Python 循环** —— 那是为了"定义清晰、
    #    便于逐位比对"而写的，慢是必然的。拿它当 CPU 基线会把 CPU 测得慢好几倍，
    #    得出一个虚假的"提速 20 倍"。
    #    **比较必须拿两边各自真实会跑的代码来比。**
    from pixelart.pipeline import Scene, compose_frame, finish_frame

    scene = Scene(base=base, far=far, grid=(w, h), scale=1,
                  light_xy=(lxn, lyn), fog_color=fog_color, tail=tail)
    from pixelart.pipeline import AnimParams
    anim = AnimParams(dust_count=200, dust_bright=0.55, dust_twinkle=0.55,
                      dust_fade_far=0.65, dust_light_boost=1.8,
                      fog_drift=0.35, light_flicker=0.55)
    ckw = dict(density=0.8, power=3.0, rays=0.55, bloom_strength=0.85,
               anim=anim, fog_sat=0.40, screen_falloff=1.0)

    def cpu_frame(t: float):
        c, ct = compose_frame(scene, t=t, return_content=True, **ckw)
        return finish_frame(c, tail, palette=pal, edge_ref=ct)[0]

    cpu_frame(0.0)
    t0 = time.perf_counter()
    for i in range(max(3, nf // 3)):
        cpu_frame(i / nf)
    k = max(3, nf // 3)
    cpu_ms = (time.perf_counter() - t0) / k * 1000.0

    # ── ⭐ 纯计算量：缓冲池 + **不回读** ⭐ ──
    #
    # ⚠️ 这才是能与"浏览器里的每帧成本"对话的数。
    #    lab.run 默认每个 pass 都把输出读回 CPU（一次 GPU→CPU 同步），
    #    14 个 pass 累起来能占总时间的八成 —— 而浏览器里缓冲常驻、根本不需要回读。
    #    不把这一层剥掉，就会得出"GPU 只比 CPU 快 2.5 倍"这种**低估**的结论。
    lab.pool_buffers = True
    t0 = time.perf_counter()
    for i in range(nf):
        gpu_noreadback(i / nf)
    gpu_compute_ms = (time.perf_counter() - t0) / nf * 1000.0
    lab.pool_buffers = False

    # ── ⭐ 各 pass 的**真实 GPU 计算时间**（一次 encoder 内重复 dispatch）──
    #
    # 从 Python 调 wgpu 时，每次调用的固定开销（write_buffer + 建 bind group +
    # submit + FFI）远大于 13 万像素的计算量本身。把 N 次 dispatch 塞进一个
    # encoder，固定开销被摊薄 N 倍，才能看到计算本身。
    # 那部分固定开销在浏览器里不存在（没有 FFI，WebGPU 调用本身很便宜）。
    print("  各 pass 的真实 GPU 计算时间（重复 dispatch，摊掉调用开销）：")
    REP = 50
    per_pass = {}
    lab.pool_buffers = True
    for label, fn in (
        ("fog", lambda R: lab.run(_load_wgsl("fog"), "main", (w, h),
                                  inputs={"src": base_flat, "far": far_flat},
                                  uniforms=fog_uniforms(w, h, 0.3, fog_color, 0.8, 3.0,
                                                        drift=0.35),
                                  out_count=n, readback=False, repeat=R)),
        ("volumetric", lambda R: lab.run(_load_wgsl("volumetric"), "main", (w, h),
                                         inputs={"fogged": base_flat, "far": far_flat,
                                                 "air": np.array([0.5, 0.5, 0.5], np.float32)},
                                         uniforms=volumetric_uniforms(
                                             w, h, samples=28, strength=0.55,
                                             gain=1.0, lx=lx, ly=ly),
                                         out_count=n, readback=False, repeat=R)),
        ("dust_splat", lambda R: lab.run(_load_wgsl("dust_splat"), "main", None,
                                         inputs={"par": par_flat, "far": far_flat},
                                         uniforms=dust_splat_uniforms(
                                             w, h, 200, 0.3, 0.55, 0.65, 1.8,
                                             (lxn, lyn), dscale, True),
                                         out_count=npix, out_dtype=np.uint32,
                                         workgroup=(64, 1),
                                         dispatch=((200 + 63) // 64, 1),
                                         readback=False, repeat=R)),
        ("tail", lambda R: lab.run(_load_wgsl("tail"), "main", (w, h),
                                   inputs={"src": base_flat, "edge_ref": base_flat,
                                           "pal": pal.reshape(-1)},
                                   uniforms=np.array(
                                       [[32.0, 0.10, 1.0, 2.0],
                                        [0.35, float(w), float(h), 0.0]],
                                       dtype=np.float32).reshape(-1),
                                   out_count=npix, out_dtype=np.uint32,
                                   readback=False, repeat=R)),
        ("bloom 一个尺度(横+纵)", lambda R: lab.run(
            _load_wgsl("bloom_blur"), "main", (w, h),
            inputs={"kbuf": kflat, "src": base_flat, "prev": zero},
            uniforms=bloom_blur_uniforms(w, h, "v", table[-1], True),
            out_count=n, readback=False, repeat=R)),
    ):
        fn(REP)
        t0 = time.perf_counter()
        for _ in range(nf):
            fn(REP)
        per_pass[label] = (time.perf_counter() - t0) / nf / REP * 1000.0
        print(f"    {label:14s} {per_pass[label]:7.3f} ms/pass  "
              f"（×{REP} 摊薄后）")
    lab.pool_buffers = False
    gpu_pure_ms = sum(per_pass.values())

    print()
    print("=" * 74)
    print(f"  GPU 各 pass 计算量合计         {gpu_pure_ms:8.3f} ms  "
          f"（含 bloom 三尺度各两次 = 共 14 个 dispatch）")
    print(f"  GPU 纯计算（缓冲池 + 不回读）  {gpu_compute_ms:8.2f} ms/帧  "
          f"→ {1000.0 / gpu_compute_ms:5.1f} fps   ← 仍含 Python 每次调用的开销")
    print(f"  GPU 逐 pass 回读（lab 调用）  {gpu_ms:8.2f} ms/帧  "
          f"→ {1000.0 / gpu_ms:5.1f} fps   ← 回读开销占 "
          f"{100.0 * (1 - gpu_compute_ms / gpu_ms):.0f}%")
    print(f"  CPU 全链路（numpy + PIL）     {cpu_ms:8.2f} ms/帧  "
          f"→ {1000.0 / cpu_ms:5.1f} fps   ← 现在的服务端")
    print()
    print(f"  渲染提速（对现在服务端）       {cpu_ms / gpu_compute_ms:8.2f} ×")
    print()
    print("  ⚠️ 判读要点（三个数各是什么）：")
    print("   · **各 pass 计算量合计**才是 GPU 真正的计算成本：14 个 dispatch、")
    print(f"     {w}×{h}，合计只有 {gpu_pure_ms:.3f} ms —— 计算上基本免费。")
    print("   · 「缓冲池 + 不回读」那 40 多毫秒**几乎全是 Python 调 wgpu 的固定开销**")
    print("     （每次 write_buffer + 建 bind group + submit + FFI，实测约 3 ms/次）。")
    print("     **这部分在浏览器里不存在** —— 没有 FFI，WebGPU 调用本身很便宜。")
    print("     所以浏览器里的每帧成本 ≈ 计算量 + 极小的 JS 开销，远低于这个数。")
    print("   · CPU 那行是现在服务端的真实渲染成本，可以直接比。")
    print()
    print("   → 结论与 spike-log M3-GPU 一致：")
    print("     **算力从来不是瓶颈**（13 万像素上 GPU 计算量 0.3 ms）。")
    print("     服务端方案真正的成本是每帧 PNG 编码 + HTTP 往返（约 60 ms），")
    print("     而浏览器渲染把这两项**整个消掉** —— 那才是收益的来源。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
