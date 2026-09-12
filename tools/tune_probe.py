"""tune 模块自检：参数序列化、预览渲染、自检统计。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelart.tune import (  # noqa: E402
    TuneParams,
    aspect_value,
    build_scene,
    display_scale,
    render_preview,
    render_still,
    render_sweep,
)


def main() -> int:
    print("=== 参数序列化往返 ===")
    p = TuneParams(density=0.8, rays_x=0.3, rays_auto_center=False, colors=24, aspect="16:9")
    qs = p.to_query()
    print(f"  query 片段: {qs[:100]} ...")
    p2 = TuneParams.from_query(dict(kv.split("=", 1) for kv in qs.split("&")))
    print(f"  round-trip 完全一致: {p.to_dict() == p2.to_dict()}")
    print(f"  bool '0'/'1' -> {TuneParams.from_query({'rays_auto_center': '0'}).rays_auto_center}"
          f" / {TuneParams.from_query({'rays_auto_center': '1'}).rays_auto_center}")
    print(f"  非法值被忽略（不抛）: density={'abc'} -> {TuneParams.from_query({'density': 'abc'}).density}")
    print(f"  未知字段被忽略: {TuneParams.from_query({'nonsense': '1'}).density}")

    print("\n=== aspect 解析 ===")
    for a in ("native", "16:9", "2.16", "0.603"):
        print(f"  {a!r:10s} -> {aspect_value(a):.4f}")

    print("\n=== 颜色 AUTO 语义 ===")
    d = TuneParams()
    print(f"  默认（全 AUTO）雾色 -> {d.fog_color()}   散射色 -> {d.rays_color()}")
    e = TuneParams(fog_r=0.1, fog_g=0.2, fog_b=0.3)
    print(f"  显式雾色 -> {e.fog_color()}")

    print("\n=== 预览渲染（240 网格）===")
    img = Image.open("assets/input/ref04_lain_room.jpg")
    t0 = time.perf_counter()
    sc = build_scene(img, TuneParams())
    t_build = time.perf_counter() - t0
    print(f"  build_scene {t_build:.2f}s   网格 {sc.grid}  x{sc.scale}  "
          f"输出 {sc.out_size}  display_scale={display_scale(sc)}")

    for fps in (8, 12):
        t0 = time.perf_counter()
        frames, pal, st = render_preview(sc, TuneParams(), preview_fps=fps)
        dt = time.perf_counter() - t0
        print(f"  render_preview preview_fps={fps:2d} -> {len(frames):3d} 帧  {dt:5.2f}s"
              f"  (保持 {TuneParams().seconds:.0f}s 时长)")

    print(f"\n  自检：循环差 {st['loop_diff']} (逐位闭合={st['loop_ok']})"
          f"   色板 {st['palette_size']} 色   越界 {st['colors_outside']}")
    a = st["amplitude"]
    print(f"  动画幅度：静止 {a['static'] * 100:.0f}%  微弱 {a['faint'] * 100:.0f}%"
          f"  看得出 {a['visible'] * 100:.0f}%  明显 {a['strong'] * 100:.0f}%   max {a['max']}")
    m = st["motion"]
    print(f"  相邻帧差 {m['adj_min']:.3f} ~ {m['adj_max']:.3f}   半周期 {m['half_period']:.3f}")
    print(f"  光源（用到的）{st['light_xy']} -> {[round(v, 2) for v in st['light_xy_used']]}")
    print(f"  雾色 {[round(v, 3) for v in st['fog_color']]} auto={st['fog_color_auto']}")

    t0 = time.perf_counter()
    render_still(sc, TuneParams())
    print(f"\n  render_still {1000 * (time.perf_counter() - t0):.0f} ms （拖滑杆的即时反馈）")

    print("\n=== 参数扫描 ===")
    sw = render_sweep(sc, TuneParams(), "density", [0.4, 0.7, 1.1, 1.4])
    print(f"  4 个取值 -> {[v for v, _ in sw]}   帧形状 {sw[0][1].shape}")

    print("\n=== 静态参数必须真的静态 ===")
    z = TuneParams(fog_drift=0.0, flicker=0.0, dust_count=0)
    _, _, sz = render_preview(sc, z, preview_fps=6, include_last=False)
    az = sz["amplitude"]
    print(f"  全关时静止占比 {az['static'] * 100:.1f}%  相邻帧差均值 {sz['motion']['adj_mean']:.4f}")
    print("  （应当 100% 静止、帧差 0 —— 除量化噪声外）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
