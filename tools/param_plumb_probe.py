"""系统性排查：「参数只对单帧生效、对整段序列不生效」这类 bug。

用户报障：尘埃控件似乎只影响静止帧，一点播放就跳回预设效果。
并怀疑这是**一类广泛存在的 bug**（参数只接了单帧路径，没接整段路径）。

这个怀疑很值得认真查 —— 因为渲染有**两条路径**：

    单帧路径   render_still()   → 拖滑杆时用
    整段路径   render_preview() → 播放/出图时用
    再加上界面自己维护的 `frames[]` URL 列表

任何一条漏接，症状都是"调参时看着对、播放时不对"。

这里对 **TuneParams 的每一个字段**逐个做实验：
  改这个字段 → 单帧输出变了吗？整段输出变了吗？
把"改了但某条路径没反应"的字段全部列出来。

⚠️ 判据必须是"输出变了"，不能只看"参数传进去了" ——
   参数传进去但下游没用（比如被 scene.tail 的快照覆盖）一样是 bug。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pixelart.paths import INPUT  # noqa: E402

from pixelart.palette import palette_from_frames  # noqa: E402
from pixelart.tune import TuneParams, build_scene, render_preview, render_still  # noqa: E402


#: 每个字段给一个"明显不同"的取值。
#: ⚠️ 不能随便给：要选**真的会改变画面**的值，否则测出来是"没变化"但其实是取值没意义。
PROBE_VALUE: dict[str, object] = {
    "grid_long": 160,
    "work_long": 640,
    "aspect": "16:9",
    "levels": 0.0,
    "sharpen": 0.0,
    "var_gain": 5.0,
    "density": 1.6,
    "power": 1.0,
    "fog_r": 0.9, "fog_g": 0.2, "fog_b": 0.2,
    "fog_sat": 0.0,
    "fog_tint": 0.0,
    "cone_x": 0.85,
    "cone_y": 0.15,
    "light2_on": True,
    "light2_x": 0.15,
    "light2_y": 0.75,
    "light2_gain": 2.0,
    "light2_spread": 0.5,
    "sky_on": True,
    "sky_mode": "dusk",
    "sky_stars": 2.0,
    "detail": 0.7,
    "quantize_mode": "continuous",
    "rays": 1.4,
    "rays_auto_center": False,
    "rays_x": 0.15, "rays_y": 0.15,
    "rays_sat": 0.0,
    "rays_spread": 0.0,
    "rays_r": 0.9, "rays_g": 0.3, "rays_b": 0.9,
    "bloom": 1.8,
    "colors": 8,
    "dither": 0.35,
    "edge_strength": 0.0,
    "edge_gain": 6.0,
    "fog_drift": 0.9,
    "flicker": 0.5,
    "dust_count": 0,
    "dust_bright": 1.4,
    "dust_twinkle": 0.0,
    "dust_fade_far": 0.0,
    "dust_light_boost": 3.0,
    "seconds": 1.5,
    "fps": 12,
}

#: 只在"场景构建期"生效的字段 —— 改了它们必须**重建场景**才有效果，
#: 所以探针要重建 scene，否则测出来是"没变化"（那是探针的问题，不是 bug）。
SCENE_ONLY = {"grid_long", "work_long", "aspect", "levels", "sharpen", "var_gain",
              "detail", "sky_on", "sky_mode", "sky_stars"}

#: 只影响"帧数/帧率"、不影响单帧画面的字段
COUNT_ONLY = {"seconds", "fps"}

#: ⚠️ AUTO 哨兵是**整组**生效的：只设 fog_r 而 fog_g/fog_b 仍是 -1，
#: 则 fog_color() 仍返回 None（走自动）。探针必须整组一起改。
_FOG_RGB = {"fog_r": 0.85, "fog_g": 0.20, "fog_b": 0.25}
_RAYS_RGB = {"rays_r": 0.85, "rays_g": 0.30, "rays_b": 0.85}
_RAYS_XY = {"rays_x": 0.15, "rays_y": 0.15, "rays_auto_center": False}
# ⚠️ 锥顶点只有锥开着才影响画面 → 必须整组一起改（含 rays_cone/rays_shaft）
_CONE_XY = {"cone_x": 0.85, "cone_y": 0.15, "rays_cone": 0.7,
            "rays_dir": -30.0, "rays_shaft": 1.2}
# ⚠️ 副光源字段同理：light2_on=False 时其余全部无效果 → 整组开关一起改
_LIGHT2 = {"light2_on": True, "light2_x": 0.15, "light2_y": 0.75,
           "light2_gain": 2.0, "light2_spread": 0.5}
# ⚠️ 天空模式/星密度只有在 sky_on 开着时才影响画面 → 成组
_SKY = {"sky_on": True, "sky_mode": "dusk", "sky_stars": 2.0}

GROUPS: dict[str, dict[str, object]] = {
    "fog_r": _FOG_RGB, "fog_g": _FOG_RGB, "fog_b": _FOG_RGB,
    "rays_r": _RAYS_RGB, "rays_g": _RAYS_RGB, "rays_b": _RAYS_RGB,
    "rays_x": _RAYS_XY, "rays_y": _RAYS_XY,
    "cone_x": _CONE_XY, "cone_y": _CONE_XY,
    "light2_on": _LIGHT2, "light2_x": _LIGHT2, "light2_y": _LIGHT2,
    "light2_gain": _LIGHT2, "light2_spread": _LIGHT2,
    "sky_mode": _SKY, "sky_stars": _SKY,
}


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """平均绝对差。**形状不同直接算作"生效"**（网格变了画面当然变了）。"""
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def main() -> int:
    img = Image.open(INPUT / "ref10_green_cliff.jpg")   # 带天空的素材：天空参数的探针需要它

    base = TuneParams(grid_long=240, work_long=960, seconds=1.5, fps=12)
    sc = build_scene(img, base)
    still0, _ = render_still(sc, base)
    prev0, _, _ = render_preview(sc, base, preview_fps=4, include_last=False)

    print(f"基准：网格 {sc.grid}  单帧 {still0.shape}  整段 {len(prev0)} 帧\n")
    print(f"{'字段':20s} {'单帧有变化':>10s} {'整段有变化':>10s}  判定")
    print("-" * 62)

    broken: list[tuple[str, str]] = []
    for f in dataclasses.fields(TuneParams):
        name = f.name
        if name not in PROBE_VALUE:
            print(f"{name:20s} {'(未提供探测值)':>10s}")
            continue

        if name in COUNT_ONLY:
            print(f"{name:20s} {'—':>10s} {'—':>10s}  （只影响帧数，不验）")
            continue

        overrides = dict(GROUPS.get(name, {})) or {name: PROBE_VALUE[name]}
        p2 = dataclasses.replace(base, **overrides)
        note = ""
        # 场景构建期的字段：必须重建场景才有效果
        sc2 = build_scene(img, p2) if name in SCENE_ONLY else sc
        try:
            still1, _ = render_still(sc2, p2)
            d_still = diff(still0, still1)
        except Exception as e:                                  # noqa: BLE001
            d_still = -1.0
            note = f"单帧异常:{type(e).__name__} "
        try:
            prev1, _, _ = render_preview(sc2, p2, preview_fps=4, include_last=False)
            n = min(len(prev0), len(prev1))
            d_prev = max(diff(prev0[i], prev1[i]) for i in range(n)) if n else -1.0
        except Exception as e:                                  # noqa: BLE001
            d_prev = -1.0
            note += f"整段异常:{type(e).__name__}"

        ok_s = d_still > 1e-9
        ok_p = d_prev > 1e-9
        tag = "✅" if (ok_s and ok_p) else "❌"
        if not (ok_s and ok_p):
            broken.append((name, f"单帧{ok_s} 整段{ok_p}"))
        print(f"{name:20s} {d_still:10.4f} {d_prev:10.4f}  {tag} {note}")

    print("\n" + "=" * 62)
    if broken:
        print("❌ 有字段没有同时影响两条路径：")
        for name, why in broken:
            print(f"    {name}: {why}")
    else:
        print(f"✅ 全部 {len(PROBE_VALUE)} 个字段都同时影响单帧与整段")
    print("=" * 62)
    print()
    print("⚠️ 注意：这个探针只覆盖 **Python 侧**的两条路径。")
    print("   界面还有第三条路径 —— 它自己维护的 `frames[]` URL 列表。")
    print("   如果改了参数而没让 frames[] 失效，播放时就会放**旧参数渲出的帧** ——")
    print("   症状正是用户报的「调参看着对、一点播放就跳回预设效果」。")
    print("   那条路径在 tools/m3_uicheck.py 里查。")
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
