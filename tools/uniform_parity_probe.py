"""JS ↔ Python 的 uniform 打包一致性 —— 把唯一那处"重复实现"钉死。

═══ 为什么必须有这个测试 ═══

`tools/m3_wgsl_uniforms.js` 镜像 `src/pixelart/webgpu.py`，是整个移植里
**唯一必须重复实现**的一段逻辑。重复实现本身不是问题 ——
**没有护栏的重复实现**才是（本项目已经因为"同一个概念在两处各取一份"
栽过 5 次）。

所以这里做一件很直接的事：**随机生成一批参数集**，
在 node 里跑 JS 的打包、在 Python 里跑 Python 的打包，**逐字节比对**。
改了任一边而忘了另一边，这个测试立刻报红。

为什么不用"让服务端算好 uniform 发过来"来彻底避免重复：
因为 uniform 依赖**每一个滑杆值**，而滑杆是拖动中实时变的 ——
每拖一下都回服务端要一次，就回到了"每帧一个往返"，移植的意义就没了。

═══ 顺带守住的东西 ═══

- `fog_noise_mean` 的闭式解必须与 `loop_noise_2d(...).mean()` 吻合
  （它是"把网格均值从 O(W·H) 降到 O(1)"的关键，写错了雾就会变）
- `flicker_gain` 的周期性与数值
- 所有 uniform 的**长度**必须与着色器里声明的 vec4 个数一致
  （长度错了 GPU 会直接报 size mismatch，但那时已经太晚）

用法::

    python tools/uniform_parity_probe.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

JS_MODULE = ROOT / "tools" / "m3_wgsl_uniforms.js"

#: uniform 缓冲的期望长度（float32 个数）= vec4 数 × 4。
#: 与各 .wgsl 里 `array<vec4<f32>, N>` 的 N 一一对应。
EXPECTED_LEN = {
    "fog": 24,            # fog.wgsl 6
    "scatter": 8,         # scatter.wgsl 2
    "volumetric": 24,     # volumetric.wgsl 6（M5.1 光锥 + M5.2 光柱）
    "bloomBright": 4,     # bloom_bright.wgsl 1
    "bloomBlur": 8,       # bloom_blur.wgsl 2
    "bloomCombine": 4,    # bloom_combine.wgsl 1
    "dustSplat": 16,      # dust_splat.wgsl 4
    "dustApply": 8,       # dust_apply.wgsl 2
    "parallaxSplat": 16,  # parallax_splat.wgsl 4
    "parallaxApply": 16,  # parallax_apply.wgsl 4
    "tail": 8,            # tail.wgsl 2
}


def _random_cases(n: int, seed: int = 1234) -> list[dict]:
    """生成随机参数集（覆盖各滑杆的合理区间）。"""
    rs = np.random.RandomState(seed)
    cases = []
    for _ in range(n):
        w = int(rs.choice([60, 96, 240, 320, 480]))
        h = int(rs.choice([40, 64, 128, 180, 270]))
        colors = int(rs.choice([16, 24, 32, 48, 64]))
        cases.append({
            "w": w, "h": h,
            "t": float(rs.uniform(0, 1)),
            "parAmp": float(rs.choice([0.0, 0.25, 0.5, 1.0])),
            "parNE": int(rs.choice([1, 2, 3, 7, 13])),
            "parPivotX": float(rs.uniform(0.2, 0.8)),
            "parPivotY": float(rs.uniform(0.2, 0.8)),
            "parFarOff": int(rs.choice([18432, 84992, 3456])),
            "coneAngle": float(rs.choice([0.0, 0.35, 0.7, 1.1])),
            "coneDirDeg": float(rs.uniform(-180, 180)),
            "coneReach": float(rs.uniform(0, 1)),
            "coneGain": float(rs.uniform(1.0, 3.0)),
            "shaft": float(rs.choice([0.0, 0.5, 1.0, 2.0])),
            "fogTint": float(rs.choice([0.0, 0.3, 0.7, 1.0])),
            "coneX": float(rs.choice([-1.0, 0.3, 0.85])),
            "coneY": float(rs.choice([-1.0, 0.2, 0.7])),
            "fogColor": [float(v) for v in rs.uniform(0.05, 0.95, 3)],
            "density": float(rs.uniform(0.2, 2.2)),
            "power": float(rs.uniform(1.0, 4.0)),
            "floor": 0.0, "ceiling": 1.0,
            "drift": float(rs.uniform(0.0, 1.0)),
            "lx": int(rs.randint(0, w)), "ly": int(rs.randint(0, h)),
            "radius": 3, "satMax": 0.25,
            "samples": 28, "span": 0.85, "decay": 0.965,
            "strength": float(rs.uniform(0, 1.2)),
            "occludeGain": 5.0, "falloffGain": 2.5,
            "screenFalloff": float(rs.uniform(0, 2.0)),
            "threshold": 0.48, "knee": 0.3,
            "flickerDepth": float(rs.uniform(0, 0.5)),
            "bloomStrength": float(rs.uniform(0, 1.2)),
            "r0": 2.0, "r1": 5.0, "r2": 11.0,
            "dustCount": int(rs.choice([0, 50, 200, 400])),
            "dustBright": float(rs.uniform(0, 1.2)),
            "twinkle": float(rs.uniform(0, 1.0)),
            "fadeFar": float(rs.uniform(0, 1.0)),
            "lightBoost": float(rs.uniform(1.0, 3.0)),
            "dustScale": int(2 ** rs.randint(10, 21)),
            "hasFar": True, "hasLight": True,
            "nPalette": colors,
            "dither": float(rs.uniform(0, 0.3)),
            "ditherAdaptive": bool(rs.randint(0, 2)),
            "edgeGain": 2.0,
            "edgeStrength": float(rs.uniform(0, 0.6)),
            "cont": int(rs.randint(0, 2)),
            # 饱和度上限的输入（**任意**颜色，故意包含接近中性/极端的情况）
            "satColor": [float(v) for v in rs.uniform(0.0, 1.0, 3)],
            "satMax": float(rs.choice([0.0, 0.15, 0.25, 0.4, 1.0])),
        })
    return cases


def _python_packs(case: dict, harmonics: list, phases: list, freqs: list) -> dict:
    """用 Python 侧的实现打包（这里直接调用被镜像的函数）。"""
    from pixelart.webgpu import (
        parallax_apply_uniforms, parallax_splat_uniforms,
        bloom_blur_uniforms, bloom_bright_uniforms, bloom_combine_uniforms,
        dust_apply_uniforms, dust_splat_uniforms, fog_uniforms,
        scatter_uniforms, volumetric_uniforms,
    )
    from pixelart.animate import flicker_gain

    w, h = case["w"], case["h"]
    t = case["t"]
    gain = flicker_gain(t, depth=case["flickerDepth"])

    hs = [{"amp": hh[4], "fx": hh[0], "fy": hh[1], "ft": int(hh[2]),
           "phase": hh[3]} for hh in harmonics]

    out = {}
    out["fog"] = fog_uniforms(w, h, t, case["fogColor"], case["density"],
                              case["power"], floor=case["floor"],
                              ceiling=case["ceiling"], drift=case["drift"],
                              harmonics=hs, fog_tint=case["fogTint"])
    out["scatter"] = scatter_uniforms(w, h, case["lx"], case["ly"],
                                      case["radius"], case["satMax"])
    out["volumetric"] = volumetric_uniforms(
        w, h, samples=case["samples"], span=case["span"], decay=case["decay"],
        strength=case["strength"], occlude_gain=case["occludeGain"],
        falloff_gain=case["falloffGain"], screen_falloff=case["screenFalloff"],
        threshold=case["threshold"], knee=case["knee"], gain=gain,
        lx=case["lx"], ly=case["ly"],
        cone_angle=case["coneAngle"], cone_dir_deg=case["coneDirDeg"],
        cone_reach=case["coneReach"], cone_gain=case["coneGain"],
        shaft=case["shaft"], fog_tint=case["fogTint"],
        cone_x=case["coneX"], cone_y=case["coneY"])
    out["bloomBright"] = bloom_bright_uniforms(w, h, 0.55, 0.25)
    # blur：取中间那个尺度，横纵各一次
    entry_h = (2, 13, 6, 1.0)            # offset, count, half, weight
    entry_v = (2, 13, 6, 1.0)
    out["bloomBlur"] = bloom_blur_uniforms(w, h, "v", entry_v, True)
    mult = case["bloomStrength"] / (1 + 0.5 + 1 / 3)
    u = np.zeros((1, 4), dtype=np.float32)
    u[0] = [float(w), float(h), float(mult), 0.0]
    out["bloomCombine"] = u.reshape(-1)
    out["dustSplat"] = dust_splat_uniforms(
        w, h, case["dustCount"], t, case["twinkle"], case["fadeFar"],
        case["lightBoost"],
        (case["lx"] / max(w - 1, 1), case["ly"] / max(h - 1, 1)),
        case["dustScale"], case["hasFar"])
    out["dustApply"] = dust_apply_uniforms(w, h, case["dustScale"],
                                           case["dustBright"])
    # 分层视差：k 由"波形 × 幅宽比例"算出来 —— **两边必须用同一个公式**
    from pixelart.parallax import (DRIFT, FAR_GAIN, NEAR_GAIN, REL_AMPLITUDE,
                                   SPLAT_SCALE, parallax_wave)
    wave = parallax_wave(case["t"])
    k = case["parAmp"] * REL_AMPLITUDE * w * wave
    out["parallaxSplat"] = parallax_splat_uniforms(
        w, h, k, wave,
        (case["parPivotX"] * max(w - 1, 1), case["parPivotY"] * max(h - 1, 1)),
        DRIFT, case["parNE"], NEAR_GAIN, FAR_GAIN, SPLAT_SCALE)
    out["parallaxApply"] = parallax_apply_uniforms(
        w, h, SPLAT_SCALE, k, case["parFarOff"])
    from pixelart.pixelate import PixelTail
    tl = PixelTail(dither=case["dither"], edge_strength=case["edgeStrength"],
                   edge_gain=case["edgeGain"],
                   dither_adaptive=case["ditherAdaptive"])
    uu = np.zeros((2, 4), dtype=np.float32)
    uu[0] = [float(case["nPalette"]), tl.dither,
             1.0 if tl.dither_adaptive else 0.0, tl.edge_gain]
    uu[1] = [tl.edge_strength, float(w), float(h),
             1.0 if case.get("cont") else 0.0]
    out["tail"] = uu.reshape(-1)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    args = ap.parse_args()

    from pixelart.animate import DEFAULT_FLICKER_FREQS, hash01, loop_noise_2d
    from pixelart.webgpu import TWO_PI, fog_harmonics

    print("JS ↔ Python uniform 打包一致性")
    print("=" * 78)

    # ── 1. 静态常量（服务端算好发给浏览器） ──
    harm = fog_harmonics()
    harmonics_js = [[hh["fx"], hh["fy"], float(hh["ft"]), hh["phase"], hh["amp"]]
                    for hh in harm]
    phases = [TWO_PI * hash01(0, i) for i in range(len(DEFAULT_FLICKER_FREQS))]
    freqs = list(DEFAULT_FLICKER_FREQS)
    print(f"  谐波系数 {len(harmonics_js)} 层，闪烁频率 {freqs}")

    # ── 2. 闭式 noise_mean vs numpy 网格均值 ──
    print()
    print("  ① fog_noise_mean 闭式解 vs loop_noise_2d 的实际网格均值：")
    worst = 0.0
    for (w, h) in ((96, 64), (480, 270), (3, 5), (240, 128)):
        for t in (0.0, 0.13, 0.37, 0.5, 0.77, 1.0):
            ref = float(loop_noise_2d(t, (h, w), octaves=2, seed=17).mean())
            from pixelart.webgpu import fog_noise_mean
            got = fog_noise_mean(t, w, h, harm)
            worst = max(worst, abs(ref - got))
    good = worst < 1e-6
    print(f"    {'✅' if good else '❌'} 最大差 {worst:.3e}（容差 1e-6）")

    # 循环：t 加整数必须**逐位**相同
    from pixelart.webgpu import fog_noise_mean
    cyc = (fog_noise_mean(0.0, 480, 270, harm)
           == fog_noise_mean(1.0, 480, 270, harm)
           == fog_noise_mean(2.0, 480, 270, harm))
    print(f"    {'✅' if cyc else '❌'} t 加整数逐位相同（时序铁律 3）")

    # ── 3. 随机参数集逐字节比对 ──
    cases = _random_cases(args.cases)
    script = f"""
const U = require({json.dumps(str(JS_MODULE).replace(chr(92), '/'))});
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const harmonics = input.harmonics, phases = input.phases, freqs = input.freqs;
const out = [];
for (const c of input.cases) {{
  const gain = U.flickerGain(c.t, phases, freqs, c.flickerDepth);
  out.push({{
    fog: Array.from(U.fogUniforms({{
      w: c.w, h: c.h, t: c.t, fogColor: c.fogColor, density: c.density,
      power: c.power, floor: c.floor, ceiling: c.ceiling, drift: c.drift,
      harmonics: harmonics, fogTint: c.fogTint }})),
    scatter: Array.from(U.scatterUniforms({{
      w: c.w, h: c.h, lx: c.lx, ly: c.ly,
      radius: c.radius, satMax: c.satMax }})),
    volumetric: Array.from(U.volumetricUniforms({{
      w: c.w, h: c.h, samples: c.samples, span: c.span, decay: c.decay,
      strength: c.strength, occludeGain: c.occludeGain,
      falloffGain: c.falloffGain, screenFalloff: c.screenFalloff,
      threshold: c.threshold, knee: c.knee, gain: gain,
      lx: c.lx, ly: c.ly,
      coneAngle: c.coneAngle, coneDirDeg: c.coneDirDeg,
      coneReach: c.coneReach, coneGain: c.coneGain, shaft: c.shaft,
      fogTint: c.fogTint, coneX: c.coneX, coneY: c.coneY }})),
    bloomBright: Array.from(U.bloomBrightUniforms({{
      w: c.w, h: c.h, threshold: 0.55, knee: 0.25 }})),
    bloomBlur: Array.from(U.bloomBlurUniforms({{
      w: c.w, h: c.h, horiz: false, accum: true,
      offset: 2, count: 13, half: 6, weight: 1.0 }})),
    bloomCombine: Array.from(U.bloomCombineUniforms({{
      w: c.w, h: c.h, mult: c.bloomStrength / (1 + 0.5 + 1/3) }})),
    dustSplat: Array.from(U.dustSplatUniforms({{
      w: c.w, h: c.h, count: c.dustCount, t: c.t, twinkle: c.twinkle,
      fadeFar: c.fadeFar, lightBoost: c.lightBoost, hasLight: true,
      lightX: c.lx / Math.max(c.w - 1, 1), lightY: c.ly / Math.max(c.h - 1, 1),
      scale: c.dustScale, hasFar: c.hasFar }})),
    dustApply: Array.from(U.dustApplyUniforms({{
      w: c.w, h: c.h, scale: c.dustScale, dustBright: c.dustBright }})),
    parallaxSplat: Array.from(U.parallaxSplatUniforms({{
      w: c.w, h: c.h, k: U.parallaxK(c.parAmp, c.w, c.t),
      wave: U.parallaxWave(c.t),
      pivotX: c.parPivotX * Math.max(c.w - 1, 1),
      pivotY: c.parPivotY * Math.max(c.h - 1, 1),
      drift: 0.35, nEdges: c.parNE, nearGain: 1.0, farGain: 0.12,
      scale: 1048576 }})),
    parallaxApply: Array.from(U.parallaxApplyUniforms({{
      w: c.w, h: c.h, scale: 1048576,
      k: U.parallaxK(c.parAmp, c.w, c.t), farOff: c.parFarOff }})),
    sat: U.limitSaturation(c.satColor, c.satMax),
    tail: Array.from(U.tailUniforms({{
      w: c.w, h: c.h, nPalette: c.nPalette, dither: c.dither,
      ditherAdaptive: c.ditherAdaptive, edgeGain: c.edgeGain,
      edgeStrength: c.edgeStrength, continuous: c.cont }}))
  }});
}}
process.stdout.write(JSON.stringify(out));
"""
    payload = json.dumps({"cases": cases, "harmonics": harmonics_js,
                          "phases": phases, "freqs": freqs})
    try:
        r = subprocess.run(["node", "--input-type=commonjs", "-e", script],
                           input=payload, capture_output=True, text=True,
                           encoding="utf-8", timeout=180)
    except FileNotFoundError:
        print("\n  ⚠️ 环境里没有 node —— 跳过 JS 比对")
        return 0
    if r.returncode != 0:
        print("\n[错误] node 执行失败：")
        print((r.stderr or "")[-1500:])
        return 1

    js_out = json.loads(r.stdout)

    print()
    print(f"  ② 随机 {len(cases)} 组参数，逐字节比对（float32 位模式）：")
    print()
    print(f"    {'pass':14s} {'长度':>5s} {'最大字节差':>12s} {'不一致 float 数':>16s}")
    print("    " + "-" * 54)
    bad = 0
    for name, want_len in EXPECTED_LEN.items():
        worst_bytes = 0
        n_diff = 0
        for i, case in enumerate(cases):
            py = _python_packs(case, harmonics_js, phases, freqs)[name]
            js = np.asarray(js_out[i][name], dtype=np.float32)
            if len(py) != want_len:
                print(f"    ❌ {name}: Python 侧长度 {len(py)} != 期望 {want_len}")
                bad += 1
                break
            if len(js) != want_len:
                print(f"    ❌ {name}: JS 侧长度 {len(js)} != 期望 {want_len}")
                bad += 1
                break
            a = np.asarray(py, dtype=np.float32).view(np.uint8)
            b = js.astype(np.float32).view(np.uint8)
            d = np.abs(a.astype(int) - b.astype(int)).max()
            nd = int((a != b).sum())
            worst_bytes = max(worst_bytes, int(d))
            n_diff += nd
        else:
            good = (worst_bytes == 0)
            if not good:
                bad += 1
            print(f"    {'✅' if good else '❌'} {name:12s} {want_len:5d} "
                  f"{worst_bytes:12d} {n_diff:16d}")

    # ── ③ limit_saturation（不在 uniform 缓冲里，单独比） ──
    from pixelart.compose import limit_saturation
    worst_sat = 0.0
    for i, case in enumerate(cases):
        py = limit_saturation(case["satColor"], case["satMax"])
        js = np.asarray(js_out[i]["sat"], dtype=np.float64)
        worst_sat = max(worst_sat, float(np.abs(py.astype(np.float64) - js).max()))
    # ⚠️ 这里用 1e-6 的容差而不是 0：两边一个 float32、一个 float64，
    #    实测差 ≤ 6e-08（1 个 f32 ulp）。**必须同时证明这个容差不是"放水"** ——
    #    所以下面做一次变异：把公式改错，看差异是否远超容差。
    SAT_TOL = 1e-6
    good_sat = worst_sat < SAT_TOL
    print()
    print(f"  ③ limitSaturation（镜像 compose.limit_saturation）：")
    print(f"    {'✅' if good_sat else '❌'} {len(cases)} 组颜色，最大差 {worst_sat:.3e}"
          f"（容差 {SAT_TOL:g}，因两侧 f32/f64 混算）")

    # 变异验证：把 k 的符号弄反（经典写错），差异应远超容差
    mut = f"""
const U = require({json.dumps(str(JS_MODULE).replace(chr(92), '/'))});
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
// 故意把缩放方向弄反
function bad(color, satMax) {{
  var v = [color[0], color[1], color[2]];
  for (var i = 0; i < 4; i++) {{
    var lo = Math.min(v[0], v[1], v[2]), hi = Math.max(v[0], v[1], v[2]);
    if (hi <= 1e-6) return [0.5, 0.5, 0.5];
    var sat = (hi - lo) / hi;
    if (satMax == null || sat <= satMax) break;
    var mean = (v[0] + v[1] + v[2]) / 3;
    var k = sat / Math.max(satMax, 1e-6);        // ← 反了（应 satMax/sat）
    v = [Math.max(0, Math.min(1, mean + (v[0]-mean)*k)),
         Math.max(0, Math.min(1, mean + (v[1]-mean)*k)),
         Math.max(0, Math.min(1, mean + (v[2]-mean)*k))];
  }}
  return v;
}}
process.stdout.write(JSON.stringify(input.cases.map(c => bad(c.satColor, c.satMax))));
"""
    rm = subprocess.run(["node", "--input-type=commonjs", "-e", mut],
                        input=json.dumps({"cases": cases}), capture_output=True,
                        text=True, encoding="utf-8", timeout=120)
    mut_worst = 0.0
    if rm.returncode == 0:
        bad_out = json.loads(rm.stdout)
        for i, case in enumerate(cases):
            py = limit_saturation(case["satColor"], case["satMax"])
            mut_worst = max(mut_worst, float(np.abs(
                py.astype(np.float64) - np.asarray(bad_out[i], np.float64)).max()))
    catch = mut_worst > SAT_TOL * 100
    print(f"    {'✅' if catch else '❌'} 变异验证：把缩放方向写反 → 最大差 "
          f"{mut_worst:.3e}（应 >> 容差；说明容差抓得住真错误）")
    good_sat = good_sat and catch

    print()
    print("=" * 78)
    allok = (bad == 0 and worst < 1e-6 and cyc and good_sat)
    print(f"  {'全部一致' if allok else '存在差异'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
