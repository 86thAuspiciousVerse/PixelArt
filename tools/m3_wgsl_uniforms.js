/* WGSL uniform 打包 —— 浏览器与 node 共用，纯函数、无 DOM。
 *
 * ⚠️⚠️ 这个文件**镜像** src/pixelart/webgpu.py。它是整个移植里唯一
 *     "必须重复实现"的一段逻辑，所以用 tools/uniform_parity_probe.py
 *     把它**钉死**：随机生成参数集，在 node 里跑一遍、在 Python 里跑一遍，
 *     逐字节比对 uniform 缓冲区。改了任一边而没改另一边，测试立刻报红。
 *
 * 为什么还是选择重复实现（而不是让服务端算好发过来）：
 *   uniform 依赖**每一个滑杆值**，而滑杆是拖动中实时变化的。
 *   每拖一下都回服务端要一次 uniform → 又回到"每帧一个往返"。
 *   所以必须在本地算 —— 那就必须有一份 JS 实现，
 *   而"能被逐字节比对"的重复实现是可控的。
 *
 * ⚠️ 但**静态常量不重复**：fog 的谐波系数、闪烁的相位，都由服务端算好放进
 *     场景包。所以这里没有 splitmix64 —— 不重复"哈希"这种容易写错的东西。
 *
 * 各个 uniform 的布局在对应的 .wgsl 文件顶部有说明（权威）。
 */
(function (root) {
  'use strict';

  var TWO_PI = 6.283185307179586;

  /* ──────────────────────────────────────────────────────────
   * 噪声场的网格均值（O(1) 闭式）
   *
   * ⚠️ 镜像 webgpu.py::fog_noise_mean。**不能**改成"每帧遍历 W×H 算一遍"：
   *    480×270 就是 13 万点 × 2 个 sin，每帧白烧几毫秒。
   *    也不能改成常量或查找表：这个量在 t 上峰峰有 0.27，近似会看出来。
   *
   * 推导见 Python 侧的文档字符串。要点：
   *   (1/n)·Σ_m e^{i·2π·a·(m+0.5)/n}
   *     = e^{iπa}·sin(πa) / (n·sin(πa/n))        （等比级数 + 半角公式）
   *   当 a/n 为整数时退化，每个都是 e^{iπa} → (-1)^(a/n)。
   * ────────────────────────────────────────────────────────── */
  function gridPhasor(a, n) {
    var m = a / n;
    var r = Math.round(m);
    if (Math.abs(m - r) < 1e-9) return { re: (r % 2 === 0 ? 1 : -1), im: 0 };
    var pa = Math.PI * a;
    var re = Math.cos(pa), im = Math.sin(pa);           // e^{iπa}
    var s = Math.sin(pa) / (n * Math.sin(pa / n));      // 实标量
    return { re: re * s, im: im * s };
  }

  function cmul(x, y) {
    return { re: x.re * y.re - x.im * y.im, im: x.re * y.im + x.im * y.re };
  }

  function fogNoiseMean(t, w, h, harmonics) {
    var acc = 0, ampSum = 0;
    for (var k = 0; k < harmonics.length; k++) {
      var hh = harmonics[k];
      var ft = hh[2] | 0;
      var v = ft * t;
      var phT = TWO_PI * (v - Math.floor(v));           // ⚠️ fract(ft·t)
      var e = { re: Math.cos(hh[3] + phT), im: Math.sin(hh[3] + phT) };
      var m = cmul(cmul(e, gridPhasor(hh[0], w)), gridPhasor(hh[1], h));
      acc += hh[4] * m.im;
      ampSum += hh[4];
    }
    return acc / Math.max(ampSum, 1e-9);
  }

  /* ──────────────────────────────────────────────────────────
   * 光源闪烁增益
   *
   * ⚠️ 镜像 animate.flicker_gain：整数频率正弦叠加，严格以 1 为周期。
   *    每帧只有一个标量，所以**不能**放进着色器让每个像素重复算。
   *    phases 由服务端按 hash01(seed, i) 算好传来，这里只做加法与 sin。
   * ────────────────────────────────────────────────────────── */
  function flickerGain(t, phases, freqs, depth) {
    if (!(depth > 0)) return 1;
    var acc = 0, wsum = 0;
    for (var i = 0; i < freqs.length; i++) {
      var w = 1 / (i + 1);
      acc += w * Math.sin(TWO_PI * freqs[i] * t + phases[i]);
      wsum += w;
    }
    return 1 + depth * (acc / Math.max(wsum, 1e-9));
  }

  /* ──────────────────────────────────────────────────────────
   * 饱和度上限（镜像 compose.limit_saturation）
   *
   * ⚠️ 这一段**一度漏在探针覆盖之外** —— 我在渲染器里另写了一份，
   *    而 uniform_parity_probe 只比对 uniform 缓冲，管不到它。
   *    "没被测试钉住的重复实现"正是本项目栽过 5 次的那类问题，
   *    所以把它挪到这里、并加进比对清单。
   * ────────────────────────────────────────────────────────── */
  function limitSaturation(color, satMax) {
    // ⚠️ 这一段**不追求逐位一致**，差在 1 个 f32 ulp（实测 ≤ 6e-08）。
    //
    //    原因：Python 侧 `limit_saturation` 在 float32 数组上算，
    //    但 `hi/lo/sat/mean/k` 会被 `float(...)` 提到 float64，
    //    再和 float32 数组混合运算 —— 每一步的舍入位置很难精确复刻。
    //    我试过给 JS 加 `Math.fround`，只把差异从 8.5e-08 压到 5.96e-08，
    //    做不到 0。
    //
    //    这不是逻辑错误，所以不值得继续追：**这个值只是一个雾色**，
    //    写进 float32 uniform，1 ulp 远低于它的存储精度，更远低于
    //    最终 8 位量化的 1/255。
    //
    //    所以探针对它用 **1e-6 的容差**（而不是 0），并且额外做了
    //    变异验证：把公式故意改错会差到 1e-2 量级，**远超容差** ——
    //    也就是说这个容差能抓住"逻辑错误"，只是放过"浮点末位"。
    var v = [color[0], color[1], color[2]];
    for (var i = 0; i < 4; i++) {
      var lo = Math.min(v[0], v[1], v[2]);
      var hi = Math.max(v[0], v[1], v[2]);
      if (hi <= 1e-6) return [0.5, 0.5, 0.5];
      var sat = (hi - lo) / hi;
      if (satMax == null || sat <= satMax) break;
      var mean = (v[0] + v[1] + v[2]) / 3;
      var k = satMax / sat;
      v = [
        Math.max(0, Math.min(1, mean + (v[0] - mean) * k)),
        Math.max(0, Math.min(1, mean + (v[1] - mean) * k)),
        Math.max(0, Math.min(1, mean + (v[2] - mean) * k))
      ];
    }
    return v;
  }

  /* ──────────────────────────────────────────────────────────
   * 各 pass 的 uniform
   *
   * 统一用 float32 数组返回（值会被隐式转成 f32，与 GPU 一致）。
   * 布局与对应的 .wgsl 顶部注释一一对应。
   * ────────────────────────────────────────────────────────── */

  // fog.wgsl —— 6 个 vec4
  function fogUniforms(o) {
    var u = new Float32Array(24);
    var c = o.fogColor;
    u[0] = c[0]; u[1] = c[1]; u[2] = c[2];
    u[3] = (o.fogTint === undefined) ? 1 : o.fogTint;
    u[4] = o.density; u[5] = o.power; u[6] = o.floor || 0;
    u[7] = 1 / Math.max(1e-6, (o.ceiling === undefined ? 1 : o.ceiling) - (o.floor || 0));
    u[8] = o.drift;
    u[9] = fogNoiseMean(o.t, o.w, o.h, o.harmonics);
    u[10] = o.w; u[11] = o.h;
    var amps = 0;
    for (var k = 0; k < 2; k++) {
      var hh = o.harmonics[k];
      u[12 + k * 4] = hh[0]; u[13 + k * 4] = hh[1];
      u[14 + k * 4] = hh[2]; u[15 + k * 4] = hh[3];
      amps += hh[4];
    }
    u[20] = o.harmonics[0][4]; u[21] = o.harmonics[1][4];
    u[22] = 1 / Math.max(amps, 1e-9);
    u[23] = o.t;
    return u;
  }

  // scatter.wgsl —— 2 个 vec4
  function scatterUniforms(o) {
    var u = new Float32Array(8);
    u[0] = o.lx; u[1] = o.ly; u[2] = o.w; u[3] = o.h;
    u[4] = o.radius; u[5] = o.satMax;
    return u;
  }

  // volumetric.wgsl —— 4 个 vec4
  function volumetricUniforms(o) {
    var u = new Float32Array(24);
    u[0] = o.samples; u[1] = o.span; u[2] = o.decay; u[3] = o.strength;
    u[4] = o.occludeGain; u[5] = o.falloffGain;
    u[6] = o.screenFalloff; u[7] = o.threshold;
    u[8] = o.knee; u[9] = o.gain; u[10] = o.w; u[11] = o.h;
    u[12] = o.lx; u[13] = o.ly;
    // ⭐ 锥顶点（M5.4）：U[3].zw，打包侧解析（<0=跟随光源 → 代入 lx/ly）。
    //    ⚠️ 舍入必须与 webgpu.py 同一字：floor(x+0.5)（不能用 Math.round ——
    //    与 Python round 的银行家舍入分叉，探针已抓到过）。
    u[14] = (o.coneX === undefined || o.coneX < 0) ? o.lx
          : Math.floor(o.coneX * (o.w - 1) + 0.5);
    u[15] = (o.coneY === undefined || o.coneY < 0) ? o.ly
          : Math.floor(o.coneY * (o.h - 1) + 0.5);
    // ── 光锥（M5.1）── ⚠️ 度 → 弧度：**必须**与 webgpu.py 用同一个公式
    //    （唯一的守门是 tools/uniform_parity_probe.py 的逐字节比对）。
    u[16] = o.coneAngle || 0;
    u[17] = (o.coneDirDeg || 0) * Math.PI / 180;
    u[18] = (o.coneReach === undefined) ? 0.8 : o.coneReach;
    u[19] = (o.coneGain === undefined) ? 1.6 : o.coneGain;
    // M5.2 光柱强度（独立加性层）
    u[20] = o.shaft || 0;
    u[21] = (o.fogTint === undefined) ? 1 : o.fogTint;
    return u;
  }

  // bloom_bright.wgsl —— 1 个 vec4
  function bloomBrightUniforms(o) {
    var u = new Float32Array(4);
    u[0] = o.w; u[1] = o.h; u[2] = o.threshold; u[3] = o.knee;
    return u;
  }

  // bloom_blur.wgsl —— 2 个 vec4
  function bloomBlurUniforms(o) {
    var u = new Float32Array(8);
    u[0] = o.w; u[1] = o.h;
    u[2] = o.horiz ? 0 : 1;
    u[3] = o.accum ? 1 : 0;
    u[4] = o.offset; u[5] = o.count; u[6] = o.half; u[7] = o.weight;
    return u;
  }

  // bloom_combine.wgsl —— 1 个 vec4
  function bloomCombineUniforms(o) {
    var u = new Float32Array(4);
    u[0] = o.w; u[1] = o.h; u[2] = o.mult;
    return u;
  }

  // dust_splat.wgsl —— 4 个 vec4
  function dustSplatUniforms(o) {
    var u = new Float32Array(16);
    u[0] = o.w; u[1] = o.h; u[2] = o.count; u[3] = o.t;
    u[4] = o.twinkle; u[5] = o.fadeFar; u[6] = o.lightBoost;
    u[7] = o.hasLight ? 1 : 0;
    u[8] = o.hasLight ? o.lightX : 0;
    u[9] = o.hasLight ? o.lightY : 0;
    u[10] = o.scale; u[11] = o.hasFar ? 1 : 0;
    return u;
  }

  // dust_apply.wgsl —— 2 个 vec4
  function dustApplyUniforms(o) {
    var u = new Float32Array(8);
    u[0] = o.w; u[1] = o.h; u[2] = 1 / o.scale; u[3] = o.dustBright;
    var c = o.color || [1, 1, 1];
    u[4] = c[0]; u[5] = c[1]; u[6] = c[2];
    return u;
  }

  // ── 分层视差（M5）──────────────────────────────────────────────
  //
  // ⚠️ 这三个量与 src/pixelart/parallax.py 必须**一字不差**：
  // 波形（1 - cos(τt)，t=0/t=1 逐位为 0）、幅度按**幅宽比例**（不是固定像素）、
  // 以及 uniform 的字段顺序。tools/uniform_parity_probe.py 会逐字节比对。

  //: 位移幅度相对画面宽度的比例（parallax.REL_AMPLITUDE）。
  var PARALLAX_REL_AMP = 0.06;

  /** 推拉波形：0 → 2 → 0。⚠️ cos(2π) 在 IEEE754 下**恰好**是 1.0，
   *  所以 t=1 时 wave 精确为 0（与 numpy 同）—— 用 sin 就会留下 -2.4e-16。 */
  function parallaxWave(t) {
    return 1 - Math.cos(TWO_PI * t);
  }

  /** 像素单位的位移基数 k = amp · 0.06 · gw · wave。 */
  function parallaxK(amp, gw, t) {
    return amp * PARALLAX_REL_AMP * gw * parallaxWave(t);
  }

  // parallax_splat.wgsl —— 4 个 vec4
  function parallaxSplatUniforms(o) {
    var u = new Float32Array(16);
    u[0] = o.w; u[1] = o.h; u[2] = o.k; u[3] = o.wave;
    u[4] = o.pivotX; u[5] = o.pivotY; u[6] = o.drift; u[7] = o.nEdges;
    u[8] = o.nearGain; u[9] = o.farGain; u[10] = o.scale;
    return u;
  }

  // parallax_apply.wgsl —— 4 个 vec4
  function parallaxApplyUniforms(o) {
    var u = new Float32Array(16);
    u[0] = o.w; u[1] = o.h; u[2] = 1 / o.scale; u[3] = o.k;
    u[4] = o.farOff;
    return u;
  }

  // tail.wgsl —— 2 个 vec4
  function tailUniforms(o) {
    var u = new Float32Array(8);
    u[0] = o.nPalette; u[1] = o.dither;
    u[2] = o.ditherAdaptive ? 1 : 0;
    u[3] = o.edgeGain;
    u[4] = o.edgeStrength; u[5] = o.w; u[6] = o.h;
    u[7] = o.continuous ? 1 : 0;          // 连续色模式（M5.10）：跳过吸附
    return u;
  }

  var API = {
    TWO_PI: TWO_PI,
    gridPhasor: gridPhasor,
    fogNoiseMean: fogNoiseMean,
    flickerGain: flickerGain,
    limitSaturation: limitSaturation,
    fogUniforms: fogUniforms,
    scatterUniforms: scatterUniforms,
    volumetricUniforms: volumetricUniforms,
    bloomBrightUniforms: bloomBrightUniforms,
    bloomBlurUniforms: bloomBlurUniforms,
    bloomCombineUniforms: bloomCombineUniforms,
    dustSplatUniforms: dustSplatUniforms,
    dustApplyUniforms: dustApplyUniforms,
    parallaxWave: parallaxWave,
    parallaxK: parallaxK,
    parallaxSplatUniforms: parallaxSplatUniforms,
    parallaxApplyUniforms: parallaxApplyUniforms,
    PARALLAX_REL_AMP: PARALLAX_REL_AMP,
    tailUniforms: tailUniforms
  };

  root.WgslUniforms = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
