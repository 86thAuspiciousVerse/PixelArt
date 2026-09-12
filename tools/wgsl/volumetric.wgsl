// 体积光（深度感知的光散射）—— 加性图层叠到雾后的画面上。
//
// ⚠️ 对应 src/pixelart/compose.py::volumetric_light。
//
// 和"屏幕空间光轴"的区别（也就是"光源周围一圈白"和"体积雾"的差别）：
//
//   1. **遮挡**：沿光线采样时，用深度判断采样点是否比光源近很多 ——
//      近很多的物体挡在前面，光应该被挡住而不是穿过去。没有这一步光会穿墙。
//   2. **按场景距离衰减**：用 |z_pixel − z_light|（场景里的距离），
//      不是屏幕上的像素距离。
//   3. **散射带颜色**：体积光照亮的是空气，所以叠上去的应该是光源的颜色。
//      但必须**低饱和** —— 见 scatter.wgsl 里那段说明（ref10 的变绿事故）。
//
// 本 pass 直接输出 ``clip(fogged + vol, 0, 1)``，与 ``compose_frame`` 里
// ``lit = np.clip(fogged + vol, 0, 1)`` 一致 —— 少一次往返。
//
// uniform:
//   U[0] = (samples, span, decay, strength)
//   U[1] = (occlude_gain, falloff_gain, screen_falloff, threshold)
//   U[2] = (knee, gain, W, H)
//   U[3] = (light_x, light_y, apex_x, apex_y)        ← 锥顶点（M5.4，打包侧解析：<0→=光源）
//   U[4] = (cone_angle, cone_dir_rad, cone_reach, cone_gain)  ← 光锥（M5.1）
//   U[5] = (shaft, fog_tint, —, —)                    ← 光柱强度（M5.2）+ 散射色上色强度（M5.3，⚠️ .y 槽）
//
// ⚠️ ``gain`` 是 flicker_gain(t) 的结果，在 CPU 上算好传进来（每帧一个标量）。
//    那是四个整数频率正弦的叠加，放进着色器只会让每个像素重复算同样的常量。

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 6>;
// 雾后的画面 (H*W*3)
@group(0) @binding(1) var<storage, read> fogged: array<f32>;
// 深度 (H*W)
@group(0) @binding(2) var<storage, read> far: array<f32>;
// 散射色（3 个 float），由 scatter.wgsl 写入
@group(0) @binding(3) var<storage, read> air_in: array<f32, 3>;
// 输出 (H*W*3)
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

/// 高光提取 + 投影到亮度。
///
/// ⚠️⚠️ 这里必须返回**标量**，不能返回 vec3。踩过的坑：
///
///   numpy 版的累积是 ``bright_lum[off] * (illum*occl)``，而
///   ``bright_lum = bright @ LUMA`` 是个**标量场**
///   （= ``mask * (rgb·LUMA)``，按亮度加权）。
///
///   我第一版把 `rgb * mask` 整个 vec3 拿去做累加 —— 那等于**按通道**加权。
///   数学上完全不是一回事：高光是彩色的区域会系统性偏掉。
///   实测最大差 2.3e-3、5% 的像素差 >1e-3，而且采样下标是**完全一致**的
///   （0/172032），所以问题不在几何，就在这个加权方式上。
///
/// 这类"看起来对、量起来不对"的错误，肉眼对比截图是发现不了的。
fn bright_lum_at(base: u32, denom: f32, threshold: f32, knee: f32) -> f32 {
  let r = vec3<f32>(fogged[base], fogged[base + 1u], fogged[base + 2u]);
  let l = dot(r, LUMA);
  let t = clamp((l - threshold) / denom, 0.0, 1.0);
  // ⚠️ 必须保留 numpy 的 knee 分支。软阈值（knee>0）和硬阈值是**两个不同公式**：
  //     硬：mask = t
  //     软：mask = clip(t² / (t + knee) / (1 + knee), 0, 1)
  //    只写软的那支时，knee=0 会退化成 t²/(t+1e-6) —— 与 t 差得很远。
  //    实测（knee=0）：最大差 1.7e-2、89% 的像素越界。
  //    当前调用方都传 knee>0，所以这个 bug 是**潜伏**的 ——
  //    数值比对是唯一能提前发现它的手段（看图完全看不出来）。
  var m = t;
  if (knee > 0.0) {
    m = t * t / (t + knee + 1e-6);
    m = clamp(m / (1.0 + knee), 0.0, 1.0);
  }
  return m * l;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[2].z);
  let H: u32 = u32(U[2].w);
  if (gid.x >= W || gid.y >= H) { return; }

  let i: u32 = gid.y * W + gid.x;
  let b: u32 = i * 3u;

  let samples_f = U[0].x;
  let span = U[0].y;
  let decay = U[0].z;
  let strength = U[0].w;
  let occlude_gain = U[1].x;
  let falloff_gain = U[1].y;
  let screen_falloff = U[1].z;
  let threshold = U[1].w;
  let knee = U[2].x;
  let gain = U[2].y;
  let lx = i32(U[3].x);
  let ly = i32(U[3].y);

  let Wf = f32(W);
  let Hf = f32(H);
  let lxf = f32(lx);
  let lyf = f32(ly);

  let denom = max(1e-6, 1.0 - threshold);

  // 光源所在处的深度
  let z_light = far[u32(ly) * W + u32(lx)];

  // ── 沿光线累积 ──
  // 采样点的**像素下标**只跟几何有关（分辨率 / 光源位置 / 采样数 / 跨度），
  // 与画面内容无关 —— Python 版把这层缓存起来跨帧复用；
  // 在 GPU 上每次现算只要二十几次整数运算，不值得缓存。
  let b_pix: u32 = gid.y * W + gid.x;      // 本像素下标（阴影量要用自己的深度）
  let xx = f32(gid.x);
  let yy = f32(gid.y);
  let dx = (lxf - xx) / Wf;
  let dy = (lyf - yy) / Hf;

  let samples = i32(samples_f);
  // ⚠️ acc 是**标量**（按亮度加权），不是 vec3 —— 见 bright_lum_at 的说明。
  var acc = 0.0;
  // 屏幕空间阴影量：沿"像素→光源"的射线，若某采样点比**本像素**更近，
  // 就有东西挡在中间 → 这一格在阴影里（光柱被切断）。
  // ⚠️ 与 compose.py 的 shade 同一行；bias 0.02 也必须一致。
  var shade = 0.0;
  let z_pix = far[b_pix];
  var illum = 1.0;
  for (var s_i = 1; s_i <= samples; s_i = s_i + 1) {
    let s = (f32(s_i) / samples_f) * span;
    // ⚠️ 运算顺序必须与 numpy 一致：(dx*w)*s，不是 dx*(w*s)。
    //    两者的舍入不同，可能让采样点落到相邻像素上 —— 那就是逐像素的分叉。
    let sx = clamp(i32(xx + (dx * Wf) * s), 0, i32(W) - 1);
    let sy = clamp(i32(yy + (dy * Hf) * s), 0, i32(H) - 1);
    let o = u32(sy) * W + u32(sx);

    // 遮挡：采样点比光源近很多 → 它在光源前面，挡住光
    let zq = far[o];
    let occl = exp(-occlude_gain * max(z_light - zq, 0.0));

    acc += bright_lum_at(o * 3u, denom, threshold, knee) * (illum * occl);
    shade = max(shade, z_pix - zq - 0.02);
    illum = illum * decay;
  }
  acc = acc / max(1.0, samples_f);

  // ── 按**场景深度差**衰减，不是屏幕像素距离 ──
  let dist = abs(far[i] - z_light);
  var out_v = acc * (1.0 / (1.0 + falloff_gain * dist * 4.0));

  // ── 屏幕空间衰减：让光"属于那个位置"，而不是均匀铺满全图 ──
  //
  // 只按深度差衰减时，与光源深度相近的像素会均匀吃到带色的光 ——
  // 观感就是"整幅图被往某个色相上拉，像蒙了一层滤镜"（用户报障）。
  // 实测加上它以后近处（<0.25）占比从 53% 提到 69%、全图注入量减半。
  if (screen_falloff > 0.0) {
    let dsx = xx / Wf - lxf / Wf;
    let dsy = yy / Hf - lyf / Hf;
    let d_screen = sqrt(dsx * dsx + dsy * dsy);
    out_v = out_v / (1.0 + pow(d_screen * 4.0, screen_falloff));
  }

  // ── 光锥（M5.1）：把"径向均匀的光晕"塑成有直边的形 ──
  //
  // ⚠️⚠️ 这一段必须与 src/pixelart/compose.py::cone_weight **一字不差** ——
  //    两边各写一次，迟早分叉（这个项目已经踩过 5 次）。
  //    运算顺序照抄 CPU：等距化 → 单位化 → cos → smoothstep(t²(3−2t)) → 长度衰减。
  //
  // ⚠️ cone_angle <= 0 时**连乘法都不做** —— 乘 1.0 也会在最后一位上分叉，
  //    而"默认行为逐位不变"是需要保证的。
  let cone_angle = U[4].x;
  if (cone_angle > 0.0) {
    let cone_dir = U[4].y;
    let cone_reach = U[4].z;
    // ⭐ 锥顶点（M5.4）：U[3].zw（打包侧已解析 —— <0 时=光源，数值同旧路径）。
    //    只有锥形与光柱的锚点换掉；遮挡/衰减/散射色仍以光源（U[3].xy）为准。
    let axf = U[3].z;
    let ayf = U[3].w;
    let vx = (xx - axf) / max(Hf - 1.0, 1.0) * (Wf / max(Hf, 1.0));
    let vy = (yy - ayf) / max(Hf - 1.0, 1.0);
    let vlen = sqrt(vx * vx + vy * vy);
    let inv = 1.0 / max(vlen, 1e-6);
    let dotv = (vx * inv) * cos(cone_dir) + (vy * inv) * sin(cone_dir);
    // ⚠️ 光源自身那个像素 vlen≈0、方向未定义 → 显式视为在锥内
    //    （否则光源自己是暗的）。与 cone_weight 里那一行同义。
    let cosang = select(1.0, dotv, vlen > 1e-6);
    let outer = cos(cone_angle);
    let inner = cos(cone_angle * 0.5);
    let tt = clamp((cosang - outer) / max(inner - outer, 1e-6), 0.0, 1.0);
    var cw = tt * tt * (3.0 - 2.0 * tt);
    if (cone_reach < 1.0) {
      cw = cw / (1.0 + (1.0 - cone_reach) * 3.0 * vlen);
    }
    // ⚠️ 锥内增益：单纯"锥外减掉"几乎看不出（实测开/关只差 1.8/255）——
    //    光柱要的是**锥内空气比锥外亮**，所以要乘性增益。与 CPU 同义。
    let cone_gain = U[4].w;
    if (cone_gain != 1.0) {
      cw = cw * cone_gain;
    }
    out_v = out_v * cw;

    // ── ⭐ 光柱：独立加性层（M5.2）──
    // ⚠️ 不能只乘权重：实测体积光整层只有 1~3/255（falloff + screen_falloff
    //    把能量压掉两个数量级）。做法是拿**沿射线累积的亮度**（acc 已归一化，
    //    带着遮挡结构）直接乘锥形与强度，**不经过**那两道压制。
    //    与 compose.py 里同一条公式；shaft=0 时连加法都不做（逐位不变）。
    // ★ 第二次修正：不能乘 acc（那是"能否看见点光源"，锥内多数像素的射线
    //   打不到那个亮点 —— 实测锥内平均只有 2/255）。光柱表达的是
    //   "**锥内的空气被照亮**"：沿柱基本均匀、随距离衰减、被屏幕阴影切断。
    let shaft = U[5].x;
    if (shaft > 0.0) {
      let ax_dx = (xx - axf) / max(Hf - 1.0, 1.0);
      let ax_dy = (yy - ayf) / max(Hf - 1.0, 1.0);
      let axial = 1.0 / (1.0 + 1.6 * sqrt(ax_dx * ax_dx + ax_dy * ax_dy));
      let beam = cw * axial * exp(-4.0 * max(shade, 0.0));
      out_v = out_v + beam * shaft;
    }
  }

  let air0 = vec3<f32>(air_in[0], air_in[1], air_in[2]);
  // ── 散射色的上色强度（M5.3）：与 fog 的 fog_tint 同一个旋钮 ──
  // ⚠️ 槽位必须与两侧 packer 对齐：webgpu.py `u[5] = [shaft, fog_tint, 0, 0]`、
  //    m3_wgsl_uniforms.js `u[21] = fogTint` → 都是 U[5].**y**。
  //    第一版写成 U[5].z（恒 0）→ GPU 永远全去色且 uniform 探针全绿
  //    （它只比打包两侧，不比 shader 槽位）。wgsl_lab 的 fog_tint 扫描管这个。
  let fog_tint_v = U[5].y;
  var air = air0;
  if (fog_tint_v < 0.999) {
    let lum = dot(air0, vec3<f32>(0.2126, 0.7152, 0.0722));
    air = vec3<f32>(lum) + (air0 - vec3<f32>(lum)) * fog_tint_v;
  }
  // ⚠️ 标量先 clip、再广播成 vec3、最后乘散射色 —— 与 numpy 的
  //    `np.clip(out*strength*gain, 0, 1)[..., None] * air` 一致。
  let vol = vec3<f32>(clamp(out_v * strength * gain, 0.0, 1.0)) * air;

  // ⚠️ 出口处就是 compose_frame 的那一行：lit = clip(fogged + vol, 0, 1)
  let rgb = vec3<f32>(fogged[b], fogged[b + 1u], fogged[b + 2u]);
  let lit = clamp(rgb + vol, vec3<f32>(0.0), vec3<f32>(1.0));
  dst[b] = lit.x;
  dst[b + 1u] = lit.y;
  dst[b + 2u] = lit.z;
}
