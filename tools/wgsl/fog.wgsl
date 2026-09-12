// depth_fog —— 指数空气透视，WGSL 计算着色器版本。
//
// ⚠️ 这份代码必须与 src/pixelart/compose.py::depth_fog **逐行对应**。
//    任何"顺手优化"都会让 GPU 与 CPU 的输出分叉，而验收台会立刻报出来。
//
// 与 numpy 版的对应关系：
//
//   z  = clip(far, 0, 1)
//   d  = max(density, 1e-6)
//   if drift > 0:  d *= clip(1 + drift*(0.6*n + 0.4*n.mean()), 0.05, inf)
//   tf = 1 - exp(-d * z**power)
//   tf = clip((tf - floor) / (ceiling - floor), 0, 1)
//   out = clip(rgb*(1-tf) + fog*tf, 0, 1)
//
// 两处必须走 uniform 而不能在这里现算：
//   · n.mean()  —— 全局归约，CPU 侧算好（见 pixelart.webgpu.fog_uniforms）
//   · 噪声谐波系数 —— 只依赖 seed，不必让每个像素重算哈希
//
// uniform 布局见 pixelart.webgpu.fog_uniforms 的文档字符串。

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 6>;

// 基础色：(H*W*3) 个 f32，行优先
@group(0) @binding(1) var<storage, read> src: array<f32>;
// 深度：(H*W) 个 f32，0 = 近，1 = 远
@group(0) @binding(2) var<storage, read> far: array<f32>;
// 输出：(H*W*3) 个 f32
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

const TAU: f32 = 6.283185307179586;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[2].z);
  let H: u32 = u32(U[2].w);
  if (gid.x >= W || gid.y >= H) { return; }

  let idx: u32 = gid.y * W + gid.x;

  let fog_color = U[0].xyz;
  // ── 雾的上色强度（M5.3）：1 = 原雾色；<1 把雾色拉向中性灰（保色相）。
  // ⚠️ 必须与 compose.cone_weight 同级的"两侧一字不差"—— numpy 在
  //    depth_fog 里有同一行。fog_tint >= 0.999 时跳过（逐位不变）。
  let fog_tint = U[0].w;
  var fog_col = fog_color;
  if (fog_tint < 0.999) {
    let lum = dot(fog_color, vec3<f32>(0.2126, 0.7152, 0.0722));
    fog_col = vec3<f32>(lum) + (fog_color - vec3<f32>(lum)) * fog_tint;
  }
  let density0  = U[1].x;
  let power     = U[1].y;
  let floor_    = U[1].z;
  let inv_span  = U[1].w;
  let drift     = U[2].x;
  let noise_mean = U[2].y;
  let t         = U[5].w;

  let z = clamp(far[idx], 0.0, 1.0);

  var d = max(density0, 1e-6);

  if (drift > 0.0) {
    // 严格循环的空间噪声场（谱合成）：每层的时间频率都是整数，
    // 所以 t 加 1 后相位精确回到原处 —— 循环处不可能有接缝。
    // ⚠️ 这三个坐标要和 numpy 的 np.meshgrid(xx, yy, indexing="xy") 一致：
    //    gx 沿列（gid.x），gy 沿行（gid.y），且都取 **像素中心**（+0.5）。
    let gx = (f32(gid.x) + 0.5) / f32(W);
    let gy = (f32(gid.y) + 0.5) / f32(H);

    // ⚠️⚠️ 时间项必须写成 fract(ft * t)，不能写 ft * t。
    //
    //   数学上两者等价（ft 是整数，相差的整数倍 TAU 对 sin 无影响），
    //   但数值上差别很大：
    //
    //   · 写 ft*t 时，t=1 会把相位累加到 TAU*ft + 空间项（实测约 314 弧度）。
    //     f32 在大角度上精度掉到约 1.5e-6，于是 **t=0 与 t=1 不再逐位相同**，
    //     循环接缝处会有约 1e-7 的残差 —— 时序铁律 3 在 GPU 侧就破了。
    //     （numpy 那边没事，因为它在 float64 里算 sin，误差舍入掉了。）
    //   · 写 fract(ft*t) 时相位始终被约束在 [0, TAU) 附近，精度更好；
    //     且 t=1 时 fract(ft*1.0) 对整数 ft **精确等于 0**，
    //     于是 t=1 与 t=0 的相位逐位相同 → 循环严格闭合。
    var n = 0.0;
    n += U[5].x * sin(TAU * (U[3].x * gx + U[3].y * gy + fract(U[3].z * t)) + U[3].w);
    n += U[5].y * sin(TAU * (U[4].x * gx + U[4].y * gy + fract(U[4].z * t)) + U[4].w);
    n = n * U[5].z;                       // 除以 Σamp

    d = d * clamp(1.0 + drift * (0.6 * n + 0.4 * noise_mean), 0.05, 1e9);
  }

  let tf0 = 1.0 - exp(-d * pow(z, max(power, 1e-6)));
  let tf  = clamp((tf0 - floor_) * inv_span, 0.0, 1.0);

  let b = idx * 3u;
  let rgb = vec3<f32>(src[b], src[b + 1u], src[b + 2u]);
  let outc = clamp(rgb * (1.0 - tf) + fog_col * tf, vec3<f32>(0.0), vec3<f32>(1.0));

  dst[b]        = outc.x;
  dst[b + 1u]   = outc.y;
  dst[b + 2u]   = outc.z;
}
