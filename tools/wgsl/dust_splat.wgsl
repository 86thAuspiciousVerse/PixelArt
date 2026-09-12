// 尘埃粒子 · pass 1：把每颗粒子的贡献**散射**到它所在的像素上。
//
// ⚠️ 对应 src/pixelart/animate.py::dust_layer。
//
// ── 为什么这个 pass 和别的不一样 ──
//
// 前面所有 pass 都是 **gather**：一个线程读一片输入、写一个输出像素。
// 粒子是 **scatter**：一颗粒子写**一个**像素，而且多颗可能落在同一个像素上
// （默认 200 颗撒在 480×270 上，碰撞不少）。
// gather 模型做不了这件事 —— 所以这里开"每颗粒子一个线程"，
// 用 atomicAdd 把贡献累加到目标像素。
//
// ── 为什么用 u32 定点而不是 float 原子加 ──
//
// 目标是**与 numpy 逐位一致**。numpy 侧是 ``out[py,px] += a``，
// 顺序是粒子编号顺序。
//
// 浮点加法**不满足结合律** —— ``(a+b)+c ≠ a+(b+c)``。而 GPU 上多个线程
// 的原子加顺序是**不确定的**，所以 float 原子加的结果每次运行都可能不同，
// 根本谈不上"与 CPU 逐位一致"。
//
// 整数加法是**精确且与顺序无关**的，所以把它缩放到 u32 定点上累加：
// 每颗粒子贡献 ``round(clip(a,0,1) · SCALE)``，最后除以 SCALE。
// 量化误差每颗 ≤ 1/(2·SCALE)，取 SCALE = 2^20 时约 4.8e-7，
// 远小于一个 uint8 色阶（1/255 ≈ 3.9e-3）。
//
// ⚠️ 溢出边界：累加上界是 ``count · SCALE``，必须 < 2^32。
//    ``SCALE`` 由 ``webgpu.dust_fixed_point_scale`` 按 count 反推，
//    保证不溢出（见那里的说明）。
//
// ── 为什么粒子参数由 CPU 算好上传 ──
//
// 粒子的一切都只依赖 ``(seed, i)``，是**静态**的。
// 而 ``animate.hash01`` 是 splitmix64 —— **WGSL 没有 64 位整数**
// （WebGPU 规范里没有 i64/u64，只有需要扩展的 shader-int64，不能依赖）。
// 与其在 WGSL 里用两个 u32 手搓 64 位乘法和移位，不如在 CPU 算一遍上传：
//
//   · 没有 64 位运算 → 没有可移植性风险
//   · **hash 只有一份实现** → 不可能两边漂移
//   · 缓冲区很小：每颗 11 个 float，500 颗 = 22 KB，而且**每场景只算一次**
//
// uniform:
//   U[0] = (W, H, count, t)
//   U[1] = (twinkle, fade_far, light_boost, has_light)
//   U[2] = (light_x, light_y, scale, has_far)
//   U[3] = (drift_unused, —, —, —)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 4>;
// 每颗 11 个 float：x0 y0 ax ay fx fy phx phy tf tp base
@group(0) @binding(1) var<storage, read> par: array<f32>;
@group(0) @binding(2) var<storage, read> far: array<f32>;
@group(0) @binding(3) var<storage, read_write> bin: array<atomic<u32>>;

const TAU: f32 = 6.283185307179586;
const STRIDE: u32 = 11u;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  let count: u32 = u32(U[0].z);
  let t = U[0].w;
  if (gid.x >= count) { return; }

  let twinkle = U[1].x;
  let fade_far = U[1].y;
  let light_boost = U[1].z;
  let has_light = U[1].w > 0.5;
  let light_x = U[2].x;
  let light_y = U[2].y;
  let scale = U[2].z;
  let has_far = U[2].w > 0.5;

  let b = gid.x * STRIDE;
  let x0 = par[b];
  let y0 = par[b + 1u];
  let ax = par[b + 2u];
  let ay = par[b + 3u];
  let fx = par[b + 4u];
  let fy = par[b + 5u];
  let phx = par[b + 6u];
  let phy = par[b + 7u];
  let tf = par[b + 8u];
  let tp = par[b + 9u];
  let base = par[b + 10u];

  // 位置：Lissajous 摆动（有界、不折返），严格周期
  // ⚠️ 相位写成 fract(f · t) 而不是 f · t —— 与雾那边同一条理由：
  //    数学等价（f 是整数，相差整数倍 TAU 对 sin 无影响），但数值上前者精度更好，
  //    且 t=1 时 fract(整数) **精确为 0** → frame(1) 与 frame(0) 逐位相同。
  //    （dust_layer 是时序铁律 3 的直接依赖项：粒子的循环闭合就靠这个。）
  let nx = fract(x0 + ax * sin(TAU * fract(fx * t) + phx));
  let ny = fract(y0 + ay * sin(TAU * fract(fy * t) + phy));

  // ⚠️ numpy 是 `int(nx * w) % w` —— int() 向零截断；nx 已在 [0,1) 内，
  //    所以截断 = 向下取整。u32() 在 WGSL 里也是向零截断。
  let px = u32(nx * f32(W)) % W;
  let py = u32(ny * f32(H)) % H;
  let idx = py * W + px;

  // 闪烁：整数频率正弦，严格周期
  let tw = 1.0 - twinkle * 0.5 * (1.0 - sin(TAU * fract(tf * t) + tp));
  var a = base * tw;

  if (has_far) {
    a = a * (1.0 - fade_far * clamp(far[idx], 0.0, 1.0));
  }
  if (has_light) {
    // 靠近光源的尘埃更亮（仿佛被光柱照亮），按归一化距离衰减
    let dx = f32(px) / max(f32(W) - 1.0, 1.0) - light_x;
    let dy = f32(py) / max(f32(H) - 1.0, 1.0) - light_y;
    a = a * (1.0 + (light_boost - 1.0) / (1.0 + 6.0 * sqrt(dx * dx + dy * dy)));
  }

  let q = u32(clamp(a, 0.0, 1.0) * scale + 0.5);
  atomicAdd(&bin[idx], q);
}
