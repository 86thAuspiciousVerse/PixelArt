// 分层视差 · pass 2：定点累加 → 归一化 + 破洞兜底，输出「搬完的 base + far」。
//
// ⚠️ 对应 parallax_warp 的收尾部分：
//
//     wsum  = Σw
//     alpha = clip(wsum · 2, 0, 1)
//     out   = (Σ(v·w) / wsum) · alpha + 原值 · (1 - alpha)
//
// 为什么要 alpha 混合而不是直接除以 wsum：
// 破洞（没有任何源覆盖）与"覆盖很薄"的格，除以一个很小的权重会把量化噪声放大。
// 推拉幅度只有百分之几幅宽，"没动够（保留原值）"远比"拉出一片噪点"不显眼。
//
// ── 输出布局（单缓冲区，避免一个 pass 两个输出）──
//
//   out[0 .. 3N)          = 搬完的 base（3 通道交错，与别的 pass 一致）
//   out[far_off .. +N)    = 搬完的 far
//   far_off = N_pad（**64 个 float 的整数倍**，见 webgpu.parallax_pack_layout）
//
// ⚠️ 为什么 far 要放在**对齐偏移**上：下游 fog / volumetric / dust 的 far
//    绑定要指向这个同一块缓冲的 far 区，而 WebGPU 要求
//    `minStorageBufferOffsetAlignment`（256 字节 = 64 float）整除。
//    取整到 64 float 之后，一次分配、一次回读、下游按偏移绑定 —— 不再需要
//    "warp 版 / 原版"两套绑定组。
//
// ⚠️ k = 0（t=0 / t=1 / amp=0）时走**逐位拷贝**分支：这时位移恒为零，
//    绝不能让它经过定点量化（round(v·S)/S ≠ v，会破坏"默认路径逐位不变"）。
//
// uniform:
//   U[0] = (W, H, inv_scale, k)
//   U[1] = (far_off, 0, 0, 0)
//   U[2] = (unused ×4)
//   U[3] = (unused ×4)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 4>;
@group(0) @binding(1) var<storage, read> acc: array<u32>;       // stride 5
@group(0) @binding(2) var<storage, read> base: array<f32>;      // 原图（3 通道交错）
@group(0) @binding(3) var<storage, read> far: array<f32>;       // 原深度
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // base 区 + far 区

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  if (gid.x >= W || gid.y >= H) { return; }

  let inv_scale = U[0].z;
  let k = U[0].w;
  let far_off = u32(U[1].x);

  let i: u32 = gid.y * W + gid.x;
  let b: u32 = i * 3u;

  // ── 无位移：逐位拷贝（不能过定点）──
  if (abs(k) < 1e-9) {
    out[b] = base[b];
    out[b + 1u] = base[b + 1u];
    out[b + 2u] = base[b + 2u];
    out[far_off + i] = far[i];
    return;
  }

  let ab: u32 = i * 5u;
  let wsum = f32(acc[ab + 3u]) * inv_scale;
  let safe = max(wsum, 1e-8);
  let alpha = clamp(wsum * 2.0, 0.0, 1.0);

  let r = f32(acc[ab + 0u]) * inv_scale / safe;
  let g = f32(acc[ab + 1u]) * inv_scale / safe;
  let bl = f32(acc[ab + 2u]) * inv_scale / safe;
  let df = f32(acc[ab + 4u]) * inv_scale / safe;

  out[b] = r * alpha + base[b] * (1.0 - alpha);
  out[b + 1u] = g * alpha + base[b + 1u] * (1.0 - alpha);
  out[b + 2u] = bl * alpha + base[b + 2u] * (1.0 - alpha);
  out[far_off + i] = df * alpha + far[i] * (1.0 - alpha);
}
