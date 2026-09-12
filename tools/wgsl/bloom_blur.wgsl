// 辉光 · pass 2：可分离高斯模糊（先横后纵，各一次 dispatch）。
//
// ⚠️ 与 src/pixelart/compose.py::blur_float 对应。
//
// ── 为什么核系数要从 CPU 上传，而不是在这里现算 ──
//
// 核系数形如 ``exp(-x²/2σ²)`` 再归一化。``exp`` 在 GPU 与 libm 上**末位不同**，
// 于是同一个 σ 会算出**略有差异的核** —— 经 67 个抽头累积后被放大，
// 最终表现为"GPU 的模糊和 CPU 的不是同一个模糊"。
//
// 与其逐位对齐两边的 exp，不如**只算一次**：CPU 侧
// ``compose.gaussian_kernel`` 算好、拼成一段 float32 传给这里。
// 这样"核"是一个共享常量，而不是两份各自近似的东西。
// （这正是 src/pixelart/webgpu.py 存在的理由 —— 见 docs/arch-02-webgpu.md。）
//
// ── 边界 ──
// 用**边缘延展**（clamp），与 ``blur_float`` 的 ``np.pad(mode="edge")`` 一致。
// 边界处理不一样的话，画面边缘一圈就对不上，验收台会把它报成"移植误差"。
//
// ── 累积模式 ──
// ``accum=1`` 时做 ``dst = blurred · weight + prev``，让三个尺度**叠起来**。
//
// ⚠️ 累积量必须走**独立的 prev 输入缓冲**，不能让着色器把 dst 当累积器
//    读回来再加（``acc = dst + ...; dst = acc``）。
//    那种写法在浏览器里能跑（同一个 buffer 既读又写同一个下标），
//    但**任何按"输出缓冲每次新建"的方式调用就会静默失效** ——
//    验收台就是这么调用的，于是"三个尺度加权求和"退化成"只留最后一个"，
//    实测让 bloom 差出 8.4e-2，还很容易被误当成移植错误。
//    → 把累积做成显式输入，语义就不再依赖"缓冲区是否被复用"。
//
// uniform:
//   U[0] = (W, H, direction(0=横 1=纵), accum)
//   U[1] = (kernel_offset, kernel_count, half, weight)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 2>;
// 全部尺度的核拼在一起（CPU 侧拼好，见 webgpu.bloom_kernels）
@group(0) @binding(1) var<storage, read> kbuf: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
// 上一个尺度的累积结果（第一个尺度传全零）
@group(0) @binding(3) var<storage, read> prev: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  if (gid.x >= W || gid.y >= H) { return; }

  let horiz = U[0].z < 0.5;
  let accum = U[0].w > 0.5;
  let k_off = u32(U[1].x);
  let k_cnt = u32(U[1].y);
  let half = i32(U[1].z);
  let weight = U[1].w;

  let b: u32 = (gid.y * W + gid.x) * 3u;
  var acc = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < k_cnt; i = i + 1u) {
    let k = kbuf[k_off + i];
    var sx = i32(gid.x);
    var sy = i32(gid.y);
    if (horiz) {
      sx = clamp(i32(gid.x) + i32(i) - half, 0, i32(W) - 1);
    } else {
      sy = clamp(i32(gid.y) + i32(i) - half, 0, i32(H) - 1);
    }
    let o = (u32(sy) * W + u32(sx)) * 3u;
    acc += vec3<f32>(src[o], src[o + 1u], src[o + 2u]) * k;
  }

  if (accum) {
    acc = vec3<f32>(prev[b], prev[b + 1u], prev[b + 2u]) + acc * weight;
  }
  dst[b] = acc.x;
  dst[b + 1u] = acc.y;
  dst[b + 2u] = acc.z;
}
