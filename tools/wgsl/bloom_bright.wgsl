// 辉光 · pass 1：高光提取。
//
// ⚠️ 对应 src/pixelart/compose.py::bright_pass。
//
// 单独成一个 pass 是因为：三个尺度都要模糊**同一张**高光图。
// 若把高光提取折进第一个尺度的模糊里，后面两个尺度就得自己再算一遍。
//
// ⚠️ knee 的两个分支都要写。硬阈值是 mask=t，软阈值是
//    clip(t²/(t+knee)/(1+knee), 0, 1) —— **两个不同公式**。
//    只写软的那支时 knee=0 会退化成 t²/(t+1e-6)，差得极远。
//
// uniform:
//   U[0] = (W, H, threshold, knee)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 1>;
@group(0) @binding(1) var<storage, read> rgb: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  if (gid.x >= W || gid.y >= H) { return; }

  let b: u32 = (gid.y * W + gid.x) * 3u;
  let c = vec3<f32>(rgb[b], rgb[b + 1u], rgb[b + 2u]);

  let threshold = U[0].z;
  let knee = U[0].w;
  let denom = max(1e-6, 1.0 - threshold);
  let t = clamp((dot(c, LUMA) - threshold) / denom, 0.0, 1.0);

  var mask = t;                                 // 硬阈值分支
  if (knee > 0.0) {                             // 软阈值分支
    mask = t * t / (t + knee + 1e-6);
    mask = clamp(mask / (1.0 + knee), 0.0, 1.0);
  }

  let out = c * mask;
  dst[b] = out.x;
  dst[b + 1u] = out.y;
  dst[b + 2u] = out.z;
}
