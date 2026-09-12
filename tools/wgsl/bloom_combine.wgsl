// 辉光 · pass 3：合并。
//
// ⚠️ 与 src/pixelart/compose.py::bloom_float 的最后一步对应：
//    ``out = clip(rgb + acc/wsum * strength, 0, 1)``
//
// ``acc`` 里已经是 ``Σ w_i · blurred_i``（由 blur pass 的累积模式叠好），
// 所以这里只需要乘以 ``strength / wsum`` 再相加。
// 把 ``strength / wsum`` 在 CPU 算成一个标量传进来，省掉每像素一次除法。
//
// uniform:
//   U[0] = (W, H, mult, —)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 1>;
@group(0) @binding(1) var<storage, read> rgb: array<f32>;
@group(0) @binding(2) var<storage, read> acc: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  if (gid.x >= W || gid.y >= H) { return; }

  let b: u32 = (gid.y * W + gid.x) * 3u;
  let mult = U[0].z;

  let c = vec3<f32>(rgb[b], rgb[b + 1u], rgb[b + 2u]);
  let a = vec3<f32>(acc[b], acc[b + 1u], acc[b + 2u]);
  let out = clamp(c + a * mult, vec3<f32>(0.0), vec3<f32>(1.0));

  dst[b] = out.x;
  dst[b + 1u] = out.y;
  dst[b + 2u] = out.z;
}
