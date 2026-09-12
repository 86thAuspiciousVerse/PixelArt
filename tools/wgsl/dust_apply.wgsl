// 尘埃粒子 · pass 2：定点缓冲 → 加性图层，叠到合成画面上。
//
// ⚠️ 对应 ``compose_frame`` 里的这一行：
//    ``lit = np.clip(lit + dust * a.dust_bright, 0.0, 1.0)``
//    其中 ``dust = np.clip(out, 0, 1)[..., None] * col``。
//
// 分两步（splat 累加 → apply 叠加）而不是合成一个 pass，是因为
// scatter 与 gather 没法在同一个 dispatch 里完成：
// 粒子个数与像素个数没有对应关系。
//
// uniform:
//   U[0] = (W, H, inv_scale, dust_bright)
//   U[1] = (col_r, col_g, col_b, —)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 2>;
@group(0) @binding(1) var<storage, read> bin: array<u32>;
@group(0) @binding(2) var<storage, read> lit: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  if (gid.x >= W || gid.y >= H) { return; }

  let i: u32 = gid.y * W + gid.x;
  let b: u32 = i * 3u;

  // ⚠️ 先除回 float、再 clip 到 1，最后乘颜色 —— 顺序与 numpy 一致：
  //    ``np.clip(out, 0, 1)[..., None] * col``
  let amt = min(f32(bin[i]) * U[0].z, 1.0);
  let col = vec3<f32>(U[1].x, U[1].y, U[1].z);
  let dust = vec3<f32>(amt) * col;

  let c = vec3<f32>(lit[b], lit[b + 1u], lit[b + 2u]);
  let out = clamp(c + dust * U[0].w, vec3<f32>(0.0), vec3<f32>(1.0));

  dst[b] = out.x;
  dst[b + 1u] = out.y;
  dst[b + 2u] = out.z;
}
