// 分层视差 · pass 1：把每个**源像素**按位移散射到目标像素（前向 splatting）。
//
// ⚠️ 对应 src/pixelart/parallax.py::parallax_warp。
//
// ── 为什么是 scatter 而不是 gather ──
//
// 后向采样（gather）不知道遮挡：前景移走后，它"拉"来的源像素可能是错的层的内容。
// 前向 splat 把每个像素按双线性权重摊到 4 个邻格 —— 遮挡天然正确，
// 没被覆盖的格（disocclusion 破洞）在 pass 2 里按覆盖率与原图混合兜底。
//
// 所以这里**每源像素一个线程**（线程数 = 像素数），用 atomicAdd 累加到目标格 ——
// 与尘埃同一个套路（那边是"每颗粒子一个线程"）。
//
// ── 为什么用 u32 定点而不是 float 原子加 ──
//
// 浮点加法不满足结合律，而 GPU 上多个线程的原子加顺序**由驱动决定** ——
// 每次运行的结果都可能不同，就谈不上"与 CPU 参考可复现"。
// 整数加法精确且顺序无关，所以每份贡献先缩放到整数：
//
//   贡献 = round(clip(v,0,1) · w · SCALE)     （w 是双线性权重，≤1）
//   最后 out = (Σ贡献) / (Σ权重·SCALE)
//
// SCALE = 2^20（8 个通道累加后仍 ≪ 2^32），单份量化误差 ≤ 4.8e-7，
// 比一个 uint8 色阶（3.9e-3）小 8000 倍。
//
// ⚠️ 溢出核算：每个目标像素收到的权重和 = Σw ≈ 1（推拉幅度 ≤10% 幅宽，
//    汇聚系数很小），乘 SCALE 后 ≈ 2^20；即使保守到 8 倍也只有 2^23。
//
// ── 层边界由 CPU 算好上传 ──
//
// 分层用的是**分位数**边界（等距切会在真实深度分布上切出空层），
// 而 WGSL 没有 quantile / 排序。所以边界数组由 CPU 算一次放进场景包，
// 这里只做"数一数有几个边界 ≤ far"这件能在着色器里做的事。
// 规则必须与 numpy 侧 layer_weights_from_edges 一字不差：
//   k = 边界中 ≤ far 的个数（E = 边界数，切成 E-1 段）
//   idx = clamp(k-1, 0, E-2) → u = idx/(E-2) → w = near_gain + (far_gain-near_gain)·u
//   ⚠️ 分母是 **E-2**（= 段数-1），让最远段恰好落到 far_gain。
//
// uniform:
//   U[0] = (W, H, k, wave)                 k = amp · 0.06 · W · wave（像素）
//   U[1] = (pivot_x_px, pivot_y_px, drift, n_edges)
//   U[2] = (near_gain, far_gain, SCALE, far_base_index)   —— 后者本 pass 不用
//   U[3] = (unused…)
//   edges = array<f32>（n_edges 个，升序去重）

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 4>;
@group(0) @binding(1) var<storage, read> edges: array<f32>;
@group(0) @binding(2) var<storage, read> base: array<f32>;      // 3 通道交错
@group(0) @binding(3) var<storage, read> far: array<f32>;
// 每像素 5 个累加器：[Σr, Σg, Σb, Σw, Σfar·w]
@group(0) @binding(4) var<storage, read_write> acc: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  if (gid.x >= W || gid.y >= H) { return; }

  let k = U[0].z;
  if (abs(k) < 1e-9) { return; }        // 波形为 0（t=0 / t=1）→ 无位移，pass 2 走拷贝

  let wave = U[0].w;
  let pvx = U[1].x;
  let pvy = U[1].y;
  let drift = U[1].z;
  let n_edges = u32(U[1].w);
  let near_gain = U[2].x;
  let far_gain = U[2].y;
  let scale = U[2].z;

  let i: u32 = gid.y * W + gid.x;
  let fv = clamp(far[i], 0.0, 1.0);

  // 层序号：数一数有几个边界 ≤ fv（与 numpy searchsorted(side="right") 同义）
  var cnt: u32 = 0u;
  for (var e: u32 = 0u; e < n_edges; e = e + 1u) {
    if (edges[e] <= fv) { cnt = cnt + 1u; }
  }
  // ⚠️⚠️ 归一化的分母是 **段数 - 1 = 边数 - 2**，不是"边数 - 1"。
  //    E 个边界切成 E-1 段，段号 0..E-2；要让**最远段恰好等于 far_gain**，
  //    归一化必须除以 (E-2)。
  //    我第一版写成 E-1：13 个边界时最远段只到 0.193 而不是 0.12
  //    —— 远层多动了 60%，端到端最大差 0.18（验收台抓出来的）。
  //    这就是"同一个概念在两处各写一次"的老问题：numpy 侧是对的
  //    （除以 edges.size-2），着色器少减了 1。
  var idx: i32 = i32(cnt) - 1;
  let nseg: i32 = max(i32(n_edges) - 2, 1);
  idx = clamp(idx, 0, nseg);
  let u01 = f32(idx) / f32(nseg);
  let w = near_gain + (far_gain - near_gain) * u01;

  // 前向位移：径向推拉（近层系数大）+ 随层的垂直漂移
  let gw_ = max(f32(W) - 1.0, 1.0);
  let gh_ = max(f32(H) - 1.0, 1.0);
  let ox = k * w * (f32(gid.x) - pvx) / gw_;
  let oy = k * w * ((f32(gid.y) - pvy) / gh_ + drift);

  let nx = f32(gid.x) + ox;
  let ny = f32(gid.y) + oy;
  let x0 = i32(floor(nx));
  let y0 = i32(floor(ny));
  let fx = nx - floor(nx);
  let fy = ny - floor(ny);

  let b: u32 = i * 3u;
  let fv_s = f32(fv);
  let r = clamp(base[b], 0.0, 1.0);
  let g = clamp(base[b + 1u], 0.0, 1.0);
  let bl = clamp(base[b + 2u], 0.0, 1.0);

  // 4 邻格（双线性）。越界的目标被钳到边缘格 —— 与 numpy 的 np.clip 一致，
  // 于是"内容出画"的那部分贡献会叠在边缘格上（分母同步增长，不会变亮）。
  for (var dy: i32 = 0; dy < 2; dy = dy + 1) {
    for (var dx: i32 = 0; dx < 2; dx = dx + 1) {
      let xx = clamp(x0 + dx, 0, i32(W) - 1);
      let yy = clamp(y0 + dy, 0, i32(H) - 1);
      let wx = select(1.0 - fx, fx, dx == 1);
      let wy = select(1.0 - fy, fy, dy == 1);
      let w4 = wx * wy;
      if (w4 <= 0.0) { continue; }

      let ti: u32 = u32(yy) * W + u32(xx);
      let tb: u32 = ti * 5u;
      let qw = u32(w4 * scale + 0.5);
      atomicAdd(&acc[tb + 0u], u32(r * w4 * scale + 0.5));
      atomicAdd(&acc[tb + 1u], u32(g * w4 * scale + 0.5));
      atomicAdd(&acc[tb + 2u], u32(bl * w4 * scale + 0.5));
      atomicAdd(&acc[tb + 3u], qw);
      atomicAdd(&acc[tb + 4u], u32(fv_s * w4 * scale + 0.5));
    }
  }
}
