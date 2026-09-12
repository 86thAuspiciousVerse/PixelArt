// 像素尾巴 —— 感知空间 → 抖动 → 色板吸附 → 边缘压暗 → 再吸附。
//
// ⚠️ 必须与 src/pixelart/pixelate.py::pixelate **逐行对应**。
//    这个阶段是**硬要求逐位一致**的：色板吸附是纯比较，抖动是纯查表，
//    没有任何超越函数 —— 所以任何差异都是逻辑错误，不是精度问题。
//
// 三条本项目用事故换来的约束，在这里落地：
//   1. 抖动相位只依赖**方块坐标**（y%8, x%8），表达式里绝不能出现 t。
//   2. 量化与抖动永远是最后一步（所以这个 pass 画的是最终像素）。
//   3. 边缘压暗的梯度必须从**抖动之前的内容**上求（edge_ref），
//      不能从抖动结果上求 —— 否则整幅会被判成"处处是边缘"、统一压暗。
//
// ── 一处刻意的实现差异（更好的那个） ──
//
// numpy 的 snap_exact 用**展开式** |p|² - 2x·p + |x|² 并升到 float64 算，
// 因为展开式在 x≈p 时会发生灾难性抵消（两个接近 1 的量相减得接近 0）。
//
// WGSL 只有 f32，用展开式会更糟。所以这里改用**直接形式** Σ(pᵢ-xᵢ)²：
// 数学上完全等价，但没有任何抵消，f32 下反而比展开式的 f32 更准。
//
// uniform 布局（array<vec4<f32>>）：
//   U[0] = (n_palette, dither, dither_adaptive, edge_gain)
//   U[1] = (edge_strength, W, H, continuous)
//          continuous = 1 → 跳过色板吸附（M5.10 连续色高保真模式：抖动/边缘照常，
//          只是不做 snap —— 与 numpy pixelate 的 quantize_mode="continuous" 对应）

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 2>;

// 合成后的连续色（已含雾/体积光/粒子/辉光）：(H*W*3)
@group(0) @binding(1) var<storage, read> src: array<f32>;
// 边缘参考图（**不含粒子**的合成图）：(H*W*3)
@group(0) @binding(2) var<storage, read> edge_ref: array<f32>;
// 色板（**感知空间**）：(N*3)
@group(0) @binding(3) var<storage, read> pal: array<f32>;
// 输出：打包的 RGBA8，一个像素一个 u32
@group(0) @binding(4) var<storage, read_write> dst: array<u32>;

// Bayer 8×8（行优先，值是 0..63）。与 pixelart.dither.BAYER8 * 64 完全一致。
const BAYER: array<i32, 64> = array<i32, 64>(
   0, 32,  8, 40,  2, 34, 10, 42,
  48, 16, 56, 24, 50, 18, 58, 26,
  12, 44,  4, 36, 14, 46,  6, 38,
  60, 28, 52, 20, 62, 30, 54, 22,
   3, 35, 11, 43,  1, 33,  9, 41,
  51, 19, 59, 27, 49, 17, 57, 25,
  15, 47,  7, 39, 13, 45,  5, 37,
  63, 31, 55, 23, 61, 29, 53, 21,
);

//: 亮度权重，与 compose.LUMA / palette.LUMA 一致
const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

/// 感知空间最近色吸附。
///
/// ⚠️ 平局取**更小的下标**（严格 `<`，不是 `<=`）—— numpy 的 argmin 就是取首次出现，
///    用 `<=` 会在并列距离处选中最后一项，输出就与 CPU 分叉了。
fn snap(x: vec3<f32>, n: u32) -> vec3<f32> {
  var best_i: u32 = 0u;
  var best_d: f32 = 1e30;
  for (var k: u32 = 0u; k < n; k = k + 1u) {
    let p = vec3<f32>(pal[k * 3u], pal[k * 3u + 1u], pal[k * 3u + 2u]);
    let d = p - x;
    let dd = d.x * d.x + d.y * d.y + d.z * d.z;
    if (dd < best_d) { best_d = dd; best_i = k; }
  }
  return vec3<f32>(pal[best_i * 3u], pal[best_i * 3u + 1u], pal[best_i * 3u + 2u]);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W: u32 = u32(U[1].y);
  let H: u32 = u32(U[1].z);
  if (gid.x >= W || gid.y >= H) { return; }

  let x: u32 = gid.x;
  let y: u32 = gid.y;
  let i: u32 = y * W + x;
  let b: u32 = i * 3u;

  let n_pal: u32 = u32(U[0].x);
  let dither: f32 = U[0].y;
  let adaptive: f32 = U[0].z;
  let edge_gain: f32 = U[0].w;
  let edge_strength: f32 = U[1].x;
  let continuous: f32 = U[1].w;

  // ── 感知空间（x^(1/2)）──
  // 色板存的就是感知空间，所以吸附前必须先过来，否则整个色彩空间就是错的。
  let lin = clamp(vec3<f32>(src[b], src[b + 1u], src[b + 2u]),
                  vec3<f32>(0.0), vec3<f32>(1.0));
  let q = sqrt(lin);

  // ── 抖动 ──
  // ⚠️ 相位只由**方块整数坐标**决定：索引是 (y%8)*8 + (x%8)。
  //    表达式里不能出现 t，也不能出现任何"帧"相关量 —— 那会整屏闪烁。
  var src_rgb = q;
  if (dither > 0.0) {
    let tile = (f32(BAYER[(y % 8u) * 8u + (x % 8u)]) / 64.0) - 0.5;
    var amp = 1.0;
    if (adaptive > 0.5) {
      // 中间调最强、纯黑纯白关掉（避免暗部出脏噪点）
      let luma = dot(q, LUMA);
      amp = pow(clamp(1.0 - abs(2.0 * luma - 1.0), 0.0, 1.0), 0.8);
    }
    src_rgb = clamp(q + vec3<f32>(tile * dither * amp), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  // 连续色模式：跳过吸附（抖动已在上面注入；色板循环整个跳过）
  var out = src_rgb;
  if (continuous < 0.5) {
    out = snap(src_rgb, n_pal);
  }

  // ── 边缘压暗（模拟手绘描边）──
  // ⚠️ 梯度从 edge_ref 上求，**不是**从 out(src_rgb) 上求。
  //    抖动本身是高频信号，在抖动结果上求梯度会把整幅判成"处处是边缘"，
  //    统一乘 (1-strength) → 实测整体压暗约 30%，症状"像素化之后又灰又糊"。
  // ⚠️ edge_ref 还必须**不含粒子**：粒子是大气不是内容，
  //    否则每颗粒子会被当成强边界、周围压成黑点（"黑点在移动"事故）。
  if (edge_strength > 0.0) {
    let pe = sqrt(clamp(vec3<f32>(edge_ref[b], edge_ref[b + 1u], edge_ref[b + 2u]),
                       vec3<f32>(0.0), vec3<f32>(1.0)));
    let lum = dot(pe, LUMA);

    var gx = 0.0;
    if (x > 0u) {
      let bl = (y * W + (x - 1u)) * 3u;
      let pl = sqrt(clamp(vec3<f32>(edge_ref[bl], edge_ref[bl + 1u], edge_ref[bl + 2u]),
                         vec3<f32>(0.0), vec3<f32>(1.0)));
      gx = abs(lum - dot(pl, LUMA));
    }
    var gy = 0.0;
    if (y > 0u) {
      let bt = ((y - 1u) * W + x) * 3u;
      let pt = sqrt(clamp(vec3<f32>(edge_ref[bt], edge_ref[bt + 1u], edge_ref[bt + 2u]),
                         vec3<f32>(0.0), vec3<f32>(1.0)));
      gy = abs(lum - dot(pt, LUMA));
    }
    // np.diff(..., prepend=...) 让第 0 列/行取 0 —— 上面两个 if 就是这件事
    let g = clamp((gx + gy) * edge_gain, 0.0, 1.0);
    out = out * (1.0 - edge_strength * g);
    if (continuous < 0.5) {
      out = snap(out, n_pal);
    }
  }

  // ── 回线性空间并量化到 8 位 ──
  let lin_out = clamp(out * out, vec3<f32>(0.0), vec3<f32>(1.0));

  // ⚠️ numpy 的 `.astype(np.uint8)` 是**截断**（truncation），不是四舍五入。
  //    即 `clip(x*255, 0, 255).astype(uint8)` —— 所以这里也必须截断。
  //    用 +0.5 再取整会让**每一个**非整数像素都差 1 个色阶，
  //    在暗部（值本来就小）尤其明显。
  let t8 = vec3<u32>(clamp(lin_out * 255.0, vec3<f32>(0.0), vec3<f32>(255.0)));
  dst[i] = t8.x | (t8.y << 8u) | (t8.z << 16u) | (255u << 24u);
}
