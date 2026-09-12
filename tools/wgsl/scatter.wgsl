// 散射色采样 —— 单线程计算"光源附近窗口的中位数 + 饱和度上限"。
//
// ⚠️ 对应 src/pixelart/compose.py::auto_scatter_color。
//
// 为什么要单独一个 pass：体积光需要的是一个**标量级的散射色**，
// 而它要在"雾后"的图上取窗口统计量。逐像素各算一遍是浪费，
// 在 CPU 上算又需要把 GPU 的雾后结果读回来（同步点，会毁掉整条流水线）。
// 所以开一个 1 线程的微型 dispatch，结果写进 buffer 给下一个 pass 读 ——
// **全程留在 GPU 上，没有任何 CPU 往返**。
//
// 两处细节都是踩坑得来的，别"顺手简化"：
//
//   1. 取**中位数**不是均值。均值会被单个炸白的高光点带跑。
//      ⚠️ 也不能改成"只取较亮的那半像素再取中位数"：实测那样会丢掉约 25% 的暖色
//      （ref02 的暖度 R−B 从 +0.23 掉到 +0.11），因为暖光场景里最亮的
//      那批像素往往正是**炸白的高光**，色相已经丢了。
//   2. **必须压饱和度**。这不是锦上添花 —— 实测 ref10 的光源中心落在草丛里，
//      窗口 7×7 全是绿色，无论取什么统计量都是饱和黄绿，
//      叠上去等于给整幅图加一层绿光。**只有饱和度上限能把它拉回中性。**
//
// uniform:
//   U[0] = (lx, ly, W, H)
//   U[1] = (radius, sat_max, —, —)
//
// ⚠️ 排序用插入排序（49 个元素，单线程，开销可忽略）。
//    WGSL 没有内置排序，而这里只需要中位数，不值得上排序网络。

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 2>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32, 3>;

const MAX_N: u32 = 49u;      // (2*3+1)^2

/// limit_saturation 的 WGSL 版（与 compose.limit_saturation 逐行对应）。
///
/// 按 ``mean + (c-mean)*k`` 缩放色度：保持平均亮度不变，只把"离中性多远"压下来。
/// 饱和度和整体缩放无关，所以要**迭代几次逼近**（Python 版也是 4 次）。
fn limit_saturation(c_in: vec3<f32>, sat_max: f32) -> vec3<f32> {
  var c = clamp(c_in, vec3<f32>(0.0), vec3<f32>(1.0));
  for (var it = 0; it < 4; it = it + 1) {
    let lo = min(c.x, min(c.y, c.z));
    let hi = max(c.x, max(c.y, c.z));
    // ⚠️ 全黑是没有色相的，Python 版直接返回中灰 —— 这里必须一致
    if (hi <= 1e-6) { return vec3<f32>(0.5, 0.5, 0.5); }
    let sat = (hi - lo) / hi;
    if (sat <= sat_max) { break; }
    let m = (c.x + c.y + c.z) / 3.0;
    c = clamp(vec3<f32>(m) + (c - vec3<f32>(m)) * (sat_max / sat),
              vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return c;
}

@compute @workgroup_size(1)
fn main() {
  let W: i32 = i32(U[0].z);
  let H: i32 = i32(U[0].w);
  let cx: i32 = i32(U[0].x);
  let cy: i32 = i32(U[0].y);
  let radius: i32 = i32(U[1].x);
  let sat_max: f32 = U[1].y;

  var med = vec3<f32>(0.7, 0.7, 0.7);      // Python 版窗口为空时的兜底值

  var n: u32 = 0u;
  // 先数一下有效元素（窗口在画面边缘会被裁掉一部分）
  for (var dy = -radius; dy <= radius; dy = dy + 1) {
    let y = cy + dy;
    if (y < 0 || y >= H) { continue; }
    for (var dx = -radius; dx <= radius; dx = dx + 1) {
      let x = cx + dx;
      if (x < 0 || x >= W) { continue; }
      n = n + 1u;
    }
  }

  if (n > 0u) {
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
      var a: array<f32, 49>;
      var k: u32 = 0u;
      for (var dy = -radius; dy <= radius; dy = dy + 1) {
        let y = cy + dy;
        if (y < 0 || y >= H) { continue; }
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
          let x = cx + dx;
          if (x < 0 || x >= W) { continue; }
          a[k] = src[(u32(y) * u32(W) + u32(x)) * 3u + ch];
          k = k + 1u;
        }
      }

      // 插入排序
      for (var i = 1u; i < n; i = i + 1u) {
        let key = a[i];
        var j = i;
        while (j > 0u && a[j - 1u] > key) {
          a[j] = a[j - 1u];
          j = j - 1u;
        }
        a[j] = key;
      }

      // ⚠️ numpy 的 median 规则：奇数取中间那个，偶数取中间两个的**平均**
      var m: f32;
      if ((n % 2u) == 1u) {
        m = a[n / 2u];
      } else {
        m = 0.5 * (a[n / 2u - 1u] + a[n / 2u]);
      }
      if (ch == 0u) { med.x = m; }
      else if (ch == 1u) { med.y = m; }
      else { med.z = m; }
    }
  }

  let c = limit_saturation(med, sat_max);
  dst[0] = c.x;
  dst[1] = c.y;
  dst[2] = c.z;
}
