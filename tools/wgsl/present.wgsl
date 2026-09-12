// 呈现：把像素尾巴产出的**打包 RGBA8（u32）**画到画布上。
//
// ⚠️ 这个 pass 刻意做成"只是搬运"，不含任何像素运算：
//    `tail.wgsl` 的数学已经在验收台里与 numpy **逐位比对**过，
//    写进 u32 缓冲的就是最终像素值。这里只做 bit 解包 + 上屏。
//    所以它不需要跑数值验收 —— 它没有可错的数学。
//
// 为什么要经过 u32 缓冲而不是让 tail 直接写纹理：
//    验收台验证的就是"写 u32 缓冲"这个版本的 tail。若为了上屏把它改成
//    走纹理，浏览器跑的就成了**另一个没被验证过**的着色器。
//    多一个 5 行的搬运 pass，换来"浏览器跑的就是验过的那份"。
//
// uniform:
//   U[0] = (W, H, —, —)

@group(0) @binding(0) var<storage, read> U: array<vec4<f32>, 1>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
};

// 全屏三角形（比全屏四边形少一次插值，且不需要顶点缓冲）
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0),
  );
  var o: VSOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let W: u32 = u32(U[0].x);
  let H: u32 = u32(U[0].y);
  // ⚠️ frag.xy 是像素中心（+0.5），所以直接截断就是像素下标。
  //    越界时钳住 —— 画布尺寸就是 W×H，正常不会越界，
  //    但窗口缩放的一瞬间画布尺寸可能还没跟上。
  let x = min(u32(frag.x), W - 1u);
  let y = min(u32(frag.y), H - 1u);
  let v = packed[y * W + x];
  return vec4<f32>(
    f32(v & 0xFFu) / 255.0,
    f32((v >> 8u) & 0xFFu) / 255.0,
    f32((v >> 16u) & 0xFFu) / 255.0,
    1.0,
  );
}
