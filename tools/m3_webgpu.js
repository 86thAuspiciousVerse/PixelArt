/* 浏览器端 WebGPU 渲染器 —— 把管线搬到显卡上跑。
 *
 * ═══ 设计要点 ═══
 *
 * ① **着色器由服务端提供同一个文件**（`/api/wgsl?name=X`）。
 *    这是刻意的：`tools/wgsl_lab.py` 验证过的就是那几个 .wgsl 文件的内容，
 *    所以"验过了"对浏览器同样成立。若把 WGSL 抄一份进 JS 字符串里，
 *    验收台验的与浏览器跑的就是两份东西，验收就白做了。
 *
 * ② **服务端只送"一次"那部分的数据**（base / far / 色板 / 核 / 粒子参数）。
 *    因为 `compose_frame` 全程在**网格分辨率**上跑，
 *    这些就是它需要的全部输入；滑杆变化不改变它们。
 *
 * ③ **uniform 在本地打包**（`m3_wgsl_uniforms.js`），否则每拖一下滑杆
 *    都要回服务端要一次 uniform —— 又变成了每帧一个往返。
 *    那份 JS 镜像了 Python，所以有 `tools/uniform_parity_probe.py`
 *    做逐字节比对把它钉死。
 *
 * ④ **失败一律回退**。任何一步出错都把 reason 交回界面，由界面切回服务端。
 *    GPU 是加速手段，不是必需条件。
 *
 * ═══ ⚠️ 一条很容易踩的 WebGPU 规则 ═══
 *
 * `queue.writeBuffer` **不能在一个 command buffer 内部交错**。
 * 它是在*队列*时间线上排序的：一次 submit 里的所有 dispatch 都会看到
 * 该缓冲的**最后一次**写入。
 *
 * 这一条直接决定了 bloom 的实现方式：三个尺度各要一次"横"一次"纵"、
 * 共 6 次 dispatch，而它们的 uniform 各不相同。若共用一个 uniform 缓冲、
 * 写 6 次再 submit 一次 —— 6 次 dispatch 会全部用最后一组参数，
 * 辉光就变成"只用最后一个尺度、且纵横参数错乱"。
 *
 * 所以 `bloom_blur` 用 **6 组独立的 uniform 缓冲 + 6 个 bind group**，
 * 各自只写一次，然后一次 submit 全部 dispatch。
 * （其余 pass 每帧只写一次自己的 uniform，不受这条规则影响。）
 */
(function (root) {
  'use strict';

  var U = root.WgslUniforms;

  var SHADERS = ['fog', 'scatter', 'volumetric', 'bloom_bright', 'bloom_blur',
                 'bloom_combine', 'dust_splat', 'dust_apply',
                 'parallax_splat', 'parallax_apply', 'tail', 'present'];

  //: 每个 pass 的 uniform 长度（float32 个数）。⚠️ 必须与 .wgsl 里
  //  `array<vec4<f32>, N>` 的 N 一致 —— N*4 就是这里的值。
  //  由 tools/browser_pipeline_check.py 解析 .wgsl 交叉核对。
  var UNIFORM_LEN = {
    fog: 24, scatter: 8, volumetric: 24,
    bloom_bright: 4, bloom_blur: 8, bloom_combine: 4,
    dust_splat: 16, dust_apply: 8,
    parallax_splat: 16, parallax_apply: 16,
    tail: 8, present: 4
  };

  //: 每个 pass 的 storage 绑定数（含末尾的输出）。同样由检查脚本核对。
  var BINDING_COUNT = {
    fog: 4, scatter: 3, volumetric: 5,
    bloom_bright: 3, bloom_blur: 5, bloom_combine: 4,
    dust_splat: 4, dust_apply: 4,
    parallax_splat: 5, parallax_apply: 5,
    tail: 5
  };

  //: 渲染管线（present）的绑定数。它不属于 compute 那套布局，
  //  所以单独一张表 —— 混在一起会让 _buildPipelines 试图给它建 compute 管线。
  var RENDER_BINDING_COUNT = { present: 2 };

  //: bloom 的尺度数（与 compose.BLOOM_RADII 的长度一致）
  var BLOOM_SCALES = 3;

  function GpuRenderer(canvas) {
    this.canvas = canvas;
    this.dev = null;
    this.ctx = null;
    this.modules = {};
    this.pipelines = {};
    this.buf = {};
    this.uni = {};
    this.bind = {};
    this.scene = null;
    this.assets = null;
    this.info = '';
    this.frames = 0;
    this.lastErr = '';
    this._blurSlots = [];
  }

  /* ── 设备 ─────────────────────────────────────────────── */
  GpuRenderer.prototype.init = function () {
    var self = this;
    if (!('gpu' in navigator)) {
      return Promise.reject(new Error('浏览器没有 WebGPU（navigator.gpu 不存在）'));
    }
    return navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      .then(function (adapter) {
        if (!adapter) throw new Error('没有可用的 GPU 适配器');
        try {
          var i = adapter.info || {};
          self.info = [i.vendor, i.architecture, i.device, i.description]
            .filter(Boolean).join(' ').trim() || '（适配器信息不可用）';
        } catch (e) { self.info = '（适配器信息不可用）'; }
        var limits = adapter.limits || {};
        // 早失败：超大网格可能超过 GPU 的 storage binding 上限
        self.limits = limits;
        return adapter.requestDevice();
      })
      .then(function (dev) {
        self.dev = dev;
        dev.lost.then(function (info) {
          self.lastErr = 'device lost: ' + (info && info.message ? info.message : '?');
          self.dev = null;
        });
        self.ctx = self.canvas.getContext('webgpu');
        if (!self.ctx) throw new Error('拿不到 webgpu 画布上下文');
        self.fmt = navigator.gpu.getPreferredCanvasFormat();
        self.ctx.configure({ device: dev, format: self.fmt, alphaMode: 'opaque' });
        return Promise.all(SHADERS.map(function (n) { return self._loadShader(n); }));
      })
      .then(function () { return self; });
  };

  GpuRenderer.prototype._loadShader = function (name) {
    var self = this;
    return fetch('/api/wgsl?name=' + encodeURIComponent(name))
      .then(function (r) {
        if (!r.ok) throw new Error('取 ' + name + '.wgsl 失败（HTTP ' + r.status + '）');
        return r.text();
      })
      .then(function (src) {
        var mod = self.dev.createShaderModule({ code: src, label: name });
        if (!mod.getCompilationInfo) { self.modules[name] = mod; return; }
        return mod.getCompilationInfo().then(function (ci) {
          var errs = (ci.messages || []).filter(function (m) { return m.type === 'error'; });
          if (errs.length) {
            throw new Error(name + '.wgsl 编译失败: ' +
              errs.slice(0, 3).map(function (m) {
                return 'L' + m.lineNum + ' ' + m.message;
              }).join(' | '));
          }
          self.modules[name] = mod;
        });
      });
  };

  /* ── 场景包解码 ───────────────────────────────────────── */
  // 格式：u32 头部长度 + JSON 头部 + 负载（见 server.pack_segments）
  GpuRenderer.decodePack = function (ab) {
    var dv = new DataView(ab);
    var hlen = dv.getUint32(0, true);
    var head = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, hlen)));
    var base = 4 + hlen;
    var segs = {};
    for (var k in head.segments) {
      var s = head.segments[k];
      segs[k] = new Uint8Array(ab, base + s.offset, s.nbytes);
    }
    return { meta: head.meta, segs: segs };
  };

  /* ── 缓冲工具 ─────────────────────────────────────────── */
  //: 工作缓冲的统一 usage。
  //  ⚠️⚠️ **必须带 COPY_SRC**，否则 `copyBufferToBuffer` 读回是非法的：
  //      WebGPU 会报 "usage (CopyDst|Storage) doesn't include CopySrc"，
  //      然后**把整个 command buffer 丢掉** —— 读回来全是 0，
  //      而 submit 不抛异常、渲染链其实一切正常。
  //      实测被这个坑了很久：诊断显示"所有缓冲都是 0（连 base/far 也是）"，
  //      看起来像"上传失败 / 渲染链断了"，实际是**读不回来**。
  //      → **测量工具的缺陷伪装成了被测对象的缺陷。**
  var WORK_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
                 | GPUBufferUsage.COPY_SRC;

  GpuRenderer.prototype._storage = function (u8) {
    var n = Math.max(4, Math.ceil(u8.byteLength / 4) * 4);
    var b = this.dev.createBuffer({
      size: n,
      usage: WORK_USAGE,
      mappedAtCreation: true
    });
    if (u8.byteLength) new Uint8Array(b.getMappedRange()).set(u8);
    b.unmap();
    return b;
  };

  GpuRenderer.prototype._blank = function (bytes, zero) {
    var b = this.dev.createBuffer({
      size: Math.max(4, bytes),
      usage: WORK_USAGE
    });
    if (zero) this.dev.queue.writeBuffer(b, 0, new Uint8Array(Math.max(4, bytes)));
    return b;
  };

  // uniform 缓冲不需要被读回（它们只是输入），所以不带 COPY_SRC
  // uniform 缓冲不需要被读回（它们只是输入），所以不带 COPY_SRC
  GpuRenderer.prototype._ubo = function (bytes) {
    return this.dev.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
  };

  GpuRenderer.prototype._destroyAll = function () {
    var k;
    for (k in this.buf) { try { this.buf[k].destroy(); } catch (e) {} }
    for (k in this.uni) {
      var v = this.uni[k];
      if (Array.isArray(v)) v.forEach(function (b) { try { b.destroy(); } catch (e) {} });
      else { try { v.destroy(); } catch (e) {} }
    }
    this.buf = {}; this.uni = {}; this.bind = {}; this._blurSlots = [];
  };

  /* ── 上传：场景（base / far） ─────────────────────────── */
  GpuRenderer.prototype.setScene = function (pack) {
    var g = pack.meta.grid, W = g[0], H = g[1];
    if (!pack.segs.base || !pack.segs.far){
      throw new Error('场景包缺少 base/far（只传了 assets？setScene 需要完整场景）');
    }
    var npix = W * H, n3 = npix * 3;
    this.scene = { W: W, H: H, npix: npix, n3: n3, meta: pack.meta };
    // ⚠️ 留下 base/far 的字节：后面只刷新"素材"（色板/核/粒子）时，
    //    要能把新素材合并进这份场景段重传，否则 setScene 会把缓冲清掉。
    this._sceneSegs = pack.segs;

    this._destroyAll();
    var b = this.buf;
    b.base = this._storage(pack.segs.base);
    b.far = this._storage(pack.segs.far);

    // 工作缓冲（3N 个 f32）：
    //   a       雾输出（也是 scatter / volumetric 的输入）
    //   lit     体积光输出 = **不含粒子的合成图** = tail 的 edge_ref
    //   dusted  尘埃后 = 辉光的 rgb 源
    //   bright  高光提取结果
    //   tmp     模糊"横"的输出
    //   acc/acc2 模糊"纵"的累积（ping-pong，到最后落在 acc）
    //   composed 辉光合并结果 = tail 的输入
    //   zero    全零（第一段累积的 prev，以及"输入全零"用）
    b.a = this._blank(n3 * 4);
    b.lit = this._blank(n3 * 4);
    b.lit2 = this._blank(n3 * 4);             // 副光源链的输出（M5.6）
    b.dusted = this._blank(n3 * 4);
    b.bright = this._blank(n3 * 4);
    b.tmp = this._blank(n3 * 4);
    b.acc = this._blank(n3 * 4);
    b.acc2 = this._blank(n3 * 4);
    b.composed = this._blank(n3 * 4);
    b.zero = this._blank(n3 * 4, true);
    b.air = this._blank(3 * 4);
    b.air2 = this._blank(3 * 4);              // 副光源的散射色（M5.6）
    b.bin = this._blank(npix * 4, true);      // u32 定点累加（原子加目标）
    b.packed = this._blank(npix * 4);         // 打包 RGBA8

    /* ── 分层视差（M5）──
     * warp 是一块**单缓冲**：[base 区 3N][far 区 N]，far 区起点对齐到 64 float。
     * 下游 fog / volumetric / dust 的 base/far 绑定直接指向它 ——
     * 管线形状固定（不再需要"warp 版/原版"两套），因为 apply pass 在
     * k=0 时走**逐位拷贝**分支，输出与原件完全相同。
     * paracc 是 u32 定点累加器（stride 5：[r,g,b,w,far·w]），前向 splat 的目标。 */
    var L = pack.meta.par_layout || defaultParLayout(npix);
    this.par = { far_off: L.far_off, total: L.total, npix: npix };
    b.warp = this._blank(L.total * 4);
    b.paracc = this._blank(npix * 5 * 4, true);
    if (pack.segs.par_edges) {
      b.par_edges = this._storage(pack.segs.par_edges);
    }

    this._buildPipelines();
    return this;
  };

  /* ── 上传：素材（色板 / 核 / 粒子 / 常量） ───────────── */
  GpuRenderer.prototype.setAssets = function (pack) {
    this.assets = pack.meta;
    var b = this.buf;
    b.palette = this._storage(pack.segs.palette);
    b.kernels = this._storage(pack.segs.kernels);
    b.dust = this._storage(pack.segs.dust);
    this._buildBindGroups();
    return this;
  };

  /* ── 管线 / 布局 ──────────────────────────────────────── */
  function computeLayout(dev, count) {
    var e = [];
    for (var i = 0; i < count; i++) {
      e.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        // ⚠️ 只有最后一个绑定是可写的（各 .wgsl 都是"输出在最后"）。
        //    类型填错会被拒，而且报错里**没有绑定名**，很难定位。
        buffer: { type: (i === count - 1 ? 'storage' : 'read-only-storage') }
      });
    }
    return dev.createBindGroupLayout({ entries: e });
  }

  /* 绑定组。条目可以是 buffer，也可以是 {buffer, offset, size} ——
   * 后者用于把 fog/volumetric/dust 的 base/far 指向**视差 warp 缓冲**
   * 的不同区段（far 区必须在 64 个 float 对齐的偏移上）。 */
  /* 视差 warp 缓冲布局：base 区 3N + far 区 N，far 区对齐到 64 个 float。
   * ⚠️ 必须与 src/pixelart/webgpu.parallax_layout 完全一致 ——
   * 服务端在场景包 meta 里也给了 par_layout，这里只是缺省兜底。 */
  function defaultParLayout(npix) {
    var farOff = Math.ceil(npix * 3 / 64) * 64;
    return { far_off: farOff, total: farOff + npix };
  }

  function bindGroup(dev, layout, bufs) {
    return dev.createBindGroup({
      layout: layout,
      entries: bufs.map(function (buf, i) {
        var res = (buf && buf.buffer) ? buf : { buffer: buf };
        return { binding: i, resource: res };
      })
    });
  }

  GpuRenderer.prototype._buildPipelines = function () {
    var self = this, d = this.dev;
    this.lay = {};
    var names = ['fog', 'scatter', 'volumetric', 'bloom_bright', 'bloom_blur',
                 'bloom_combine', 'dust_splat', 'dust_apply',
                 'parallax_splat', 'parallax_apply', 'tail'];
    names.forEach(function (n) {
      self.lay[n] = computeLayout(d, BINDING_COUNT[n]);
      var key = n + '|main';
      self.pipelines[key] = d.createComputePipeline({
        layout: d.createPipelineLayout({ bindGroupLayouts: [self.lay[n]] }),
        compute: { module: self.modules[n], entryPoint: 'main' }
      });
    });

    /* ── 副光源（M5.6）：**复用** scatter / volumetric 同款 shader 模块，
     * 独立管线与 bind group。不进 SHADERS/BINDING_COUNT 表 —— 那两张表
     * 会被 browser_pipeline_check.py 按 .wgsl 交叉核对，副光源没有自己的
     * .wgsl（它就是同一份 shader 换一组 uniform/绑定再跑一遍）。 */
    [['scatter2', 'scatter'], ['volumetric2', 'volumetric']].forEach(function (pair) {
      var name = pair[0], base = pair[1];
      self.lay[name] = self.lay[base];
      self.pipelines[name + '|main'] = d.createComputePipeline({
        layout: d.createPipelineLayout({ bindGroupLayouts: [self.lay[base]] }),
        compute: { module: self.modules[base], entryPoint: 'main' }
      });
    });

    this.presentLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
      ]
    });
    this.present = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.presentLayout] }),
      vertex: { module: this.modules.present, entryPoint: 'vs' },
      fragment: {
        module: this.modules.present, entryPoint: 'fs',
        targets: [{ format: this.fmt }]
      },
      primitive: { topology: 'triangle-list' }
    });
  };

  GpuRenderer.prototype._buildBindGroups = function () {
    var d = this.dev, b = this.buf;
    // 每个 pass 的 uniform 只写一次/帧，所以可以共用缓冲；
    // **bloom_blur 例外**（同帧写 6 次），见文件头那条规则。
    Object.keys(BINDING_COUNT).forEach(function (n) {
      if (n !== 'bloom_blur') this.uni[n] = this._ubo(UNIFORM_LEN[n] * 4);
    }, this);
    this.uni.present = this._ubo(UNIFORM_LEN.present * 4);

    // bloom_blur：3 个尺度 × (横, 纵) = 6 组，各自独立
    this.uni.bloom_blur = [];
    for (var i = 0; i < BLOOM_SCALES * 2; i++) {
      this.uni.bloom_blur.push(this._ubo(UNIFORM_LEN.bloom_blur * 4));
    }

    var L = this.lay, g = bindGroup;
    this.bind.fog = g(d, L.fog, [this.uni.fog, b.base, b.far, b.a]);
    this.bind.scatter = g(d, L.scatter, [this.uni.scatter, b.a, b.air]);
    this.bind.volumetric = g(d, L.volumetric, [this.uni.volumetric, b.a, b.far, b.air, b.lit]);
    this.bind.dust_splat = g(d, L.dust_splat, [this.uni.dust_splat, b.dust, b.far, b.bin]);
    this.bind.dust_apply = g(d, L.dust_apply, [this.uni.dust_apply, b.bin, b.lit, b.dusted]);
    this.bind.bloom_bright = g(d, L.bloom_bright, [this.uni.bloom_bright, b.dusted, b.bright]);
    this.bind.bloom_combine = g(d, L.bloom_combine, [this.uni.bloom_combine, b.dusted, b.acc, b.composed]);
    this.bind.tail = g(d, L.tail, [this.uni.tail, b.composed, b.lit, b.palette, b.packed]);

    // ── 副光源（M5.6）：air2 / lit2 链。volumetric2 的输入 = 主链输出 lit，
    //    WGSL volumetric 的出口就是 clip(输入 + vol) → 两次链式 = 两次线性叠加，
    //    与 CPU 的 clip(clip(fogged+vol1)+vol2) 逐字对应。
    this.uni.scatter2 = this._ubo(UNIFORM_LEN.scatter * 4);
    this.uni.volumetric2 = this._ubo(UNIFORM_LEN.volumetric * 4);
    this.bind.scatter2 = g(d, L.scatter, [this.uni.scatter2, b.a, b.air2]);
    this.bind.volumetric2 = g(d, L.volumetric,
      [this.uni.volumetric2, b.lit, b.far, b.air2, b.lit2]);
    this.bind.dust_apply2 = g(d, L.dust_apply,
      [this.uni.dust_apply, b.bin, b.lit2, b.dusted]);

    // ── 分层视差：两个 pass + 三个"读 warp"的下游绑定组 ──
    if (b.par_edges){
      var n3b = this.scene.n3 * 4, par = this.par;
      var wBase = { buffer: b.warp, offset: 0, size: n3b };
      var wFar = { buffer: b.warp, offset: par.far_off * 4, size: par.npix * 4 };
      this.bind.parallax_splat = g(d, L.parallax_splat,
        [this.uni.parallax_splat, b.par_edges, b.base, b.far, b.paracc]);
      this.bind.parallax_apply = g(d, L.parallax_apply,
        [this.uni.parallax_apply, b.paracc, b.base, b.far, b.warp]);
      this.bind.fog_par = g(d, L.fog, [this.uni.fog, wBase, wFar, b.a]);
      this.bind.volumetric_par = g(d, L.volumetric,
        [this.uni.volumetric, b.a, wFar, b.air, b.lit]);
      // 副光源的深度遮挡也读 warp 后的 far（与主光源一致）
      this.bind.volumetric2_par = g(d, L.volumetric,
        [this.uni.volumetric2, b.lit, wFar, b.air2, b.lit2]);
      this.bind.dust_splat_par = g(d, L.dust_splat,
        [this.uni.dust_splat, b.dust, wFar, b.bin]);
      this.presentLayout = this.presentLayout;   // （保持可读性，勿删）
    }
    this.bind.present = g(d, this.presentLayout, [this.uni.present, b.packed]);

    // bloom_blur 的 6 个 bind group：源/目标是固定的 chain
    //   slot i*2   （横）: src=bright, dst=tmp
    //   slot i*2+1 （纵）: src=tmp,    dst=acc/acc2 交替, prev 同 dst 缓冲的旧值
    this._blurSlots = [];
    for (var s = 0; s < BLOOM_SCALES; s++) {
      var dst = (s % 2 === 0) ? b.acc : b.acc2;
      var prev = (s === 0) ? b.zero : ((s % 2 === 0) ? b.acc2 : b.acc);
      this._blurSlots.push({
        h: g(d, L.bloom_blur, [this.uni.bloom_blur[s * 2], b.kernels, b.bright, b.zero, b.tmp]),
        v: g(d, L.bloom_blur, [this.uni.bloom_blur[s * 2 + 1], b.kernels, b.tmp, prev, dst])
      });
    }
    // 3 个尺度（奇数个）→ 最后落在 acc
    this._accFinal = b.acc;
  };

  /* ── 渲染一帧 ─────────────────────────────────────────── */
  GpuRenderer.prototype.render = function (p, t) {
    if (!this.dev || !this.scene || !this.assets) return false;
    var d = this.dev, b = this.buf, sc = this.scene;
    var W = sc.W, H = sc.H, npix = sc.npix;
    var A = this.assets;

    var write = function (name, arr) {
      d.queue.writeBuffer(this.uni[name], 0, arr);
    }.bind(this);

    /* ── 组装各 pass 的 uniform（每帧一遍；几千字节，可忽略） ── */
    write('fog', U.fogUniforms({
      w: W, h: H, t: t,
      fogColor: this._fogColor(p, sc.meta),
      density: p.density, power: p.power, floor: 0, ceiling: 1,
      drift: p.fog_drift, harmonics: A.fog_harmonics,
      fogTint: (p.fog_tint === undefined) ? 1 : +p.fog_tint
    }));

    var gain = U.flickerGain(t, A.flicker_phases, A.flicker_freqs, p.flicker);
    // ⚠️ 光源位置必须**每帧解析**（M3-WIRE 同类问题第 6 次）：
    //    原来恒读 sc.meta.light_xy —— 那是**场景包时刻**的快照，而手动滑杆
    //    (rays_x/rays_y) 不是场景键、不触发重取场景包 → GPU 模式下手动滑杆
    //    完全无效，且界面标记（读滑杆/新鲜 meta）与渲染顶点（冻结值）分叉。
    //    自动模式用 light_xy_auto（场景检测结果，只依赖 scene，冻结是对的）；
    //    手动模式用滑杆即时值 —— 与 CPU 侧 tune.TuneParams.light_xy 一字对齐。
    var lightAuto = (p.rays_auto_center === undefined) ? true : !!p.rays_auto_center;
    var lsrc = sc.meta.light_xy_auto || sc.meta.light_xy;
    var lx = lightAuto ? lsrc[0] : +p.rays_x;
    var ly = lightAuto ? lsrc[1] : +p.rays_y;
    if (!isFinite(lx) || !isFinite(ly)) { lx = lsrc[0]; ly = lsrc[1]; }
    var lxp = Math.round(lx * (W - 1)), lyp = Math.round(ly * (H - 1));

    write('scatter', U.scatterUniforms({
      w: W, h: H, lx: lxp, ly: lyp, radius: 3, satMax: 0.25
    }));
    write('volumetric', U.volumetricUniforms({
      w: W, h: H, samples: 28, span: 0.85, decay: 0.965,
      strength: p.rays, occludeGain: 5.0, falloffGain: 2.5,
      screenFalloff: p.rays_spread, threshold: 0.48, knee: 0.3, gain: gain,
      lx: lxp, ly: lyp,
      coneX: (p.cone_x === undefined) ? -1 : +p.cone_x,
      coneY: (p.cone_y === undefined) ? -1 : +p.cone_y,
      coneAngle: +p.rays_cone || 0, coneDirDeg: +p.rays_dir || 0,
      coneReach: (p.rays_reach === undefined) ? 0.8 : +p.rays_reach,
      shaft: +p.rays_shaft || 0,
      fogTint: (p.fog_tint === undefined) ? 1 : +p.fog_tint
    }));
    // ── 副光源（M5.6）：off 时零成本（不写 uniform、不派发）──
    var useL2 = !!p.light2_on;
    if (useL2){
      var l2x = Math.round((+p.light2_x) * (W - 1)), l2y = Math.round((+p.light2_y) * (H - 1));
      write('scatter2', U.scatterUniforms({
        w: W, h: H, lx: l2x, ly: l2y, radius: 3, satMax: 0.25
      }));
      // ⚠️ 副光源 v1 无锥（coneAngle=0）；衰减/遮挡参数与主光源同一套；
      //    闪烁增益同主光（同一 flicker 旋钮，循环闭合不破坏）。
      write('volumetric2', U.volumetricUniforms({
        w: W, h: H, samples: 28, span: 0.85, decay: 0.965,
        strength: (p.light2_gain === undefined) ? 1 : +p.light2_gain,
        occludeGain: 5.0, falloffGain: 2.5,
        screenFalloff: (p.light2_spread === undefined) ? 1 : +p.light2_spread,
        threshold: 0.48, knee: 0.3, gain: gain,
        lx: l2x, ly: l2y,
        coneX: -1, coneY: -1,
        coneAngle: 0, coneDirDeg: 0, coneReach: 1,
        shaft: 0,
        fogTint: (p.fog_tint === undefined) ? 1 : +p.fog_tint
      }));
    }
    write('dust_splat', U.dustSplatUniforms({
      w: W, h: H, count: A.dust_count, t: t,
      twinkle: p.dust_twinkle, fadeFar: p.dust_fade_far,
      lightBoost: p.dust_light_boost,
      hasLight: true, lightX: lx, lightY: ly,
      scale: A.dust_scale, hasFar: true
    }));
    write('dust_apply', U.dustApplyUniforms({
      w: W, h: H, scale: A.dust_scale, dustBright: p.dust_bright
    }));
    write('bloom_bright', U.bloomBrightUniforms({
      w: W, h: H, threshold: A.bloom_threshold, knee: A.bloom_knee
    }));
    write('bloom_combine', U.bloomCombineUniforms({
      w: W, h: H, mult: (p.bloom * 0.5) / A.bloom_wsum
    }));
    write('tail', U.tailUniforms({
      w: W, h: H, nPalette: A.palette_len,
      dither: p.dither, ditherAdaptive: true,
      edgeGain: p.edge_gain, edgeStrength: p.edge_strength,
      continuous: (p.quantize_mode === "continuous")
    }));
    d.queue.writeBuffer(this.uni.present, 0, new Float32Array([W, H, 0, 0]));

    /* ── 分层视差（M5）──
     * 位移必须发生在雾/体积光**之前**（像素搬走，它的深度也跟着搬），
     * 所以这两个 uniform 在其它 pass 之前写好、两个 pass 最先派发。
     * ⚠️ k=0（t=0 / t=1 / amp=0）时**不派发 splat**，apply 走逐位拷贝分支 ——
     * 这样"视差关"的输出与不接这条链时**逐位相同**。 */
    var parAmp = +p.parallax || 0;
    var parWave = U.parallaxWave(t);
    var usePar = !!(b.par_edges && A.par_n_edges);
    var parK = 0;
    if (usePar){
      parK = (parAmp > 0) ? U.parallaxK(parAmp, W, t) : 0;
      var pv = A.par_pivot || [0.5, 0.42];
      d.queue.writeBuffer(this.uni.parallax_splat, 0, U.parallaxSplatUniforms({
        w: W, h: H, k: parK, wave: parWave,
        pivotX: pv[0] * (W - 1), pivotY: pv[1] * (H - 1),
        drift: A.par_drift, nEdges: A.par_n_edges,
        nearGain: A.par_near_gain, farGain: A.par_far_gain,
        scale: A.par_scale
      }));
      d.queue.writeBuffer(this.uni.parallax_apply, 0, U.parallaxApplyUniforms({
        w: W, h: H, scale: A.par_scale, k: parK, farOff: this.par.far_off
      }));
    }

    // bloom_blur 的 6 组：**必须在 submit 之前全部写好**（它们各自独立，
    // 所以不存在"写 6 次只生效最后一次"的问题）
    for (var s = 0; s < BLOOM_SCALES; s++) {
      var e = A.kbuf_table[s];
      d.queue.writeBuffer(this.uni.bloom_blur[s * 2], 0, U.bloomBlurUniforms({
        w: W, h: H, horiz: true, accum: false,
        offset: e[0], count: e[1], half: e[2], weight: e[3]
      }));
      d.queue.writeBuffer(this.uni.bloom_blur[s * 2 + 1], 0, U.bloomBlurUniforms({
        w: W, h: H, horiz: false, accum: true,
        offset: e[0], count: e[1], half: e[2], weight: e[3]
      }));
    }

    /* ── 一条 encoder、一次 submit ── */
    var enc = d.createCommandEncoder();
    var gx = Math.ceil(W / 8), gy = Math.ceil(H / 8);

    // 尘埃的定点缓冲必须每帧清零（原子加从零开始）
    enc.clearBuffer(b.bin);
    // 视差的定点累加同理（原子加从零开始）。
    // ⚠️ 必须在 beginComputePass **之前** —— 在 pass 里调 clearBuffer 是非法编码，
    //    整条 command buffer 会被丢弃，症状是画布全黑且不报错。
    if (usePar) enc.clearBuffer(b.paracc);

    var cp = enc.beginComputePass();
    var run = function (name, group, dx, dy) {
      cp.setPipeline(this.pipelines[name + '|main']);
      cp.setBindGroup(0, group);
      cp.dispatchWorkgroups(dx, dy);
    }.bind(this);

    if (usePar){
      // ⚠️ 清零**不能**放在这里 —— compute pass 已经打开，
      //    "Recording in [CommandEncoder] which is locked while
      //     [ComputePassEncoder] is open" 会让**整条命令缓冲作废**（全黑）。
      //    清零在 beginComputePass 之前，见上面 b.in 那两行。
      if (Math.abs(parK) >= 1e-9) {
        run('parallax_splat', this.bind.parallax_splat, gx, gy);
      }
      run('parallax_apply', this.bind.parallax_apply, gx, gy);
    }
    run('fog', usePar ? this.bind.fog_par : this.bind.fog, gx, gy);
    run('scatter', this.bind.scatter, 1, 1);          // 单线程（只有一个标量要算）
    run('volumetric', usePar ? this.bind.volumetric_par : this.bind.volumetric, gx, gy);
    // 副光源链（M5.6）：scatter2 → volumetric2（读 lit 加 vol2 写 lit2）
    if (useL2){
      run('scatter2', this.bind.scatter2, 1, 1);
      run('volumetric2', usePar ? this.bind.volumetric2_par : this.bind.volumetric2, gx, gy);
    }
    // 粒子：**每颗粒子一个线程** —— 线程数与像素数无关，所以用 1D dispatch
    run('dust_splat', usePar ? this.bind.dust_splat_par : this.bind.dust_splat,
        Math.ceil(A.dust_count / 64), 1);
    run('dust_apply', useL2 ? this.bind.dust_apply2 : this.bind.dust_apply, gx, gy);
    run('bloom_bright', this.bind.bloom_bright, gx, gy);
    for (var k = 0; k < BLOOM_SCALES; k++) {
      run('bloom_blur', this._blurSlots[k].h, gx, gy);
      run('bloom_blur', this._blurSlots[k].v, gx, gy);
    }
    run('bloom_combine', this.bind.bloom_combine, gx, gy);
    run('tail', this.bind.tail, gx, gy);
    cp.end();

    var rp = enc.beginRenderPass({
      colorAttachments: [{
        view: this.ctx.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 }
      }]
    });
    rp.setPipeline(this.present);
    rp.setBindGroup(0, this.bind.present);
    rp.draw(3, 1, 0, 0);
    rp.end();

    d.queue.submit([enc.finish()]);
    this.frames++;
    return true;
  };

  /* ── 雾色 ─────────────────────────────────────────────── */
  GpuRenderer.prototype._fogColor = function (p, m) {
    if (p.fog_r >= 0 && p.fog_g >= 0 && p.fog_b >= 0) {
      return [p.fog_r, p.fog_g, p.fog_b];
    }
    // ⚠️ 用 U.limitSaturation（共享实现），不要在这里另写一份 ——
    //    这个函数由 tools/uniform_parity_probe.py 与 Python 逐位比对。
    return U.limitSaturation(m.fog_color_raw, p.fog_sat);
  };

  root.GpuRenderer = GpuRenderer;
  root.GPU_SHADERS = SHADERS;
  root.GPU_UNIFORM_LEN = UNIFORM_LEN;
  root.GPU_BINDING_COUNT = BINDING_COUNT;
  root.GPU_RENDER_BINDING_COUNT = RENDER_BINDING_COUNT;
})(typeof globalThis !== 'undefined' ? globalThis : this);
