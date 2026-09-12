"""WGSL 验收台 —— 用 wgpu 无头执行计算着色器，与 numpy 参考实现逐像素比对。

═══ 为什么要有这个工具 ═══

把渲染从 Python 搬到 WebGPU 是个**大搬家**。搬对了没人知道，搬错了——
错了就是"看起来差不多但某处不对"，靠对比截图根本发现不了：
雾的指数曲线差 2%、色板吸附在边界翻了个色、抖动相位偏了一格……
这些都不会让人觉得"坏了"，只会让人觉得"好像不如以前好看"。

所以验收必须是**数值比对**，不是看图。

═══ 为什么能这么做 ═══

`wgpu-py` 直接暴露与 WebGPU 同一套 API，而本机可用后端里就有
**Vulkan 和 D3D12** —— 这正是浏览器在 Windows 上会走的两个后端。
所以在这里跑出来的结果，和用户浏览器里的结果**走的是同一段驱动代码**，
不是某种模拟。

而且成本极低：不需要浏览器、不需要截图、不需要人眼，`pytest` 里 1 秒跑完。

═══ 与 numpy 的差距从哪来 ═══

两类，必须分开对待：

1. **超越函数**（``exp`` / ``pow`` / ``sin``）—— GPU 与 libm 的实现在末位有差异，
   这是 IEEE-754 允许的。所以这类阶段用**容差**判定。
2. **纯比较 / 纯查表**（色板吸附、Bayer 抖动）—— 这类**必须逐位相同**。
   如果不同，那就是逻辑错了（或出现了并列距离的平局），不是精度问题。

这个区分很重要：**把第二类用容差放过，等于把真 bug 放过去。**

用法::

    python tools/wgsl_lab.py            # 跑全部阶段
    python tools/wgsl_lab.py --stage fog
    python tools/wgsl_lab.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WGSL_DIR = Path(__file__).resolve().parent / "wgsl"
sys.path.insert(0, str(ROOT / "src"))

#: 计算着色器一律用这个 workgroup 尺寸（8×8 = 64 线程）
WORKGROUP = 8


# ══════════════════════════════════════════════════════════════════
# wgpu 环境的建立
# ══════════════════════════════════════════════════════════════════
class WgslLab:
    """最小可用的 WGSL 执行环境。

    只做一件奢侈的事：**把出错的信息说清楚**。wgpu 的原生报错经常只有一行，
    而着色器编译失败时定位不到行号 —— 所以这里会把 WGSL 源码连同行号一起打出来。
    """

    def __init__(self, backend: str | None = None):
        import wgpu

        self.wgpu = wgpu
        self.backend = backend
        if backend:
            # backend 形如 "vulkan" / "d3d12"
            import os
            os.environ["WGPU_BACKEND_TYPE"] = backend

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter = adapter
        self.device = adapter.request_device_sync()
        self.info = dict(adapter.info) if hasattr(adapter, "info") else {}

        # ⚠️ 缓存编译产物。**不缓存的话每个 pass 的耗时会由"创建管线"主导**：
        #    实测 480×270 上一个 12 ms 的 pass，其中绝大部分是
        #    create_shader_module + create_compute_pipeline —— 不是计算。
        #    第一版没缓存，于是基准测试量出来的是"wgpu-py 的分配开销"，
        #    而不是 GPU 的算力。**测错了对象比测错数值更危险。**
        self._modules: dict[str, object] = {}
        self._layouts: dict[tuple, object] = {}
        self._pipelines: dict[tuple, object] = {}

        # ⚠️ 缓冲池。有了它才能量出"真实计算量"：
        #    每次 make_buffer 都新建缓冲区的话，耗时会由分配 + 上传主导，
        #    而不是由 dispatсh + 计算主导。浏览器里缓冲是常驻的，所以要能模拟那种情况。
        self._pool: dict[int, list] = {}
        self.pool_buffers = False        # 打开后 make_buffer 走池子

    def describe(self) -> str:
        i = self.info
        return (f"{i.get('device', '?')} | {i.get('backend_type', '?')} | "
                f"{i.get('adapter_type', '?')}")

    # ---------------------------------------------------------- 着色器
    def _compile(self, src: str, name: str = "shader"):
        """编译 WGSL（带缓存）。失败时把带行号的源码打出来 —— 否则没法定位。"""
        cached = self._modules.get(src)
        if cached is not None:
            return cached
        try:
            mod = self.device.create_shader_module(code=src, label=name)
            self._modules[src] = mod
            return mod
        except Exception as e:                                  # noqa: BLE001
            lines = src.splitlines()
            print(f"\n[WGSL 编译失败] {name}")
            print(f"  {type(e).__name__}: {e}")
            print("  ── 源码（带行号）──")
            for i, ln in enumerate(lines, 1):
                print(f"  {i:4d} | {ln}")
            raise

    # ---------------------------------------------------------- 缓冲区
    def make_buffer(self, arr: np.ndarray, usage_extra=None):
        """上传一个 numpy 数组为 GPU buffer（必须是 C 连续的 float32/uint32）。

        ``self.pool_buffers`` 打开时走缓冲池（按字节数复用 + ``write_buffer`` 上传），
        这是**量真实计算量所必需的** —— 否则耗时由分配主导。
        """
        import wgpu

        a = np.ascontiguousarray(arr)
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        if usage_extra:
            usage |= usage_extra
        if not self.pool_buffers:
            return self.device.create_buffer_with_data(data=a.tobytes(), usage=usage)

        n = int(a.nbytes)
        key = (n, int(usage))
        lst = self._pool.setdefault(key, [])
        if lst:
            buf = lst.pop()
            # ⚠️ 复用必须**整段重写**，否则会残留上一次的内容。
            #    粒子那个 pass 的输出缓冲尤其要清零（原子加从零开始）。
            self.device.queue.write_buffer(buf, 0, a.tobytes())
            return buf
        buf = self.device.create_buffer(size=max(4, n), usage=usage)
        self.device.queue.write_buffer(buf, 0, a.tobytes())
        self._pool.setdefault(key, [])
        return buf

    def recycle(self, buf, nbytes: int, usage: int):
        """把缓冲还回池子。"""
        self._pool.setdefault((int(nbytes), int(usage)), []).append(buf)

    def read_buffer(self, buf, shape, dtype=np.float32):
        import wgpu

        n = int(np.prod(shape))
        out = self.device.queue.read_buffer(buf, 0, n * np.dtype(dtype).itemsize)
        return np.frombuffer(out, dtype=dtype).reshape(shape).copy()

    # ---------------------------------------------------------- 执行
    def run(
        self,
        wgsl: str,
        entry: str,
        size: tuple[int, int] | None,
        inputs: dict[str, np.ndarray],
        uniforms: np.ndarray,
        out_count: int,
        out_dtype=np.float32,
        out_shape_tail: tuple | None = None,
        label: str = "pass",
        workgroup: tuple[int, int] = (WORKGROUP, WORKGROUP),
        dispatch: tuple[int, int] | None = None,
        readback: bool = True,
        repeat: int = 1,
    ) -> np.ndarray | None:
        """跑一个计算着色器。

        Args:
            size: (W, H) 网格尺寸，按 ``workgroup`` 折算成 dispatch 数。
                ``None`` 时必须显式给 ``dispatch``。
            dispatch: 直接给 dispatch 数。**粒子那种"线程数 ≠ 像素数"的阶段必须用它**
                —— 每颗粒子一个线程，而输出是整张图，两者没有对应关系。
            workgroup: 着色器里 ``@workgroup_size`` 的声明值，必须一致。
            inputs: ``{binding_name: array}`` —— 按 binding 顺序绑定为 storage buffer。
            uniforms: 一个 float32 数组。
            out_count: 输出数组的元素个数（= W*H*通道数）。
            repeat: 在**同一个 command encoder 内**把 dispatch 重复几次。
                ⚠️ 这是量"真实 GPU 计算量"的唯一干净办法：
                从 Python 调 wgpu 时，**每次调用**的固定开销
                （write_buffer + create_bind_group + submit + FFI）
                远大于 13 万像素的计算量本身，会把结果完全掩盖。
                把 N 次 dispatch 塞进一个 encoder，固定开销被摊薄 N 倍。
                ⚠️ ``repeat > 1`` 时结果同样是**无意义的**（重复覆盖同一输出），只测时序。
            readback: 是否把输出读回 CPU。**基准测试时要关掉** ——
                每次回读都是一次 GPU→CPU 同步，14 个 pass 累起来能占总时间的八成，
                而那在浏览器里根本不存在（缓冲常驻、不需要回读）。
        """
        import wgpu

        if dispatch is None:
            if size is None:
                raise ValueError("size 与 dispatch 必须给一个")
            w, h = size
            dispatch = ((w + workgroup[0] - 1) // workgroup[0],
                        (h + workgroup[1] - 1) // workgroup[1])
        # 记下每个缓冲的 (字节数, usage)，跑完要还回池子。
        # ⚠️ 第一版**只取不还** —— 池子永远是空的，于是每次仍然新建缓冲，
        #    "缓冲池"名存实亡（量出来的还是分配开销）。
        #    **一个不会回收的池子等于没有池子**，而它看起来还在工作。
        pooled: list[tuple] = []

        def _mk(arr, usage_extra=None):
            b = self.make_buffer(arr, usage_extra=usage_extra)
            if self.pool_buffers:
                usage = (wgpu.BufferUsage.STORAGE
                         | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
                if usage_extra:
                    usage |= usage_extra
                pooled.append((b, int(np.ascontiguousarray(arr).nbytes), int(usage)))
            return b

        ubuf = _mk(uniforms.astype(np.float32))
        in_bufs = [_mk(a) for a in inputs.values()]
        out_arr = np.zeros(out_count, dtype=out_dtype)
        outbuf = _mk(out_arr, usage_extra=wgpu.BufferUsage.COPY_SRC)

        # 用 override 把尺寸传给着色器，避免每种尺寸都要重编译 + 改常量
        module = self._compile(wgsl, label)

        entries = [{"binding": 0, "resource": {"buffer": ubuf, "offset": 0,
                                               "size": ubuf.size}}]
        b = 1
        for buf in in_bufs:
            entries.append({"binding": b, "resource": {"buffer": buf, "offset": 0,
                                                       "size": buf.size}})
            b += 1
        out_binding = b
        entries.append({"binding": out_binding, "resource": {"buffer": outbuf,
                                                             "offset": 0,
                                                             "size": outbuf.size}})

        # ⚠️ 只有**最后一个** binding 是可写的（着色器里 `read_write`），
        #    其余一律 `read`。绑定类型填错会被 wgpu 拒绝：
        #    "Storage class Storage{LOAD|STORE} doesn't match Storage{LOAD}"。
        #    这个错误信息不含 binding 名，所以这里按"最后一个即输出"的约定统一处理。
        lkey = (out_binding, len(entries))
        layout = self._layouts.get(lkey)
        if layout is None:
            layout = self.device.create_bind_group_layout(entries=[
                {"binding": e["binding"],
                 "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": (wgpu.BufferBindingType.storage
                                     if e["binding"] == out_binding
                                     else wgpu.BufferBindingType.read_only_storage)}}
                for e in entries
            ])
            self._layouts[lkey] = layout

        bind = self.device.create_bind_group(layout=layout, entries=entries)

        # ⚠️ 缓存键里**必须包含着色器本身**。
        #    第一版用 `(label, entry, lkey)` —— 而 label 默认都是 "pass"，
        #    而"binding 数 + 输出 binding"又常常相同，于是不同着色器**共享了管线**：
        #    fog 的 uniform 是 6 个 vec4、dust_splat 是 4 个，
        #    复用后直接报 "bound with size 64 where the shader expects 96"。
        #    教训：缓存键必须能**唯一确定被缓存的东西**；
        #    "调用方给的名字"不是身份标识，**内容才是**。
        pkey = (hash(wgsl), entry, lkey)
        pipeline = self._pipelines.get(pkey)
        if pipeline is None:
            pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
                compute={"module": module, "entry_point": entry},
            )
            self._pipelines[pkey] = pipeline

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind)
        for _ in range(max(1, int(repeat))):
            cp.dispatch_workgroups(int(dispatch[0]), int(dispatch[1]))
        cp.end()
        self.device.queue.submit([enc.finish()])

        if not readback:
            # ⚠️ 池模式 + 不回读 = **只测时序，结果无意义**：
            #    缓冲会被下一个 pass 复用，链上的数据其实是断的。
            #    这正是要量的东西（dispatch + 计算的开销），但别拿这个模式跑验收。
            for b, nb, us in pooled:
                self.recycle(b, nb, us)
            return None

        flat = self.read_buffer(outbuf, (out_count,), out_dtype)
        for b, nb, us in pooled:
            self.recycle(b, nb, us)
        if out_shape_tail is not None:
            return flat.reshape(out_shape_tail)
        return flat


# ══════════════════════════════════════════════════════════════════
# 比对与报告
# ══════════════════════════════════════════════════════════════════
def compare(name: str, got: np.ndarray, want: np.ndarray, tol: float,
            exact: bool = False) -> bool:
    """比对 GPU 与 numpy 的输出，打印误差分布。

    ``exact=True`` 时要求逐位相同 —— 用于纯比较/纯查表的阶段，
    那些地方的差异一定是逻辑问题，不是精度问题。
    """
    g = np.asarray(got, dtype=np.float64).ravel()
    w = np.asarray(want, dtype=np.float64).ravel()
    if g.shape != w.shape:
        print(f"  ❌ {name}: 形状不一致 {g.shape} vs {w.shape}")
        return False

    d = np.abs(g - w)
    mx = float(d.max()) if d.size else 0.0
    mean = float(d.mean()) if d.size else 0.0
    n_bad = int((d > tol).sum())
    p999 = float(np.percentile(d, 99.9)) if d.size else 0.0

    print(f"  {'✅' if (mx == 0 if exact else n_bad == 0) else '❌'} {name}")
    print(f"       最大差 {mx:.3e}   均值 {mean:.3e}   P99.9 {p999:.3e}   "
          f"超限 {n_bad}/{d.size} ({100.0 * n_bad / max(1, d.size):.4f}%)")

    if exact:
        ok = mx == 0.0
        if not ok:
            idx = int(np.argmax(d))
            print(f"       ⚠️ 要求逐位相同，但最大差 {mx:.3e}（位置 {idx}，"
                  f"GPU={g[idx]!r} vs numpy={w[idx]!r}）")
        return ok

    ok = n_bad == 0
    if not ok:
        idx = np.argsort(d)[-5:][::-1]
        print("       最差的位置：")
        for i in idx[:5]:
            print(f"         [{i}] GPU={g[i]:.6f}  numpy={w[i]:.6f}  差 {d[i]:.3e}")
    return ok


# ══════════════════════════════════════════════════════════════════
# 各阶段的验收
# ══════════════════════════════════════════════════════════════════
def _check_fog(lab: WgslLab, w: int, h: int) -> bool:
    """验收 depth_fog 的 WGSL 移植。"""
    from pixelart.compose import depth_fog
    from pixelart.webgpu import fog_uniforms

    base = _test_image(w, h)
    far = _test_depth(w, h)
    fog_color = np.array([0.42, 0.55, 0.63], np.float32)
    density, power, drift, t = 0.8, 3.0, 0.35, 0.37

    u = fog_uniforms(w, h, t, fog_color, density, power,
                     floor=0.0, ceiling=1.0, drift=drift)
    assert len(u) % 4 == 0

    got = lab.run(_load_wgsl("fog"), "main", (w, h),
                  inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                  uniforms=u, out_count=w * h * 3,
                  out_shape_tail=(h, w, 3), label="fog")

    # numpy 参考：同一组输入、同一组参数
    want = depth_fog(base, far, color=fog_color, density=density, power=power,
                     t=t, drift=drift)

    ok = compare("depth_fog（含漂移 t=0.37）", got, want, tol=2e-3)

    # 再验一次 t=0（静态图路径，drift 关掉）
    u0 = fog_uniforms(w, h, 0.0, fog_color, density, power, drift=0.0)
    got0 = lab.run(_load_wgsl("fog"), "main", (w, h),
                   inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                   uniforms=u0, out_count=w * h * 3,
                   out_shape_tail=(h, w, 3), label="fog")
    want0 = depth_fog(base, far, color=fog_color, density=density, power=power,
                      t=0.0, drift=0.0)
    ok = compare("depth_fog（静态 drift=0）", got0, want0, tol=2e-3) and ok

    # 循环性：t=0 与 t=1 必须**逐位相同**（时序铁律 3 在 GPU 侧也要成立）。
    # ⚠️ 基线必须单独渲一个 t=0.0，不能拿上面 t=0.37 的那次来比 ——
    #    第一版就是这么写错的，两个不同 t 的噪声当然不同，白白误报了一轮。
    u_zero = fog_uniforms(w, h, 0.0, fog_color, density, power, drift=drift)
    got_zero = lab.run(_load_wgsl("fog"), "main", (w, h),
                       inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                       uniforms=u_zero, out_count=w * h * 3,
                       out_shape_tail=(h, w, 3), label="fog")
    u_one = fog_uniforms(w, h, 1.0, fog_color, density, power, drift=drift)
    got_one = lab.run(_load_wgsl("fog"), "main", (w, h),
                      inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                      uniforms=u_one, out_count=w * h * 3,
                      out_shape_tail=(h, w, 3), label="fog")
    ok = compare("循环性 frame(t=1) == frame(t=0)", got_one, got_zero,
                 0.0, exact=True) and ok

    # 漂移确实在动（否则上面那条"逐位相同"可能是因为漂移根本没生效）
    got_mid = lab.run(_load_wgsl("fog"), "main", (w, h),
                      inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                      uniforms=fog_uniforms(w, h, 0.5, fog_color, density, power,
                                            drift=drift),
                      out_count=w * h * 3, out_shape_tail=(h, w, 3), label="fog")
    moved = float(np.abs(got_mid - got_zero).max())
    print(f"  {'✅' if moved > 1e-4 else '❌'} 漂移确实生效（t=0.5 vs t=0 最大差 {moved:.3e}）")
    ok = (moved > 1e-4) and ok

    # ── fog_tint（M5.3）：去色分支（tint<0.999）必须真的被执行 ──
    #   ⚠️ 上面所有用例都没传 fog_tint（默认 1.0）→ WGSL 守卫直接跳过新分支，
    #      那是"空转检验"。这里专门打 tint=0 / 0.3，两侧同参数。
    got_ft0 = None
    for ft in (0.0, 0.3):
        u_ft = fog_uniforms(w, h, t, fog_color, density, power,
                            floor=0.0, ceiling=1.0, drift=drift, fog_tint=ft)
        got_ft = lab.run(_load_wgsl("fog"), "main", (w, h),
                         inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                         uniforms=u_ft, out_count=w * h * 3,
                         out_shape_tail=(h, w, 3), label="fog")
        want_ft = depth_fog(base, far, color=fog_color, density=density,
                            power=power, t=t, drift=drift, fog_tint=ft)
        ok = compare(f"depth_fog（fog_tint={ft}，去色分支）", got_ft, want_ft,
                     tol=2e-3) and ok
        if ft == 0.0:
            got_ft0 = got_ft

    # 非空转：tint=0 与 tint=1 的 GPU 输出必须真的不同（这条会抓住
    # "shader 读错 uniform 槽位"——读错时两支看到同一个值 → 差≈0）。
    u_t1 = fog_uniforms(w, h, t, fog_color, density, power,
                        floor=0.0, ceiling=1.0, drift=drift, fog_tint=1.0)
    got_t1 = lab.run(_load_wgsl("fog"), "main", (w, h),
                     inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                     uniforms=u_t1, out_count=w * h * 3,
                     out_shape_tail=(h, w, 3), label="fog")
    dtint = float(np.abs(got_ft0 - got_t1).max())
    print(f"  {'✅' if dtint > 1e-3 else '❌'} fog 去色分支确实生效（tint=0 vs 1 最大差 {dtint:.3e}）")
    ok = (dtint > 1e-3) and ok
    return ok


def _check_tail(lab: WgslLab, w: int, h: int) -> bool:
    """验收像素尾巴的 WGSL 移植。

    ⚠️ 这个阶段要求**逐位一致**。它全程只有比较、查表和四则运算，
    没有任何超越函数 —— 所以任何差异都是逻辑错误，不是精度问题。
    """
    from pixelart.pixelate import PixelTail, pixelate

    composed = _test_image(w, h, seed=11)
    # edge_ref 刻意与 composed 不同：风/光会变，但"内容参考"是另一张图。
    # 用同一张的话，边缘压暗这条路径实际上就没被测到。
    edge_ref = _test_image(w, h, seed=12)
    pal = _test_palette(32)
    n = len(pal)

    tail = PixelTail()          # 默认值：dither=0.10 自适应、edge_strength=0.35、edge_gain=2.0
    u = np.zeros((2, 4), dtype=np.float32)
    u[0] = [float(n), tail.dither, 1.0 if tail.dither_adaptive else 0.0, tail.edge_gain]
    u[1] = [tail.edge_strength, float(w), float(h), 0.0]

    got = lab.run(_load_wgsl("tail"), "main", (w, h),
                  inputs={
                      "src": composed.reshape(-1).astype(np.float32),
                      "edge_ref": edge_ref.reshape(-1).astype(np.float32),
                      "pal": pal.reshape(-1).astype(np.float32),
                  },
                  uniforms=u, out_count=w * h, out_dtype=np.uint32, label="tail")

    want, _ = pixelate(composed, tail, palette=pal, edge_ref=edge_ref)

    # 解包 u32 → (H,W,4)，与 numpy 的 u8 比 RGB
    pk = np.asarray(got, dtype=np.uint32).reshape(h, w)
    got_rgb = np.stack([(pk & 0xFF),
                        ((pk >> 8) & 0xFF),
                        ((pk >> 16) & 0xFF)], -1).astype(np.int32)
    want_rgb = want.astype(np.int32)

    diff = np.abs(got_rgb - want_rgb)
    n_bad = int((diff.max(axis=2) > 0).sum())
    ok = n_bad == 0
    print(f"  {'✅' if ok else '❌'} 像素尾巴（吸附+抖动+边缘）")
    print(f"       逐位不一致的像素 {n_bad}/{w * h} ({100.0 * n_bad / (w * h):.4f}%)  "
          f"最大通道差 {int(diff.max())}")

    # ── 连续色模式（M5.10）：跳过吸附，抖动/边缘照常 ──
    #   同一套 dither/edge 公式，只少两处 snap —— 逐位一致应保持。
    from dataclasses import replace as _dc_replace
    tail_c = _dc_replace(tail, quantize_mode="continuous")
    u_c = u.copy()
    u_c[1][3] = 1.0
    got_c = lab.run(_load_wgsl("tail"), "main", (w, h),
                    inputs={
                        "src": composed.reshape(-1).astype(np.float32),
                        "edge_ref": edge_ref.reshape(-1).astype(np.float32),
                        "pal": pal.reshape(-1).astype(np.float32),
                    },
                    uniforms=u_c, out_count=w * h, out_dtype=np.uint32, label="tail_cont")
    want_c, _ = pixelate(composed, tail_c, palette=pal, edge_ref=edge_ref)
    pk_c = np.asarray(got_c, dtype=np.uint32).reshape(h, w)
    got_c_rgb = np.stack([(pk_c & 0xFF), ((pk_c >> 8) & 0xFF),
                          ((pk_c >> 16) & 0xFF)], -1).astype(np.int32)
    d_c = np.abs(got_c_rgb - want_c.astype(np.int32))
    bad_c = int((d_c.max(axis=2) > 0).sum())
    print(f"  {'✅' if bad_c == 0 else '❌'} 像素尾巴（连续色模式：跳过吸附）")
    print(f"       逐位不一致的像素 {bad_c}/{w * h}")
    ok = (bad_c == 0) and ok

    if not ok:
        ys, xs = np.where(diff.max(axis=2) > 0)
        print("       最差的位置：")
        order = np.argsort(-diff.max(axis=2)[ys, xs])
        for t in order[:8]:
            y, x = int(ys[t]), int(xs[t])
            print(f"         ({y:3d},{x:3d}) GPU={got_rgb[y, x].tolist()} "
                  f"numpy={want_rgb[y, x].tolist()}")

        # 逐项定位：关掉抖动、关掉边缘，看差异是否消失 —— 用来定位是哪一项错的
        for label, tw in (("关抖动", PixelTail(dither=0.0)),
                          ("关边缘", PixelTail(edge_strength=0.0)),
                          ("关抖动+关边缘", PixelTail(dither=0.0, edge_strength=0.0))):
            u2 = np.zeros((2, 4), dtype=np.float32)
            u2[0] = [float(n), tw.dither, 1.0 if tw.dither_adaptive else 0.0, tw.edge_gain]
            u2[1] = [tw.edge_strength, float(w), float(h), 0.0]
            g2 = lab.run(_load_wgsl("tail"), "main", (w, h),
                         inputs={"src": composed.reshape(-1).astype(np.float32),
                                 "edge_ref": edge_ref.reshape(-1).astype(np.float32),
                                 "pal": pal.reshape(-1).astype(np.float32)},
                         uniforms=u2, out_count=w * h, out_dtype=np.uint32, label="tail")
            w2, _ = pixelate(composed, tw, palette=pal, edge_ref=edge_ref)
            p2 = np.asarray(g2, dtype=np.uint32).reshape(h, w)
            r2 = np.stack([(p2 & 0xFF), ((p2 >> 8) & 0xFF), ((p2 >> 16) & 0xFF)],
                          -1).astype(np.int32)
            nb = int((np.abs(r2 - w2.astype(np.int32)).max(axis=2) > 0).sum())
            print(f"         · {label:14s} 不一致 {nb}/{w * h}"
                  f"{'   ← 差异来自这里' if nb == 0 else ''}")

    return ok


def _check_volumetric(lab: WgslLab, w: int, h: int) -> bool:
    """验收体积光（含散射色采样）的 WGSL 移植。"""
    from pixelart.compose import auto_scatter_color, limit_saturation, volumetric_light
    from pixelart.webgpu import scatter_uniforms, volumetric_uniforms

    # 用雾后的图当输入（体积光就是接在雾后面的）
    fogged = _test_image(w, h, seed=11)
    far = _test_depth(w, h)

    lx_n, ly_n = 0.62, 0.28
    lx = int(max(0.0, min(1.0, lx_n)) * (w - 1))
    ly = int(max(0.0, min(1.0, ly_n)) * (h - 1))
    occlude_gain, falloff_gain, screen_falloff = 5.0, 2.5, 1.0
    strength, flicker = 0.55, 0.35
    t = 0.41
    from pixelart.animate import flicker_gain
    gain = flicker_gain(t, depth=flicker)

    # ── ① 散射色 pass ──
    su = scatter_uniforms(w, h, lx, ly, radius=3, sat_max=0.25)
    air_gpu = lab.run(_load_wgsl("scatter"), "main", (1, 1),
                      inputs={"src": fogged.reshape(-1)},
                      uniforms=su, out_count=3, label="scatter")
    air_ref = auto_scatter_color(fogged, lx, ly, radius=3, sat_max=0.25)
    ok = compare("散射色（7×7 中位数 + 饱和上限）", air_gpu, air_ref,
                 1e-6, exact=False)

    # ── ② 体积光 pass ──
    vu = volumetric_uniforms(w, h, samples=28, span=0.85, decay=0.965,
                             strength=strength, occlude_gain=occlude_gain,
                             falloff_gain=falloff_gain, screen_falloff=screen_falloff,
                             threshold=0.48, knee=0.3, gain=gain, lx=lx, ly=ly)
    got = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                  inputs={"fogged": fogged.reshape(-1),
                          "far": far.reshape(-1),
                          "air": np.asarray(air_ref, np.float32)},
                  uniforms=vu, out_count=w * h * 3,
                  out_shape_tail=(h, w, 3), label="volumetric")

    # ⚠️ 必须把**每一个**参数都传给参考实现。
    #    第一版漏了 strength —— numpy 用默认 0.85、GPU 用 0.55，
    #    于是"最大差 2.2e-3"被当成移植误差，白追了一轮。
    #    **测试代码自己也是代码，也会错。**
    vol = volumetric_light(fogged, far, (lx_n, ly_n),
                           samples=28, span=0.85, decay=0.965,
                           strength=strength,
                           occlude_gain=occlude_gain, falloff_gain=falloff_gain,
                           air_sat_max=0.25, t=t, flicker=flicker,
                           screen_falloff=screen_falloff)
    want = np.clip(fogged + vol, 0.0, 1.0)
    ok = compare("体积光（28 次深度感知采样 + 遮挡 + 屏幕衰减）", got, want,
                 tol=1e-5) and ok

    # ── ③ 光锥（M5.1）：张角 + 方向 + 长度，多组参数对照 ──
    #   ⚠️ 锥权重在 CPU 与 GPU 各写了一遍（compose.cone_weight / volumetric.wgsl），
    #      这里就是钉死它们不漂移的地方。角度用**度**传，弧度转换在两侧各自的
    #      packer 里（uniform 逐字节比对另有一条守门）。
    for cone_angle, cone_dir, cone_reach, shaft in ((0.7, 80.0, 0.8, 1.2),
                                                    (0.35, -120.0, 0.4, 0.0),
                                                    (1.1, 90.0, 1.0, 2.0)):
        vu_c = volumetric_uniforms(
            w, h, samples=28, span=0.85, decay=0.965, strength=strength,
            occlude_gain=occlude_gain, falloff_gain=falloff_gain,
            screen_falloff=screen_falloff, threshold=0.48, knee=0.3,
            gain=gain, lx=lx, ly=ly, cone_angle=cone_angle,
            cone_dir_deg=cone_dir, cone_reach=cone_reach, shaft=shaft)
        got_c = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                        inputs={"fogged": fogged.reshape(-1),
                                "far": far.reshape(-1),
                                "air": np.asarray(air_ref, np.float32)},
                        uniforms=vu_c, out_count=w * h * 3,
                        out_shape_tail=(h, w, 3), label="volumetric_cone")
        vol_c = volumetric_light(fogged, far, (lx_n, ly_n),
                                 samples=28, span=0.85, decay=0.965,
                                 strength=strength, occlude_gain=occlude_gain,
                                 falloff_gain=falloff_gain, air_sat_max=0.25,
                                 t=t, flicker=flicker,
                                 screen_falloff=screen_falloff,
                                 cone_angle=cone_angle, cone_dir_deg=cone_dir,
                                 cone_reach=cone_reach, shaft=shaft)
        want_c = np.clip(fogged + vol_c, 0.0, 1.0)
        ok = compare(f"光锥+光柱（张角 {cone_angle} 方向 {cone_dir}° "
                     f"长度 {cone_reach} 光柱 {shaft}）",
                     got_c, want_c, tol=1e-5) and ok

    # ── 锥顶点解耦（M5.4）：顶点 ≠ 光源 ──
    #   numpy 侧传 cone_xy（归一化）；GPU 侧 packer 传 cone_x/y（<0=跟随）。
    #   顶点(0.12,0.85) 与光源(0.62,0.28) 明显不同 → 若 GPU 忽略顶点槽位
    #   （读成光源），与本用例的差会远超 1e-5 —— 这就是非空转保证。
    vu_d = volumetric_uniforms(
        w, h, samples=28, span=0.85, decay=0.965, strength=strength,
        occlude_gain=occlude_gain, falloff_gain=falloff_gain,
        screen_falloff=screen_falloff, threshold=0.48, knee=0.3,
        gain=gain, lx=lx, ly=ly, cone_angle=0.7, cone_dir_deg=-30.0,
        cone_reach=0.9, shaft=1.2, cone_x=0.12, cone_y=0.85)
    got_d = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                    inputs={"fogged": fogged.reshape(-1),
                            "far": far.reshape(-1),
                            "air": np.asarray(air_ref, np.float32)},
                    uniforms=vu_d, out_count=w * h * 3,
                    out_shape_tail=(h, w, 3), label="volumetric_cone")
    vol_d = volumetric_light(fogged, far, (lx_n, ly_n),
                             samples=28, span=0.85, decay=0.965,
                             strength=strength, occlude_gain=occlude_gain,
                             falloff_gain=falloff_gain, air_sat_max=0.25,
                             t=t, flicker=flicker, screen_falloff=screen_falloff,
                             cone_angle=0.7, cone_dir_deg=-30.0,
                             cone_reach=0.9, shaft=1.2, cone_xy=(0.12, 0.85))
    want_d = np.clip(fogged + vol_d, 0.0, 1.0)
    ok = compare("锥顶点解耦（顶点≠光源，出画锚点）", got_d, want_d, tol=1e-5) and ok

    # 非空转：关掉体积光，画面必须真的不同
    vu0 = volumetric_uniforms(w, h, samples=28, span=0.85, decay=0.965,
                              strength=0.0, occlude_gain=occlude_gain,
                              falloff_gain=falloff_gain,
                              screen_falloff=screen_falloff,
                              threshold=0.48, knee=0.3, gain=gain, lx=lx, ly=ly)
    off = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                  inputs={"fogged": fogged.reshape(-1), "far": far.reshape(-1),
                          "air": np.asarray(air_ref, np.float32)},
                  uniforms=vu0, out_count=w * h * 3,
                  out_shape_tail=(h, w, 3), label="volumetric")
    moved = float(np.abs(got - off).max())
    print(f"  {'✅' if moved > 1e-3 else '❌'} 体积光确实生效（strength=0 时最大差 {moved:.3e}）")
    ok = (moved > 1e-3) and ok

    # ── fog_tint（M5.3）：散射色去色分支两侧一致 + 非空转 ──
    #   ⚠️ 槽位历史：packer 写 U[5].y，shader 第一版读 U[5].z（恒 0）→
    #      GPU 永远全去色而 uniform 探针全绿。这条扫描就是它的守门。
    got_ft0 = None
    for ft in (0.0, 0.3):
        vu_ft = volumetric_uniforms(
            w, h, samples=28, span=0.85, decay=0.965, strength=strength,
            occlude_gain=occlude_gain, falloff_gain=falloff_gain,
            screen_falloff=screen_falloff, threshold=0.48, knee=0.3,
            gain=gain, lx=lx, ly=ly, fog_tint=ft)
        got_ft = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                         inputs={"fogged": fogged.reshape(-1),
                                 "far": far.reshape(-1),
                                 "air": np.asarray(air_ref, np.float32)},
                         uniforms=vu_ft, out_count=w * h * 3,
                         out_shape_tail=(h, w, 3), label="volumetric")
        vol_ft = volumetric_light(fogged, far, (lx_n, ly_n),
                                  samples=28, span=0.85, decay=0.965,
                                  strength=strength, occlude_gain=occlude_gain,
                                  falloff_gain=falloff_gain, air_sat_max=0.25,
                                  t=t, flicker=flicker,
                                  screen_falloff=screen_falloff, fog_tint=ft)
        want_ft = np.clip(fogged + vol_ft, 0.0, 1.0)
        ok = compare(f"体积光（fog_tint={ft}，散射色去色）", got_ft, want_ft,
                     tol=1e-5) and ok
        if ft == 0.0:
            got_ft0 = got_ft

    # 非空转判据要按夹具的**物理上限**校准，不能拍脑袋：
    #   本夹具里体积光对画面的总贡献只有 ~4.1e-3（上面 strength=0 那条量过），
    #   去色只改这个小贡献的色相 → 实测效应 ~7.4e-4，噪声地板 ~6e-8。
    #   阈值取 1e-4：比噪声高 3 个量级，又低于实际效应；槽位读错时差=0 必抓。
    dtint = float(np.abs(got_ft0 - got).max())
    print(f"  {'✅' if dtint > 1e-4 else '❌'} 散射色去色分支确实生效（tint=0 vs 1 最大差 {dtint:.3e}）")
    ok = (dtint > 1e-4) and ok
    return ok


def _check_chain(lab: WgslLab, w: int, h: int) -> bool:
    """端到端链路验收：fog → 体积光 → 尘埃 → 辉光 → 像素尾巴。

    ⭐ 这是**真正的验收标准**。单阶段各自一致还不够 ——
    误差会累积，而最终产物是一个 32 色的 uint8 图，
    中间任何偏差都可能让某个像素吸附到别的颜色上。

    这里同时回答两个问题：

      A. **GPU 链路 == 浮点参考链路？** 判据：最终 u8 **逐位一致**。
      B. **GPU 链路 vs 现在的服务端（PIL 辉光）差多少？** 只报数、不做硬判据 ——
         因为 GPU 侧的鲜光是真高斯、服务端是 PIL 的三次盒式近似，
         两者**本来就不会逐位相同**（见 compose.blur_float 的说明）。
         但这个数正是"用户换成浏览器渲染后会不会觉得变了"的答案，
         所以必须量出来、并且盯住它。
    """
    from pixelart.compose import (
        auto_fog_color, bloom, bloom_float, depth_fog, limit_saturation,
        volumetric_light,
    )
    from pixelart.animate import dust_layer, flicker_gain
    from pixelart.pixelate import PixelTail, pixelate
    from pixelart.webgpu import (
        bloom_blur_uniforms, bloom_bright_uniforms, bloom_combine_uniforms,
        bloom_kernels, dust_apply_uniforms, dust_fixed_point_scale,
        dust_particle_params, dust_splat_uniforms, fog_uniforms,
        scatter_uniforms, volumetric_uniforms,
    )

    base = _test_image(w, h, seed=21)
    far = _test_depth(w, h)
    pal = _test_palette(32)
    tail = PixelTail()

    fog_color = limit_saturation(np.array([0.55, 0.68, 0.58], np.float32), 0.40)
    density, power, drift = 0.8, 3.0, 0.35
    screen_falloff, strength, flicker = 1.0, 0.55, 0.35
    t = 0.37
    gain = flicker_gain(t, depth=flicker)

    from pixelart.compose import brightest_center
    lxn, lyn = brightest_center(base, blur_radius=3.0)
    lx, ly = int(lxn * (w - 1)), int(lyn * (h - 1))

    # 尘埃 / 辉光参数（与 compose_frame 的调用一致）
    dust_count, dust_seed = 200, 11
    dust_bright, dust_twinkle, dust_fade, dust_boost = 0.55, 0.55, 0.65, 1.8
    bloom_strength = 0.85 * 0.5
    radii = (2.0, 5.0, 11.0)
    kflat, table = bloom_kernels(radii)

    n = w * h * 3
    npix = w * h
    zero = np.zeros(n, dtype=np.float32)

    # ══════════ GPU 链路 ══════════
    fogged_g = lab.run(_load_wgsl("fog"), "main", (w, h),
                       inputs={"src": base.reshape(-1), "far": far.reshape(-1)},
                       uniforms=fog_uniforms(w, h, t, fog_color, density, power,
                                             drift=drift),
                       out_count=n, out_shape_tail=(h, w, 3), label="fog")

    air_g = lab.run(_load_wgsl("scatter"), "main", (1, 1),
                    inputs={"src": fogged_g.reshape(-1)},
                    uniforms=scatter_uniforms(w, h, lx, ly, 3, 0.25),
                    out_count=3, label="scatter")

    lit_g = lab.run(_load_wgsl("volumetric"), "main", (w, h),
                    inputs={"fogged": fogged_g.reshape(-1),
                            "far": far.reshape(-1),
                            "air": np.asarray(air_g, np.float32)},
                    uniforms=volumetric_uniforms(w, h, samples=28, span=0.85,
                                                 decay=0.965, strength=strength,
                                                 occlude_gain=5.0, falloff_gain=2.5,
                                                 screen_falloff=screen_falloff,
                                                 threshold=0.48, knee=0.3,
                                                 gain=gain, lx=lx, ly=ly),
                    out_count=n, out_shape_tail=(h, w, 3), label="volumetric")
    content_g = lit_g.copy()                     # edge_ref：**不含粒子**

    par = dust_particle_params(dust_count, seed=dust_seed)
    dscale = dust_fixed_point_scale(dust_count)
    bins = lab.run(_load_wgsl("dust_splat"), "main", None,
                   inputs={"par": par.reshape(-1), "far": far.reshape(-1)},
                   uniforms=dust_splat_uniforms(
                       w, h, dust_count, t, dust_twinkle, dust_fade,
                       dust_boost, (lxn, lyn), dscale, True),
                   out_count=npix, out_dtype=np.uint32, label="dust_splat",
                   workgroup=(64, 1),
                   dispatch=((dust_count + 63) // 64, 1))
    dusted_g = lab.run(_load_wgsl("dust_apply"), "main", (w, h),
                       inputs={"bin": bins, "lit": lit_g.reshape(-1)},
                       uniforms=dust_apply_uniforms(w, h, dscale, dust_bright),
                       out_count=n, out_shape_tail=(h, w, 3), label="dust_apply")

    bright_g = lab.run(_load_wgsl("bloom_bright"), "main", (w, h),
                       inputs={"rgb": dusted_g.reshape(-1)},
                       uniforms=bloom_bright_uniforms(w, h, 0.55, 0.25),
                       out_count=n, out_shape_tail=(h, w, 3),
                       label="bloom_bright")
    tmp = zero.copy()
    acc = zero.copy()
    for entry in table:
        tmp = lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                      inputs={"kbuf": kflat, "src": bright_g.reshape(-1),
                              "prev": zero},
                      uniforms=bloom_blur_uniforms(w, h, "h", entry, False),
                      out_count=n, out_shape_tail=(h, w, 3), label="bloom_blur_h")
        acc = lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                      inputs={"kbuf": kflat, "src": tmp.reshape(-1),
                              "prev": acc.reshape(-1)},
                      uniforms=bloom_blur_uniforms(w, h, "v", entry, True),
                      out_count=n, out_shape_tail=(h, w, 3), label="bloom_blur_v")
    composed_g = lab.run(_load_wgsl("bloom_combine"), "main", (w, h),
                         inputs={"rgb": dusted_g.reshape(-1), "acc": acc.reshape(-1)},
                         uniforms=bloom_combine_uniforms(w, h, bloom_strength, radii),
                         out_count=n, out_shape_tail=(h, w, 3), label="bloom_combine")

    packed = lab.run(_load_wgsl("tail"), "main", (w, h),
                     inputs={"src": composed_g.reshape(-1),
                             "edge_ref": content_g.reshape(-1),
                             "pal": pal.reshape(-1)},
                     uniforms=_tail_uniforms(w, h, pal, tail),
                     out_count=npix, out_dtype=np.uint32, label="tail")
    pk = np.asarray(packed, np.uint32).reshape(h, w)
    got = np.stack([pk & 0xFF, (pk >> 8) & 0xFF, (pk >> 16) & 0xFF], -1).astype(np.int32)

    # ══════════ 参考链路（全浮点，与 GPU 同公式）══════════
    fogged_n = depth_fog(base, far, color=fog_color, density=density, power=power,
                         t=t, drift=drift)
    lit_n = np.clip(fogged_n + volumetric_light(
        fogged_n, far, (lxn, lyn), samples=28, span=0.85, decay=0.965,
        strength=strength, occlude_gain=5.0, falloff_gain=2.5,
        air_sat_max=0.25, t=t, flicker=flicker,
        screen_falloff=screen_falloff), 0.0, 1.0)
    content_n = lit_n.copy()
    lit_vol_n = lit_n.copy()             # ⚠️ 快照：下面 lit_n 会被尘埃那步覆盖
    dn = dust_layer((h, w), far, t=t, count=dust_count, seed=dust_seed,
                    light_xy=(lxn, lyn), light_boost=dust_boost,
                    twinkle=dust_twinkle, fade_far=dust_fade)
    lit_n = np.clip(lit_n + dn * dust_bright, 0.0, 1.0)
    lit_dust_n = lit_n.copy()            # ⚠️ 同样快照，给辉光那一步用
    composed_n = bloom_float(lit_n, 0.55, bloom_strength, radii, knee=0.25)
    want, _ = pixelate(composed_n, tail, palette=pal, edge_ref=content_n)
    want = want.astype(np.int32)

    # ⚠️ 每个中间量都要跟**同一阶段的**参考比。
    #    第一版忘了快照，`lit_n` 被尘埃那步覆盖之后，
    #    打印出来的"体积光后"其实是在比"体积光 vs 尘埃后"，差了 4.77e-1 ——
    #    一个纯粹的**报告错误**，看起来却像严重的移植问题。
    #    （与约束 20 里那三次是同一类：先坏在测量，而不是坏在被测的东西。）
    print("  中间量（GPU vs 纯浮点参考）：")
    for label, g, wref in (("雾后", fogged_g, fogged_n),
                           ("体积光后", lit_g, lit_vol_n),
                           ("尘埃后", dusted_g, lit_dust_n),
                           ("辉光后", composed_g, composed_n)):
        print(f"    {label:8s} 最大差 {float(np.abs(g - wref).max()):.3e}")
    print("  最终输出：")

    d = np.abs(got - want)
    n_bad = int((d.max(axis=2) > 0).sum())
    ok = n_bad == 0
    print(f"  {'✅' if ok else '❌'} A. GPU 链路 == 纯浮点参考链路（逐位）")
    print(f"       逐位不一致 {n_bad}/{w * h} ({100.0 * n_bad / (w * h):.4f}%)  "
          f"最大通道差 {int(d.max())}")
    if not ok:
        ys, xs = np.where(d.max(axis=2) > 0)
        for t2 in np.argsort(-d.max(axis=2)[ys, xs])[:6]:
            y, x = int(ys[t2]), int(xs[t2])
            print(f"         ({y:3d},{x:3d}) GPU={got[y, x].tolist()} "
                  f"numpy={want[y, x].tolist()}")

    # ══════════ B. 与**现在的服务端**（PIL 辉光）比 ══════════
    composed_prod = bloom(lit_n, 0.55, bloom_strength, radii)
    prod, _ = pixelate(composed_prod, tail, palette=pal, edge_ref=content_n)
    prod = prod.astype(np.int32)
    dp = np.abs(got - prod)
    n_prod = int((dp.max(axis=2) > 0).sum())
    share = 100.0 * n_prod / (w * h)
    print(f"  {'✅' if share < 2.0 else '⚠️'} B. 与现在的服务端（PIL 辉光）相比")
    print(f"       不一致 {n_prod}/{w * h} ({share:.2f}%)  "
          f"最大通道差 {int(dp.max())}")
    verdict = "画面差异可忽略" if share < 2.0 else "需要复核"
    print(f"       → 换成浏览器渲染后：{verdict}"
          f"（这就是用户会不会觉得「画面变了」的答案）")
    return ok


def _tail_uniforms(w, h, pal, tail) -> np.ndarray:
    u = np.zeros((2, 4), dtype=np.float32)
    u[0] = [float(len(pal)), tail.dither,
            1.0 if tail.dither_adaptive else 0.0, tail.edge_gain]
    u[1] = [tail.edge_strength, float(w), float(h), 0.0]
    return u.reshape(-1)


def _check_bloom(lab: WgslLab, w: int, h: int) -> bool:
    """验收辉光（高光提取 + 三尺度可分离高斯 + 合并）。

    ⚠️ 判据用**容差**，与 fog/volumetric 同类。原因是这个阶段本质上是个模糊：
    核系数已由 CPU 上传（两边同一份），剩下的差异只有 f32 累积，
    但抽头很多（13+31+67=111 个），所以容差给到 1e-5。

    另外它**不追求与 PIL 版逐位** —— PIL 的 GaussianBlur 是三次盒式近似、
    且中间降到 uint8，逐位复现既不现实也不值得。
    实测两者经过完整链 + 色板吸附后只差 0.21% 的像素
    （``tools/blur_operator_probe.py``），所以这个偏离是可接受的、有量的。
    """
    from pixelart.compose import bloom_float
    from pixelart.webgpu import (
        bloom_blur_uniforms, bloom_bright_uniforms, bloom_combine_uniforms,
        bloom_kernels,
    )

    rgb = _test_image(w, h, seed=31)
    from pixelart.compose import BLOOM_KNEE, BLOOM_RADII, BLOOM_THRESHOLD
    radii = tuple(float(v) for v in BLOOM_RADII)
    threshold, knee, strength = BLOOM_THRESHOLD, BLOOM_KNEE, 0.85 * 0.5

    kflat, table = bloom_kernels(radii)
    n = w * h * 3
    zero = np.zeros(n, dtype=np.float32)

    # ── pass 1 高光 ──
    bright = lab.run(_load_wgsl("bloom_bright"), "main", (w, h),
                     inputs={"rgb": rgb.reshape(-1)},
                     uniforms=bloom_bright_uniforms(w, h, threshold, knee),
                     out_count=n, out_shape_tail=(h, w, 3), label="bloom_bright")

    # ── pass 2 三个尺度：横→纵，累积到显式的 prev 里 ──
    # ⚠️ 累积必须走**显式输入**，不能让着色器把 dst 当累积器读回来。
    #    本验收台每次 lab.run 都新建清零的输出缓冲，那种写法会静默失效
    #    （三个尺度加权求和退化成只留最后一个，实测差 8.4e-2）。
    tmp = zero.copy()
    acc = zero.copy()
    for entry in table:
        tmp = lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                      inputs={"kbuf": kflat, "src": bright.reshape(-1),
                              "prev": zero},
                      uniforms=bloom_blur_uniforms(w, h, "h", entry, False),
                      out_count=n, out_shape_tail=(h, w, 3), label="bloom_blur_h")
        acc = lab.run(_load_wgsl("bloom_blur"), "main", (w, h),
                      inputs={"kbuf": kflat, "src": tmp.reshape(-1),
                              "prev": acc.reshape(-1)},
                      uniforms=bloom_blur_uniforms(w, h, "v", entry, True),
                      out_count=n, out_shape_tail=(h, w, 3),
                      label="bloom_blur_v")

    # ── pass 3 合并 ──
    got = lab.run(_load_wgsl("bloom_combine"), "main", (w, h),
                  inputs={"rgb": rgb.reshape(-1), "acc": acc.reshape(-1)},
                  uniforms=bloom_combine_uniforms(w, h, strength, radii),
                  out_count=n, out_shape_tail=(h, w, 3), label="bloom_combine")

    want = bloom_float(rgb, threshold, strength, radii, knee=knee)
    ok = compare("辉光（高光 + 三尺度可分离高斯 + 合并）", got, want, tol=1e-5)

    # 中间量：高光提取必须与 numpy 一致
    from pixelart.compose import bright_pass
    want_b = bright_pass(rgb, threshold, knee=knee)
    # ⚠️ 容差不能定成 1e-7：值 ~0.3 时**一个 f32 ulp 就有 1.5e-7**，
    #    那不是误差，那就是 f32 的分辨率。定 1e-6。
    ok = compare("  · 高光提取", bright, want_b, tol=1e-6) and ok

    # 非空转：strength=0 时输出必须等于原图（说明确实在"加"东西）
    zero_acc = lab.run(_load_wgsl("bloom_combine"), "main", (w, h),
                       inputs={"rgb": rgb.reshape(-1), "acc": zero},
                       uniforms=bloom_combine_uniforms(w, h, 0.0, radii),
                       out_count=n, out_shape_tail=(h, w, 3),
                       label="bloom_combine")
    d = float(np.abs(zero_acc - rgb).max())
    print(f"  {'✅' if d < 1e-7 else '❌'} 非空转：strength=0 时输出 == 原图（差 {d:.3e}）")
    ok = (d < 1e-7) and ok
    # ⚠️ 这里要量 **got − rgb**（真的加了东西），不是 got − want（那是移植差异）。
    #    第一版量成了后者，于是"辉光幅度"和"最大差"是同一个数 ——
    #    两个含义完全不同的判据共用一个数字，会掩盖真问题。
    glow = float(np.abs(got - rgb).max())
    print(f"  {'✅' if glow > 1e-4 else '❌'} 辉光确实加进去了（相对原图 {glow:.3e}）")
    ok = (glow > 1e-4) and ok
    return ok


def _check_dust(lab: WgslLab, w: int, h: int) -> bool:
    """验收尘埃粒子（**散射**操作 + u32 定点原子加）。

    这是唯一一个线程数与像素数不成比例的阶段，也是唯一需要原子操作的阶段。
    判据用容差（每颗有 1/(2·SCALE) 的定点量化误差），但要落到 1e-5 以内。

    ⚠️ 另外要单独验**循环性**：dust_layer 是时序铁律 3 的直接依赖项
    （粒子必须在 t=1 时精确回到 t=0 的位置），所以在 GPU 侧也要验证
    frame(t=1) 与 frame(t=0) **逐位相同**。
    """
    from pixelart.animate import dust_layer
    from pixelart.webgpu import (
        dust_apply_uniforms, dust_fixed_point_scale, dust_particle_params,
        dust_splat_uniforms,
    )

    lit = _test_image(w, h, seed=41) * 0.6        # 压暗一点，让粒子看得见
    far = _test_depth(w, h)
    count, seed = 200, 11
    t = 0.37
    twinkle, fade_far, boost = 0.55, 0.65, 1.8
    bright = 0.55
    light_xy = (0.62, 0.28)

    par = dust_particle_params(count, seed=seed)
    scale = dust_fixed_point_scale(count)
    n = w * h * 3
    npix = w * h

    def run_frame(tt):
        binbuf = lab.run(_load_wgsl("dust_splat"), "main", None,
                         inputs={"par": par.reshape(-1),
                                 "far": far.reshape(-1)},
                         uniforms=dust_splat_uniforms(
                             w, h, count, tt, twinkle, fade_far, boost,
                             light_xy, scale, True),
                         out_count=npix, out_dtype=np.uint32,
                         label="dust_splat",
                         workgroup=(64, 1),
                         dispatch=((count + 63) // 64, 1))
        return binbuf.reshape(h, w)

    bins = run_frame(t)
    got = lab.run(_load_wgsl("dust_apply"), "main", (w, h),
                  inputs={"bin": bins.reshape(-1), "lit": lit.reshape(-1)},
                  uniforms=dust_apply_uniforms(w, h, scale, bright),
                  out_count=n, out_shape_tail=(h, w, 3), label="dust_apply")

    # numpy 参考
    d = dust_layer((h, w), far, t=t, count=count, seed=seed,
                   light_xy=light_xy, light_boost=boost,
                   twinkle=twinkle, fade_far=fade_far)
    want = np.clip(lit + d * bright, 0.0, 1.0)

    ok = compare("尘埃（200 颗粒 + 定点原子加）", got, want, tol=1e-5)

    # 定点缓冲本身也要对（这是量化误差的真正来源）
    # numpy 侧复算：每颗 clip(a,0,1)，累加，再 clip 到 1
    from pixelart.animate import hash01
    import math
    ref = np.zeros((h, w), dtype=np.float64)
    for i in range(count):
        x0, y0 = hash01(seed, i, 1), hash01(seed, i, 2)
        ax = 0.045 * (0.30 + 0.70 * hash01(seed, i, 3))
        ay = 0.045 * (0.30 + 0.70 * hash01(seed, i, 4))
        fx = 1 + int(hash01(seed, i, 5) * 2.999)
        fy = 1 + int(hash01(seed, i, 6) * 2.999)
        phx = 2 * math.pi * hash01(seed, i, 7)
        phy = 2 * math.pi * hash01(seed, i, 8)
        nx = (x0 + ax * math.sin(2 * math.pi * fx * t + phx)) % 1.0
        ny = (y0 + ay * math.sin(2 * math.pi * fy * t + phy)) % 1.0
        px, py = int(nx * w) % w, int(ny * h) % h
        tf = 1 + int(hash01(seed, i, 9) * 4.999)
        tp = 2 * math.pi * hash01(seed, i, 10)
        tw = 1.0 - twinkle * 0.5 * (1.0 - math.sin(2 * math.pi * tf * t + tp))
        a = (0.35 + 0.65 * hash01(seed, i, 11)) * tw
        a *= 1.0 - fade_far * float(np.clip(far[py, px], 0, 1))
        dx = px / max(w - 1, 1) - light_xy[0]
        dy = py / max(h - 1, 1) - light_xy[1]
        a *= 1.0 + (boost - 1.0) / (1.0 + 6.0 * math.hypot(dx, dy))
        ref[py, px] += float(np.clip(a, 0.0, 1.0))

    got_amt = np.minimum(bins.astype(np.float64) / scale, 1.0)
    want_amt = np.clip(ref, 0.0, 1.0)
    hit = want_amt > 0
    ndiff = int((np.abs(got_amt - want_amt) > 1e-6).sum())
    ok = compare("  · 定点累加缓冲（命中 %d 个像素）" % hit.sum(),
                 got_amt, want_amt, tol=1e-6) and ok
    print(f"       命中像素数 GPU={(bins > 0).sum()}  numpy={(ref > 0).sum()}")

    # 非空转：关掉粒子后画面必须不同
    empty = np.zeros((h, w), np.uint32)
    off = lab.run(_load_wgsl("dust_apply"), "main", (w, h),
                  inputs={"bin": empty.reshape(-1), "lit": lit.reshape(-1)},
                  uniforms=dust_apply_uniforms(w, h, scale, bright),
                  out_count=n, out_shape_tail=(h, w, 3), label="dust_apply")
    moved = float(np.abs(got - off).max())
    print(f"  {'✅' if moved > 1e-3 else '❌'} 粒子确实生效（无粒子时最大差 {moved:.3e}）")
    ok = (moved > 1e-3) and ok

    # ⭐ 循环性：t=1 必须与 t=0 逐位相同（时序铁律 3）
    b0, b1 = run_frame(0.0), run_frame(1.0)
    same = bool(np.array_equal(b0, b1))
    print(f"  {'✅' if same else '❌'} 循环性：定点缓冲 frame(t=1) == frame(t=0)"
          f"（不同 {int((b0 != b1).sum())} 个像素）")
    ok = same and ok

    # ══ 高密度：强制原子加**碰撞**，并验证与顺序无关 ══
    #
    # ⚠️ 200 颗撒在 96×64 上几乎不碰撞（实测 199/200 命中不同像素），
    #    而"定点原子加"的全部意义恰恰在于**多颗落在同一像素时仍与顺序无关**。
    #    所以必须单独造一个高密度场景 —— 否则这个测试等于没测原子加。
    dense = 4000
    par_d = dust_particle_params(dense, seed=seed)
    scale_d = dust_fixed_point_scale(dense)

    def run_dense():
        return lab.run(_load_wgsl("dust_splat"), "main", None,
                       inputs={"par": par_d.reshape(-1), "far": far.reshape(-1)},
                       uniforms=dust_splat_uniforms(
                           w, h, dense, t, twinkle, fade_far, boost,
                           light_xy, scale_d, True),
                       out_count=npix, out_dtype=np.uint32, label="dust_splat",
                       workgroup=(64, 1),
                       dispatch=((dense + 63) // 64, 1)).reshape(h, w)

    d1 = run_dense()
    d2 = run_dense()                       # 再跑一次：原子加顺序由驱动决定
    n_hit = int((d1 > 0).sum())
    excess = dense - n_hit                 # 重复写入同一像素的次数
    good = excess > 0
    print(f"  {'✅' if good else '❌'} 确实产生了原子加碰撞"
          f"（{dense} 颗 → {n_hit} 个像素，重复写入 {excess} 次）")
    ok = good and ok

    good = bool(np.array_equal(d1, d2))
    print(f"  {'✅' if good else '❌'} 两次运行结果**逐位相同**（原子加顺序无关性）")
    ok = good and ok

    # ⚠️ 要断言的是**不变式**（不溢出），不是"必须降档"——
    #    4000 颗时 limit=1073741 ≥ 2^20，本来就不需要降档，
    #    我第一版把打印判成"必须降档"，于是显示 ❌ 而 ok 仍是 True：
    #    **打印的判定和真正的断言不一致**，这种检查等于没有。
    no_overflow = dense * scale_d <= (1 << 32) - 1
    print(f"  {'✅' if no_overflow else '❌'} 定点累加不溢出"
          f"（{dense} 颗 → scale={scale_d}，count·scale={dense * scale_d} ≤ 2³²-1）")
    ok = no_overflow and ok

    # 玩家把 count 调很大时必须真的降档 —— 这才是"降档"该测的地方
    big = 100000
    sc_big = dust_fixed_point_scale(big)
    good = sc_big < (1 << 20) and big * sc_big <= (1 << 32) - 1
    print(f"  {'✅' if good else '❌'} 超大 count 时标度自动降档"
          f"（{big} 颗 → scale={sc_big}）")
    ok = good and ok
    return ok


def _test_image(w: int, h: int, seed: int = 7):
    """造一张有梯度和结构的测试图。用固定 seed，结果可复现。

    刻意混入：暗部、亮部、高饱和色、接近中性的浅色 ——
    因为项目历史上踩过的坑（色板槽位饿死、饱和雾色）全都出现在这些区域。
    """
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 0.15 + 0.55 * (xx / max(w - 1, 1))
    g = 0.12 + 0.50 * (yy / max(h - 1, 1))
    b = 0.20 + 0.30 * np.sin(xx / 9.0) * 0.5 + 0.25
    rgb = np.stack([r, g, b], -1)
    # 一块高饱和色（考色板槽位）
    rgb[h // 4:h // 4 + h // 5, w // 5:w // 5 + w // 4, 0] = 0.85
    # 一块明亮中性色（考"白墙被涂绿"）
    rgb[2 * h // 3:2 * h // 3 + h // 6, 2 * w // 3:2 * w // 3 + w // 6] = 0.82
    rgb += rs.uniform(-0.02, 0.02, rgb.shape).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _test_depth(w: int, h: int):
    """造一张偏斜的深度图 —— 真实的单目深度就是这个形状（一半像素挤在高位）。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 0.85 + 0.12 * (yy / max(h - 1, 1)) + 0.02 * np.sin(xx / 11.0)
    z = np.clip(z, 0, 1)
    # 前景放一个矩形（近）
    z[h // 2:h // 2 + h // 4, w // 4:w // 4 + w // 3] = 0.15
    return z.astype(np.float32)


def _test_palette(n: int = 32, seed: int = 3):
    """造一块色板：一半是绿（模拟"少数派色区饿死"的病态场景）。"""
    rs = np.random.RandomState(seed)
    greens = np.stack([rs.uniform(0.1, 0.5, n // 2),
                       rs.uniform(0.5, 0.9, n // 2),
                       rs.uniform(0.1, 0.4, n // 2)], -1)
    rest = rs.uniform(0.05, 0.95, (n - n // 2, 3))
    return np.clip(np.concatenate([greens, rest], 0), 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
def _check_parallax(lab: "WgslLab", w: int, h: int) -> bool:
    """分层视差：WGSL 前向 splatting vs numpy 参考。

    ⚠️ 这个 pass 是**唯一**一个用 u32 定点原子累加的地方（除尘埃外），
    所以判据要同时看三件事：

      ① 数值 vs CPU 参考：定点量化误差 ~1e-6 量级（SCALE=2^20）——
         比一个 uint8 色阶小 8000 倍，落在容差内即可；
      ② **可复现**：同一输入跑两次必须**逐位相同**（这正是用定点而不是
         浮点原子加的目的）；
      ③ k=0 分支必须**逐位拷贝**（不做定点量化）—— 否则会破坏
         "parallax=0 时输出与旧版逐位相同"这条不变量。
    """
    from pixelart.parallax import (
        DRIFT, FAR_GAIN, LAYERS, NEAR_GAIN, REL_AMPLITUDE, SPLAT_SCALE,
        layer_edges, parallax_warp,
    )
    from pixelart.webgpu import (parallax_apply_uniforms, parallax_layout,
                                 parallax_splat_uniforms)

    rng = np.random.RandomState(11)
    npix = w * h
    # 用**真实分布的深度**（beta 偏远景）—— 层边界是分位数，
    # 深度分布假了小层就切不出有意义的层。
    far = np.clip(rng.beta(2.0, 3.5, size=(h, w)).astype(np.float32), 0.0, 1.0)
    base = (0.15 + 0.75 * rng.rand(h, w, 3)).astype(np.float32)

    edges = layer_edges(far, LAYERS)
    layout = parallax_layout(npix)
    amp = 0.5
    pivot = (0.5 * (w - 1), 0.42 * (h - 1))
    ok = True

    def run_gpu(t: float):
        from pixelart.parallax import parallax_wave
        wave = parallax_wave(t)
        k = amp * REL_AMPLITUDE * w * wave
        acc = lab.run(_load_wgsl("parallax_splat"), "main", (w, h),
                      inputs={"edges": edges, "base": base.reshape(-1),
                              "far": far.reshape(-1)},
                      uniforms=parallax_splat_uniforms(
                          w, h, k, wave, pivot, DRIFT, len(edges),
                          NEAR_GAIN, FAR_GAIN, SPLAT_SCALE),
                      out_count=npix * 5, out_dtype=np.uint32,
                      label="parallax_splat")
        out = lab.run(_load_wgsl("parallax_apply"), "main", (w, h),
                      inputs={"acc": acc, "base": base.reshape(-1),
                              "far": far.reshape(-1)},
                      uniforms=parallax_apply_uniforms(
                          w, h, SPLAT_SCALE, k, layout["far_off"]),
                      out_count=layout["total"], label="parallax_apply")
        fo = layout["far_off"]
        wb = out[:npix * 3].reshape(h, w, 3)
        wf = out[fo:fo + npix].reshape(h, w)
        return wb, wf

    # ① 数值 vs numpy
    t = 0.3
    gb, gf = run_gpu(t)
    cb, cf, info = parallax_warp(base, far, t, amp=amp, layers=LAYERS,
                                 near_gain=NEAR_GAIN, far_gain=FAR_GAIN,
                                 drift=DRIFT, pivot=(0.5, 0.42))
    ok &= compare("视差 base（splat+apply）", gb, cb, tol=2e-5)
    ok &= compare("视差 far", gf, cf, tol=2e-5)
    print(f"    （numpy 侧最大位移 {info['max_off_px']:.1f}px，"
          f"破洞 {info['holes_pct']:.2f}%）")

    # ② 可复现：同输入两次**逐位相同**（定点的意义）
    gb2, gf2 = run_gpu(t)
    same = np.array_equal(gb, gb2) and np.array_equal(gf, gf2)
    print(f"  {'✅' if same else '❌'} 两次运行逐位相同（定点的意义）")
    ok &= same

    # ③ k=0 → 逐位拷贝，且不经定点量化
    z0, z1 = run_gpu(0.0)
    copy_ok = (np.array_equal(z0, base) and np.array_equal(z1, far))
    print(f"  {'✅' if copy_ok else '❌'} t=0 走逐位拷贝（未过定点量化）")
    ok &= copy_ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="fog", help="要验收的阶段")
    ap.add_argument("--w", type=int, default=96)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--backend", default=None, help="vulkan / d3d12")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    stages = {
        "fog": _check_fog,
        "volumetric": _check_volumetric,
        "chain": _check_chain,
        "bloom": _check_bloom,
        "dust": _check_dust,
        "parallax": _check_parallax,
        "tail": _check_tail,
    }
    if args.list:
        print("可用阶段:", ", ".join(stages))
        return 0

    if args.stage not in stages:
        print(f"未知阶段 {args.stage}；可用: {', '.join(stages)}")
        return 2

    lab = WgslLab(args.backend)
    print("=" * 74)
    print("  WGSL 验收台")
    print("=" * 74)
    print(f"  适配器  {lab.describe()}")
    print(f"  尺寸    {args.w}×{args.h}")
    print()

    ok = stages[args.stage](lab, args.w, args.h)
    print()
    print("=" * 74)
    print(f"  {'全部通过' if ok else '存在差异'}")
    print("=" * 74)
    return 0 if ok else 1


def _load_wgsl(name: str) -> str:
    f = WGSL_DIR / f"{name}.wgsl"
    if not f.exists():
        raise FileNotFoundError(f"找不到着色器 {f}")
    return f.read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
