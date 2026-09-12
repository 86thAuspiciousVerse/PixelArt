"""浏览器渲染管线的前置检查 —— 不打开浏览器也能证明"管线建得起来"。

═══ 为什么需要它 ═══

浏览器里的 WebGPU 代码有一个很不友好的性质：**它只在真浏览器里才会执行**。
所以"着色器能不能编译""绑定数对不对""uniform 长度够不够"这些问题，
按常理只能在打开浏览器之后才发现 —— 而那是最贵的时候。

但其中大部分其实**可以离线验证**：

  1. **WGSL 能不能编译** —— wgpu-py 用的是同一套 WGSL 前端
     （naga），编译通过与否和浏览器一致。
  2. **绑定数 / uniform 长度对不对** —— 从 .wgsl 里解析出来，
     和 `m3_webgpu.js` 里声明的表交叉核对。
  3. **管线能不能建** —— 按 JS 里声明的绑定布局去建管线，
     和浏览器里做的事一模一样（同一套校验规则）。
  4. **JS 语法** —— node --check。

真正只能靠浏览器的，是"设备能力"和"画出来对不对"。
那部分由 `tools/wgsl_lab.py` 的数值验收 + 界面的回退机制兜住。

═══ ⚠️ 这个工具抓的是哪一类 bug ═══

最容易犯、也最难在浏览器里定位的一类：**JS 里写的绑定数/uniform 长度
与 .wgsl 的声明不一致**。WebGPU 的报错只会说
"bound with size 64 where the shader expects 96"，不给绑定名，
在浏览器控制台里要一行行猜。这里把两边摆在一起比对，一次说清。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WGSL_DIR = ROOT / "tools" / "wgsl"
JS_RENDERER = ROOT / "tools" / "m3_webgpu.js"

#: present 是渲染管线（顶点 + 片元），不走 compute 那套检查
RENDER_SHADERS = {"present"}


def parse_wgsl(src: str) -> dict:
    """从 WGSL 里解析出各 binding 的声明，以及 uniform 数组的长度。

    只看 `@group(0) @binding(N) var<...>` 这一行 —— 我们所有着色器都是单组。
    """
    bindings = {}
    for m in re.finditer(
            r"@group\(0\)\s*@binding\((\d+)\)\s*var<([^>]+)>\s*(\w+)\s*:\s*([^;]+);",
            src):
        idx = int(m.group(1))
        addr = m.group(2).strip()
        name = m.group(3)
        typ = m.group(4).strip()
        vec4n = None
        vm = re.search(r"array<vec4<f32>,\s*(\d+)\s*>", typ)
        if vm:
            vec4n = int(vm.group(1))
        bindings[idx] = {"addr": addr, "name": name, "type": typ, "vec4": vec4n}
    return bindings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--no-wgpu", action="store_true",
                    help="只做静态核对，不编译（没有 GPU 的环境）")
    args = ap.parse_args()

    print("浏览器渲染管线前置检查")
    print("=" * 76)

    # ── ① JS 语法 ──
    js_files = ["m3_wgsl_uniforms.js", "m3_webgpu.js"]
    ok = True
    print("  ① JS 语法（node --check）")
    for f in js_files:
        path = ROOT / "tools" / f
        r = subprocess.run(["node", "--check", str(path)],
                           capture_output=True, text=True, encoding="utf-8")
        good = r.returncode == 0
        ok = ok and good
        print(f"    {'✅' if good else '❌'} {f}"
              + ("" if good else f"\n       {(r.stderr or '').strip()[:200]}"))

    # ── ② 从 JS 里取声明表 ──
    js_src = JS_RENDERER.read_text(encoding="utf-8")
    m_uniform = re.search(r"var UNIFORM_LEN = \{(.*?)\};", js_src, re.S)
    m_bind = re.search(r"var BINDING_COUNT = \{(.*?)\};", js_src, re.S)
    m_rbind = re.search(r"var RENDER_BINDING_COUNT = \{(.*?)\};", js_src, re.S)
    m_shaders = re.search(r"var SHADERS = \[(.*?)\];", js_src, re.S)
    if not (m_uniform and m_bind and m_shaders):
        print("  ❌ JS 里没找到 UNIFORM_LEN / BINDING_COUNT / SHADERS")
        return 1

    def parse_map(block: str) -> dict:
        out = {}
        for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", block):
            out[k] = int(v)
        return out

    js_uniform = parse_map(m_uniform.group(1))
    js_bind = parse_map(m_bind.group(1))
    # present 是渲染管线，绑定数单独一张表（compute 的布局不适用）
    js_rbind = parse_map(m_rbind.group(1)) if m_rbind else {}
    for k, v in js_rbind.items():
        js_bind[k] = v
    js_shaders = re.findall(r"'([\w_]+)'", m_shaders.group(1))

    print()
    print(f"  ② JS 声明的着色器：{', '.join(js_shaders)}")
    print(f"     JS 声明的绑定数：{js_bind}")
    print(f"     JS 声明的 uniform 长度：{js_uniform}")

    # ── ③ 与 .wgsl 交叉核对 ──
    print()
    print("  ③ JS 声明 vs .wgsl 实际声明")
    print()
    print(f"    {'pass':15s} {'绑定数':>16s} {'uniform 长度':>18s}  结果")
    print("    " + "-" * 62)
    parsed = {}
    for name in js_shaders:
        f = WGSL_DIR / f"{name}.wgsl"
        if not f.exists():
            print(f"    ❌ {name}: 文件不存在 {f.name}")
            ok = False
            continue
        b = parse_wgsl(f.read_text(encoding="utf-8"))
        parsed[name] = b
        n_bind = len(b)
        # 每个 .wgsl 的 binding 0 是 uniform 块
        u0 = b.get(0, {}).get("vec4")
        # present 是渲染管线：它的 binding 0 是只读 storage，不需要 vec4 对齐检查
        expect_bind = js_bind.get(name)
        if name in RENDER_SHADERS:
            good = (n_bind == expect_bind)
            ok = ok and good
            print(f"    {'✅' if good else '❌'} {name:13s} "
                  f"{n_bind:6d} / JS {str(expect_bind):>4s}   "
                  f"{'（渲染管线）':>18s}   {'' if good else '← 不一致'}")
            continue
        expect_uni = js_uniform.get(name)
        good = (n_bind == expect_bind) and (u0 == (expect_uni // 4))
        ok = ok and good
        print(f"    {'✅' if good else '❌'} {name:13s} "
              f"{n_bind:6d} / JS {str(expect_bind):>4s}   "
              f"{str(u0) + ' vec4':>10s} / JS {str(expect_uni) + ' f32':>8s}   "
              f"{'' if good else '← 不一致'}")
        # 顺带报出每个 binding 的读写属性 —— 这正是"填错类型"的高发处
        if not good:
            print(f"       .wgsl 实际：{b}")

    # uniform 长度必须能被 4 整除（vec4 对齐）
    for name, n in js_uniform.items():
        if n % 4 != 0:
            print(f"    ❌ {name}: uniform 长度 {n} 不是 4 的倍数（vec4 对齐）")
            ok = False

    # ── ④ 编译 + 建管线 ──
    print()
    if args.no_wgpu:
        print("  ④ 跳过编译（--no-wgpu）")
    else:
        try:
            import wgpu  # noqa: F401
        except ImportError:
            print("  ④ 跳过编译（没装 wgpu）")
            wgpu = None
        if wgpu is not None:
            print("  ④ 编译全部 WGSL 并按 JS 声明的布局建管线")
            sys.path.insert(0, str(ROOT / "tools"))
            from wgsl_lab import WgslLab
            lab = WgslLab(args.backend)
            print(f"     适配器 {lab.describe()}")
            dev = lab.device
            for name in js_shaders:
                src = (WGSL_DIR / f"{name}.wgsl").read_text(encoding="utf-8")
                try:
                    mod = dev.create_shader_module(code=src, label=name)
                except Exception as e:                          # noqa: BLE001
                    print(f"    ❌ {name}: 编译失败 {e}")
                    ok = False
                    continue
                try:
                    if name in RENDER_SHADERS:
                        layout = dev.create_bind_group_layout(entries=[
                            {"binding": i, "visibility": wgpu.ShaderStage.FRAGMENT,
                             "buffer": {"type": "read-only-storage"}}
                            for i in range(js_bind.get(name, len(parsed[name])))
                        ])
                        dev.create_render_pipeline(
                            layout=dev.create_pipeline_layout(
                                bind_group_layouts=[layout]),
                            vertex={"module": mod, "entry_point": "vs"},
                            fragment={"module": mod, "entry_point": "fs",
                                      "targets": [{"format": "bgra8unorm"}]},
                            primitive={"topology": "triangle-list"},
                        )
                        print(f"    ✅ {name}: 渲染管线建成功（vs + fs → bgra8unorm）")
                    else:
                        n = js_bind.get(name)
                        layout = dev.create_bind_group_layout(entries=[
                            {"binding": i, "visibility": wgpu.ShaderStage.COMPUTE,
                             "buffer": {"type": ("storage" if i == n - 1
                                                 else "read-only-storage")}}
                            for i in range(n)
                        ])
                        dev.create_compute_pipeline(
                            layout=dev.create_pipeline_layout(
                                bind_group_layouts=[layout]),
                            compute={"module": mod, "entry_point": "main"},
                        )
                        print(f"    ✅ {name}: 计算管线建成功（{n} 个绑定）")
                except Exception as e:                          # noqa: BLE001
                    print(f"    ❌ {name}: 建管线失败 {e}")
                    ok = False

    print()
    print("=" * 76)
    print(f"  {'全部通过' if ok else '存在问题'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
