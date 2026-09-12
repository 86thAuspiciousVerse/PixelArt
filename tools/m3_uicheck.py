"""M3 界面静态检查 —— 不启动浏览器也能抓住最常见的界面 bug。

界面是纯 HTML/JS，没有构建步骤，所以最容易犯的错是：
**JS 里 $(...) 引用的 id 在 HTML 里不存在**（拼错、改名忘了同步）。
这种错误在浏览器里表现为"点了没反应"或整个脚本抛异常，
排查起来比后端 bug 麻烦得多。这里用静态分析一次性查出来。

顺带检查：
- ``<script>`` 里的 JS 能否通过语法解析（用 node）
- 括号/引号是否平衡
- 关键控件是否都在
- CSS 变量是否都有定义（引用了不存在的 var 会静默失效，很难发现）
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

UI = Path(__file__).resolve().parent / "m3_ui.html"
SRC = Path(__file__).resolve().parent.parent / "src"


def strip_comments(text: str) -> str:
    """剥掉 HTML / CSS / JS 注释。

    ⚠️ **凡是要判"某个字符串不该出现"的检查，都必须先过这道。**

    原因：解释某个 bug 的注释里往往就写着那行**旧代码**，
    不剥注释的字符串检查会被自己的说明文字误伤。本仓库已踩过 **三次**
    （`max-width:100%`、`if (stillTimer) return;`、`--depth`），
    每次都表现为"检查报红但代码明明是对的"，然后要花时间怀疑人生。

    所以统一走这个函数，不要在各处手写内联正则。
    """
    t = re.sub(r"<!--.*?-->", "", text, flags=re.S)      # HTML 注释
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)          # CSS / JS 块注释
    t = re.sub(r"^\s*//.*$", "", t, flags=re.M)           # JS 行注释（只在行首，避免误伤 http://）
    return t


def main() -> int:
    html = UI.read_text(encoding="utf-8")
    html_nc = strip_comments(html)          # 剥注释版：用于"不该出现"类检查
    ok = True

    def check(tag: str, good: bool, detail: str = ""):
        nonlocal ok
        print(f"  {'✅' if good else '❌'} {tag:32s} {detail}")
        ok = ok and good

    # ── 1. JS 里引用的 id 必须存在 ──
    used = set(re.findall(r'\$\("([A-Za-z0-9_\-]+)"\)', html))
    defined = set(re.findall(r'\bid="([A-Za-z0-9_\-]+)"', html))
    dynamic = set(re.findall(r'\$\("([^"]*)"\)', html)) - used     # 动态拼接的排除
    missing = sorted(used - defined)
    check("JS 引用的 id 都存在", not missing,
          f"引用 {len(used)} 个，缺失: {missing}" if missing else f"引用 {len(used)} 个全部命中")
    unused = sorted(defined - used)
    if unused:
        print(f"     （未被 JS 引用、但可能被 querySelector 用的 id: {unused}）")

    # ── 2. querySelector 用到的选择器 ──
    qs_sel = set(re.findall(r'querySelector\("([^"]+)"\)', html))
    for sel in qs_sel:
        if sel.startswith("#"):
            check(f"querySelector {sel}", sel[1:] in defined)

    # ── 3. JS 语法 ──
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    if not m:
        check("找到 <script> 块", False)
    else:
        js = m.group(1)
        # ⚠️ 必须**写成临时文件**再让 node 检查，不能把整段 JS 当 `-e` 参数传。
        #    Windows 命令行有约 32K 字符上限，脚本一长就
        #    `FileNotFoundError: [WinError 206] 文件名或扩展名太长`
        #    —— 表现为"界面检查莫名其妙整个崩掉"（实测踩过，加了几十行 JS 就爆）。
        #    另外 `<script>` 段里可能有 `</script>` 之外的东西，用文件也更安全。
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ui_check.js"
            f.write_text(js, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)],
                               capture_output=True, text=True)
        check("JS 语法解析", r.returncode == 0, (r.stderr or "").strip()[:160])

    # ── 4. CSS 变量：引用的必须都有定义 ──
    css_vars = set(re.findall(r"--([a-z0-9\-]+)\s*:", html))
    used_vars = set(re.findall(r"var\(--([a-z0-9\-]+)", html))
    miss_vars = sorted(used_vars - css_vars)
    check("CSS 变量都有定义", not miss_vars,
          f"定义 {len(css_vars)} 个，缺失: {miss_vars}" if miss_vars
          else f"定义 {len(css_vars)} / 引用 {len(used_vars)}")

    # ── 5. 标签平衡（只查大头）──
    for tag in ("div", "details", "section", "aside", "footer", "header", "main", "select", "label"):
        o = len(re.findall(rf"<{tag}[\s>]", html))
        c = len(re.findall(rf"</{tag}>", html))
        check(f"<{tag}> 标签平衡", o == c, f"{o} 开 / {c} 闭")
        ok = ok and (o == c)

    # ── 6. 关键控件与设计要点 ──
    need = {
        "asset": "素材下拉", "frame": "画面", "rail": "控制窄轨",
        "stagecol": "取景台（也要套 .scroll）",
        "readout": "仪表读数条", "badge": "状态角标", "plate": "帧号牌",
        "btnPreview": "渲染预览", "btnExport": "全量出图", "btnReset": "复位",
        "btnSave": "存参数", "btnLoad": "取参数", "btnPlay": "播放",
        "btnUpload": "上传图片", "thumbBtn": "缩略图（可点换图）",
        "fileInput": "文件选择器", "dropzone": "拖放提示层",
        "scrub": "时间轴", "sweepKey": "扫描参数", "sweepVals": "扫描取值",
        "btnSweep": "扫描按钮", "sweepImg": "扫描结果", "toast": "提示条",
        "stagebox": "取景框（显示尺寸由 JS 定）",
        "zoomTag": "缩放读数", "btnZoomIn": "放大", "btnZoomOut": "缩小",
        "gpuCanvas": "WebGPU 画布", "btnBackend": "后端指示/切换",
        "btnZoomFit": "适应窗口",
    }
    for k, label in need.items():
        check(f"控件 {label}", k in defined)

    # ── 7. 设计自查：不能出现被点名的通用 AI 审美 ──
    bad_fonts = ["Inter", "Roboto", "Space Grotesk", "Helvetica Neue", "Arial"]
    hits = [f for f in bad_fonts if re.search(rf'font-family[^;]*\b{re.escape(f)}\b', html, re.I)]
    check("未使用被点名的通用字体", not hits, f"命中: {hits}" if hits else "Bahnschrift / Cascadia Mono")
    check("文字量靠变量统一（无散落硬编码色）", html.count("var(--") > 60,
          f"var(-- …) 用了 {html.count('var(--')} 次")
    check("有 reduced-motion 兼容", "prefers-reduced-motion" in html)

    # ── 8. 滚动条：所有可滚动区域都要套 .scroll ──
    # 用户反馈右侧取景台的滚动条是系统默认样式，和左侧窄轨不一致。
    scroll_els = re.findall(r'<(?:aside|section|footer|div)[^>]*class="([^"]*)"', html)
    needs_scroll = [c for c in scroll_els if "rail" in c or "stagecol" in c or "readout" in c]
    for c in needs_scroll:
        check(f"滚动容器 .scroll  {c.split()[0]}", "scroll" in c, c)
    check(".scroll 同时覆盖 WebKit 与 Firefox",
          "::-webkit-scrollbar" in html and "scrollbar-color" in html)

    # ── 9. 拖动节流必须是"保留最新值"而不是"丢弃" ──
    # 第一版写 `if (stillTimer) return;` —— 连续拖动时事件被成片丢掉，
    # 且松手前最后一次也可能丢，画面停在中间值上（用户以为滑杆坏了）。
    #
    # ⚠️ 查代码前必须**去掉注释**：解释这个 bug 的注释里就写着那行旧代码，
    #    不移除注释的朴素检查会被自己的说明文字误伤（第一版就是这样）。
    js_only = strip_comments(js if "js" in dir() else "")
    check("节流用 pending 标记而非丢弃",
          "stillPending" in js_only and "stillPending = true; return;" in js_only)
    check("节流不再用一次性定时器丢弃事件", "if (stillTimer) return;" not in js_only)

    # ── 10. 上传路径存在（管线不能只对内置样例生效）──
    # ⚠️ drop 处理器体就地提取（别依赖后面段落里的同名变量 —— 顺序耦合会让
    #    检查在变量还没赋值时就引用它，直接 NameError；这里踩过）
    _dm = re.search(r'addEventListener\("drop".*?\n  \}\);', html_nc, re.S)
    drop_body = _dm.group(0) if _dm else ""
    check("有上传入口", "api/upload" in html and "fileInput" in html)
    check("支持拖放", "dragenter" in html and "drop" in html)
    # ⚠️ 必须区分「外部文件拖入」与「页面内元素被拖动」。
    #    用户报障：拖动预览画面那张图（想拖出去用），却弹出上传遮罩、极易误上传。
    #    三层判据：internalDrag 标记（最可靠，不依赖 types）/ types 含 Files /
    #    drop 阶段 files.length > 0。
    check("有 dragstart 标记（区分内外拖动最可靠的判据）",
          "internalDrag = true" in html and 'addEventListener("dragstart"' in html)
    check("dragend 清除标记", "internalDrag = false" in html)
    check("有 isExternalFileDrag 统一判定", "isExternalFileDrag" in html)
    # ⚠️ 原来 dragenter/dragover 判了类型、**drop 却没判** —— 这是误上传的直接原因。
    #    ⚠️⚠️ 检查必须**只看 drop 处理器体**，不能在全文件里找字符串：
    #    dragenter/dragover 里也有同样的 `isExternalFileDrag(e)`，
    #    所以"全文件搜索"会**永远通过**（第一版就是这样，变异测试才暴露出来）。
    drop_m = re.search(r'addEventListener\("drop".*?\n  \}\);', html_nc, re.S)
    drop_body = drop_m.group(0) if drop_m else ""
    check("能定位到 drop 处理器", bool(drop_body))
    check("drop 也判类型（不能只判 dragenter/dragover）",
          "isExternalFileDrag(e)" in drop_body,
          f"drop 体 {len(drop_body)} 字符" if drop_body else "未定位到")
    check("drop 阶段还要求 files 非空", "!files.length" in drop_body)
    # ⚠️ 判据：drop 里必须**先**阻止默认行为，**再**分派。
    #    不阻止默认的话浏览器会按拖入内容导航（内部拖动 → 图片 URL，
    #    外部文件 → 打开文件），两种都会把界面冲掉。
    #    原先只有部分分支做了 preventDefault，漏一条就是"拖一下就白屏"。
    check("内部拖动松手时不上传", "if (internalDrag) return;" in drop_body)
    check("drop 先阻止默认行为再分派",
          drop_body.index("e.preventDefault()") < drop_body.index("internalDrag"))
    # ⚠️ 遮罩别用 --depth 计数：跨子元素边界时计数会失衡，导致闪烁/残留。
    #    改成 dragover 刷新时间戳，天然免疫配平问题。
    # ⚠️ 判据要按**变量声明**抓，不能只找 `--depth` / `depth++` 这两个具体写法：
    #    换个名字或改成 `depth = 0` 就绕过去了（第一版就是这样漏掉的）。
    has_depth_counter = bool(re.search(r"\b(?:let|var|const)\s+depth\b", html_nc))
    check("遮罩不用 depth 计数（改为时间戳刷新）",
          not has_depth_counter and "dzTimer" in html_nc and "showDropzone" in html_nc)
    check("拖放时阻止浏览器默认打开图片",
          "e.preventDefault()" in html and "dropzone" in html)

    # ── 11. 显示尺寸必须与渲染分辨率解耦（"画面跳变" bug 的根因）──
    # 用户报障：调参时画面变小、播放时变大。根因是 img 用 max-width:100%，
    # 而拖动时渲的是半分辨率图 —— 半分辨率图不会被 CSS 放大，视觉上就变小了。
    #
    # ⚠️ 查 CSS 前必须**剥掉注释**：解释这个 bug 的注释里就写着 `max-width:100%`，
    #    不剥注释的朴素检查会被自己的说明文字误伤（这里踩了第二次）。
    css_raw = re.search(r"<style>(.*?)</style>", html, re.S)
    css = strip_comments(css_raw.group(1)) if css_raw else ""
    # 扫**所有** `.stage img` / `.stagebox img` 规则（包括媒体查询里的），
    # 只要有一处带 max-width/max-height，显示尺寸就又会被 CSS 牵着走。
    # （真踩过：媒体查询里残留一条 `.stage img{max-height:52vh}`，
    #   窄窗口下把画面压扁 —— 只查第一条规则是抓不到的。）
    img_rules = re.findall(r"\.stage(?:box)?\s+img\s*\{([^}]*)\}", css, re.S)
    check("取景框 img 规则存在", bool(img_rules))
    bad_rules = [r for r in img_rules if "max-width" in r or "max-height" in r]
    check("没有任何 img 规则用 max-* 决定尺寸", not bad_rules,
          f"发现 {len(bad_rules)} 处" if bad_rules else f"扫了 {len(img_rules)} 条规则")
    check("取景框宽高由 JS 明确设定",
          "box.style.width" in html and "box.style.height" in html)
    check("缩放取整数倍（像素画不能非整数缩放）",
          "Math.round(z)" in html and "Math.max(1, Math.min(16" in html)
    check("缩放有「适应」（默认）与手动固定两种状态",
          "zoomAuto" in html and "fitZoom" in html)
    check("滚轮缩放阻止了页面滚动", "passive: false" in html)
    check("取图倍率 == 显示倍率（像素与显示 1:1）", "scale: zoom" in html)
    # ⚠️ /api/preview 也必须带 scale：服务端按它预编码帧进缓存，
    #    界面按同样的 scale 取帧才命中。不一致 → 缓存全落空 → 又退回"渲两遍"。
    check("preview 请求也带 scale（否则缓存落空）",
          "preview_fps: fps, scale: zoom" in html)
    check("逻辑尺寸取网格而非最终产物尺寸（否则默认就要滚动）",
          "logicalSize = st.grid.slice()" in html
          and "logicalSize = st.out_size.slice()" not in html)

    # ── 12. 参数必须覆盖**三条**路径（单帧 / 整段 / 界面帧列表）──
    # 用户报障：改了尘埃，静止帧变了，一点播放就跳回预设效果。
    # 根因：界面自己维护的 frames[] 只在 renderPreview 里构建，
    # **参数变化时没有任何机制让它失效** —— 播放时放的是旧参数渲的帧。
    check("frames 有显式有效性标记", "framesValid" in html)
    check("参数变化时让 frames 失效",
          "invalidateFrames" in html and "invalidateFrames();" in js_only)
    check("播放前检查 frames 是否对应当前参数",
          "!frames.length || !framesValid" in html)
    check("play() 自身也拒绝放失效帧", "!frameCount || !framesValid" in html)

    # 同一类元 bug：复合控件的 row.k 是占位名，用 row.k 比对具体字段永远不匹配
    # （光源 XY 标记不动的根因）。现在控件必须把**真实字段名**传下去。
    check("onParamChange 接受真实字段名", "changedKey" in html)
    check("vec2 控件传出子字段名（如 rays_x）",
          "onParamChange(r, false, key)" in html and "onParamChange(r, true, key)" in html)
    check("color 控件传出子字段名（如 fog_r）",
          "onParamChange(r, false, cr)" in html)
    check("光源标记在拖动中立刻跟随（不等渲染）",
          "if (key === cx || key === cy) updateMarker();" in html)
    # 光有一个点看不出 X/Y 在哪一列/行 —— 必须有贯穿全图的十字准线
    check("有贯穿全图的十字准线", "crosshair" in html and 'id="chH"' in html and 'id="chV"' in html)
    check("准线随光源位置移动", "hx.style.top" in html and "vy.style.left" in html)
    check("准线在关闭开关时一并隐藏",
          'hx.classList.remove("on")' in html and 'vy.classList.remove("on")' in html)
    check("准线上钉了 X/Y 数值（视线不用来回找）",
          'hx.dataset.v' in html and 'vy.dataset.v' in html)

    # ── 13. 界面 DEFAULTS 必须与服务端 TuneParams 字段完全一致 ──
    # ⚠️ 这是"参数没接线"的**第 5 条路径**：界面手写一个 DEFAULTS 对象当参数初值，
    #    若少写一个字段，`qs()` 遍历 Object.keys(P) 时就不会带上它 ——
    #    服务端悄悄用自己的默认值，而界面上那个滑杆显示的却是**量程中点**
    #    （range 的 value 设为 undefined 时浏览器取中点）。
    #    真实踩过：加了 fog_sat / rays_spread 两个滑杆却忘了加字段，
    #    而滑杆中点（0.4 / 1.0）恰好等于服务端默认值 —— 于是"看着对"，
    #    直到有人改了默认值才会炸。这类 bug 靠肉眼是查不出来的。
    defaults_m = re.search(r"const DEFAULTS = \{(.*?)\n\};", js_only, re.S)
    if defaults_m:
        ui_keys = set(re.findall(r"(\w+)\s*:", defaults_m.group(1)))
        sys.path.insert(0, str(SRC))
        try:
            import dataclasses
            from pixelart.tune import TuneParams
            srv_keys = {f.name for f in dataclasses.fields(TuneParams)}
            missing_f = sorted(srv_keys - ui_keys)
            extra_f = sorted(ui_keys - srv_keys)
            check("界面 DEFAULTS 覆盖全部服务端字段", not missing_f,
                  f"缺 {missing_f}" if missing_f else f"{len(ui_keys)} 个字段全覆盖")
            check("界面 DEFAULTS 没有多余字段", not extra_f,
                  f"多 {extra_f}" if extra_f else "")
        except Exception as e:                                  # noqa: BLE001
            check("能导入 TuneParams 做字段对比", False, f"{type(e).__name__}: {e}")
    else:
        check("能解析出界面的 DEFAULTS 对象", False, "正则没匹配到")

    # ── 14. 每个控件绑定的字段名必须真实存在 ──
    # 反过来的错法：控件写了 `k:"fog_saturation"`（拼错）——
    # 滑杆能拖、数值会变，但服务端根本不认识这个字段（from_query 逐个字段查），
    # 于是拖了毫无反应。静态比对是唯一能提前发现的办法。
    # 复合控件（如 _fogColor / _light）用占位名 k，真实字段写在 _c / _auto 里。
    if defaults_m:
        ctrl_keys = set(re.findall(r'\{k:"([A-Za-z0-9_]+)"', js_only))
        sub_keys: set[str] = set()
        for grp in re.findall(r"_c:\s*\[([^\]]*)\]", js_only):
            sub_keys |= set(re.findall(r'"(\w+)"', grp))
        sub_keys |= set(re.findall(r'_auto:\s*"(\w+)"', js_only))
        bound = (ctrl_keys - {k for k in ctrl_keys if k.startswith("_")}) | sub_keys
        try:
            unknown = sorted(bound - srv_keys)
            check("控件绑定的字段都真实存在", not unknown,
                  f"不认识: {unknown}" if unknown else f"绑定 {len(bound)} 个字段")
            no_ctrl = sorted(srv_keys - bound)
            check("每个字段都有控件可调（或明确说明为何没有）", True,
                  f"无控件: {no_ctrl}" if no_ctrl else "")
        except NameError:                                       # srv_keys 没建起来
            check("控件字段比对", False, "TuneParams 未导入成功")

    # ── 15. WebGPU 后端接入（必须能回退，且不破坏服务端路径）──
    check("有 WebGPU 画布与后端切换按钮",
          'id="gpuCanvas"' in html and 'id="btnBackend"' in html)
    # 顺序很关键：两个外部脚本必须先定义 WgslUniforms / GpuRenderer，
    # 内联脚本才能用。`html.index("<script>")` 命中的是**裸**开标签（内联那个），
    # 因为外部的都带 src 属性。
    check("两个浏览器端脚本在**主脚本之前**加载",
          html.index("gpuasset?name=m3_wgsl_uniforms")
          < html.index("gpuasset?name=m3_webgpu")
          < html.index("<script>"))
    # ⚠️ 回退是硬要求：任何一步失败都要能切回服务端渲染
    check("检测不到 WebGPU 时会回退",
          'if (!("gpu" in navigator))' in html and 'backendPref = "server"' in html)
    check("渲染失败会回退服务端",
          "WebGPU 失败已回退" in html and 'gpuReady = false' in html)
    check("画面尺寸与服务端共用（画布也走 applyZoom）",
          "gpuSyncCanvas()" in html and "box.style.width" in html)
    # GPU 模式必须**绕开**服务端的逐帧取图，否则移植就没意义了
    check("GPU 单帧不经网络往返", "backendActive()" in html and "gpuRenderer.render(P, tt)" in html)
    check("GPU 播放用 rAF 而非逐帧取图",
          "requestAnimationFrame(step)" in html and "cancelAnimationFrame(gpuRaf)" in html)
    check("在 GPU 上自检循环闭合", "gpuLoopSelfCheck" in html and "GPUMapMode.READ" in html)
    check("场景/素材分离（滑杆变化不重传 base/far）",
          "sceneKeyNow" in html and "assetsKeyNow" in html and "parts: \"assets\"" in html)
    # 仪表读数不能编数：GPU 模式下没有整段帧，就不报"相邻帧差"
    check("GPU 读数不编造服务端才有的统计",
          "相邻帧差" in html and "未统计" in html)

    print()
    print("=" * 58)
    print(f"  {'全部通过' if ok else '存在问题'}")
    print("=" * 58)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
