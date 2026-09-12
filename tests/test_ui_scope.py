"""界面脚本的**作用域**护栏 —— 一条静态规则，挡住一整类"静默失败"。

═══ 为什么需要这个文件 ═══

用户报"画面一直是黑的"。查出来的根因不是渲染，而是**作用域**：

    `m3_ui.html` 里 `init()`（启动序列的 async IIFE）**被当成了一个代码块用** ——
    为了让"声明和使用挨着写"，大量函数与状态被声明在 `init()` 里面：

        let gpuRenderer = null, gpuReady = false, ...      ← 在 init() 里
        function backendActive(){...}                      ← 在 init() 里
        function gpuSyncCanvas(){...}                      ← 在 init() 里
        function initBackend(){...}                        ← 在 init() 里
        async function runGpuSelfTest(){...}               ← 在 init() 里

    而 `renderPreview()` / `applyZoom()` / `stopPlay()` / `play()` 是**模块级**的，
    里面照样写 `backendActive()`、`gpuSyncCanvas()`、`gpuRaf` ——
    **JavaScript 的作用域是词法的（看声明位置，不是调用位置）**，
    于是这些引用全部抛 `ReferenceError`。

    症状之所以难查：
      · 首次 `renderPreview()` 在 `busy = true` **之后**抛 → `busy` 永远为 true
        → 之后每一次渲染请求都被 `if (busy) return` 吞掉 → **画面永远黑**
      · 异常抛在 async 函数里没人接 → 控制台之外**什么都看不到**
      · `?selftest=1` 偏偏是绿的 —— 自检在 `init()` 里就 `return` 了，
        用的全是在作用域内的名字。**它验的是它自己那条路。**

═══ 规则 ═══

1. `init()`（启动序列）里**只允许出现调用与赋值**，不允许声明函数/变量。
   声明放模块级 —— 这样"谁都能看见谁"这种事就不再需要推理。
2. 页面自检里那份"作用域探测"名单上的每个名字，
   都必须在**模块级**有声明（这一条不依赖浏览器，纯静态就能查）。
3. 不允许同一个名字在模块级被声明两次（改代码时最容易犯的低级错）。

这三条都不是风格要求 —— 每一条都对应上面那个真实故障的一个成因。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "tools" / "m3_ui.html"
JS = ROOT / "tools" / "m3_webgpu.js"

#: 启动序列那个 async IIFE 的签名。改这里等于改"启动序列"的定义。
INIT_SIG = "(async function init(){"


def _read_js(path: Path) -> str:
    """取出 JS 源码：`.html` 取 `<script>` 里的内容，`.js` 就是整个文件。"""
    s = path.read_text(encoding="utf-8")
    if path.suffix != ".html":
        return s
    m = re.search(r"<script>(.*)</script>", s, re.S)
    assert m, f"{path.name} 里没找到 <script>"
    return m.group(1)


def _blank_noncode(src: str) -> str:
    """把注释与字符串/模板字面量的**内容**抹成空格，**保持长度与行号不变**。

    ⚠️ 必须保持长度：我们要按字符位置做大括号配平并回报行号，
    直接删掉字符串会让行号漂移，报出来的位置就是错的（这种错很坑：
    看着像"护栏误报"，实际是护栏自己的坐标系错了）。
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if src[k] != "\n":            # 换行要留着，行号才对得上
                    out[k] = " "
            i = j
            continue
        if c in "\"'`":
            q = c
            i += 1
            while i < n and src[i] != q:
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                if i < n:
                    if src[i] != "\n":
                        out[i] = " "
                    i += 1
            i += 1                            # 收尾的引号留着（它不影响配平）
            continue
        i += 1
    return "".join(out)


def _span(src: str, start: int) -> tuple[int, int]:
    """从 `start` 处的大括号开始，返回配平后的 (起始大括号位置, 结束位置)。"""
    assert src[start] == "{", f"位置 {start} 不是大括号：{src[start-30:start+10]!r}"
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i
    raise AssertionError("大括号没有配平 —— 源文件本身有问题")


def _declared_names(statement: str) -> list[str]:
    """从一条声明语句里抽出所有被声明的名字（支持 `let a = 1, b = 2;`）。"""
    st = statement.strip()
    if st.startswith("async function "):
        st = st[len("async function "):]
    elif st.startswith("function "):
        st = st[len("function "):]
    elif st.startswith("class "):
        st = st[len("class "):]
    names = re.findall(r"([A-Za-z_$][\w$]*)\s*=", st)
    # 没有初始值的声明符（`let a, b;`）—— 只取 `,` 与 `;` 之间的裸名字
    for m in re.finditer(r",\s*([A-Za-z_$][\w$]*)\s*(?=[,;])", statement):
        names.append(m.group(1))
    if not names:
        m = re.match(r"([A-Za-z_$][\w$]*)", st)
        if m:
            names.append(m.group(1))
    return names


def _top_level_statements(src: str, lo: int, hi: int) -> list[tuple[int, str]]:
    """返回 `src[lo:hi]` 里**大括号深度为 0** 处的语句（行号, 文本）。

    深度 0 的语句 = 直接写在这段代码体里的语句。
    """
    out: list[tuple[int, str]] = []
    depth = 0
    line_no = src.count("\n", 0, lo) + 1
    buf: list[str] = []
    buf_line = line_no
    i = lo
    while i < hi:
        c = src[i]
        if c == "\n":
            line_no += 1
        if depth == 0 and c in "({[":
            pass                       # 圆括号/方括号不改变"语句"边界，忽略
        if c == "{":
            if depth == 0 and "".join(buf).strip():
                out.append((buf_line, "".join(buf).strip()))
                buf = []
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                buf = []
                buf_line = line_no + 1
        if depth == 0:
            if not buf:
                buf_line = line_no
            buf.append(c)
            if c == ";":
                txt = "".join(buf).strip()
                if txt and txt != ";":
                    out.append((buf_line, txt))
                buf = []
        i += 1
    return out


@pytest.fixture(scope="module")
def ui_src() -> str:
    return _read_js(UI)


@pytest.fixture(scope="module")
def ui_code(ui_src: str) -> str:
    """抹掉注释与字符串、但**保持行号**的源码 —— 所有静态检查都基于它。"""
    return _blank_noncode(ui_src)


# ══════════════════════════════════════════════════════════════════
def test_init_body_declares_nothing(ui_code: str):
    """⭐ `init()` 里**不允许有任何声明** —— 只能是调用与赋值。

    这是那条规则的直接编码。它挡住的正是真实故障：
    `init()` 里声明了 GPU 那一套（`gpuRenderer` / `backendActive` / `initBackend` …），
    而模块级的 `renderPreview()` 引用它们 → `ReferenceError` → 首屏永远全黑。

    ⚠️ 这条规则看似"风格"，其实是**正确性**：只要声明放在 `init()` 里，
    模块级函数就永远看不到它，而**调用点看起来完全正常**。
    """
    start = ui_code.index(INIT_SIG) + len(INIT_SIG) - 1
    lo, hi = _span(ui_code, start)
    stmts = _top_level_statements(ui_code, lo + 1, hi)

    decl_kw = ("let ", "var ", "const ", "function ", "async function ", "class ")
    bad = [(ln, t) for ln, t in stmts if t.startswith(decl_kw)]
    assert not bad, (
        "`init()` 里出现了声明 —— 它们对模块级函数**不可见**，"
        "引用时会 ReferenceError（首屏全黑就是这么来的）。"
        "请把这些声明移到模块级：\n"
        + "\n".join(f"  第 {ln} 行: {t[:90]}" for ln, t in bad))


def test_scope_probe_names_are_declared_at_module_level(ui_src: str, ui_code: str):
    """⭐ 页面自检里那份作用域名单上的每个名字，都必须在模块级有声明。

    运行时那条探测（`psScopeProbe`）能发现"名字不在作用域里"，
    但它需要真浏览器。这一条是它的**静态对偶**：不开浏览器也能拦住同一个错误，
    而且能在 pytest 里跑。
    """
    # ⚠️ 名字列表要从**原始源码**里取 —— `_blank_noncode` 会把字符串内容抹成空格，
    #    名单本身就没了。好在它**保持长度**，所以两边的偏移完全一致，可以直接按
    #    同一个区间去原文里切。
    assert len(ui_src) == len(ui_code), (
        "`_blank_noncode` 改变了长度 —— 偏移对齐的前提没了，"
        "所有按位置切原文的检查都会错位")
    m = re.search(r"function psScopeProbe\(\)\{(.*?)\n\}", ui_code, re.S)
    assert m, "找不到 psScopeProbe() —— 护栏要与它同步"
    body = ui_src[m.start(1):m.end(1)]
    names = re.findall(r"\"([A-Za-z_$][\w$]*)\"", body)
    assert len(names) >= 20, f"探测名单只有 {len(names)} 个名字，正则或源码结构变了"

    init_at = ui_code.index(INIT_SIG)
    module_scope = ui_code[:init_at]            # 启动序列之前的都是模块级
    declared: set[str] = set()
    for ln, stmt in _top_level_statements(module_scope, 0, len(module_scope)):
        if stmt.startswith(("let ", "var ", "const ", "function ",
                            "async function ", "class ")):
            declared.update(_declared_names(stmt))
    for m2 in re.finditer(r"^\s*(?:let|var|const)\s+([^;]+);", module_scope, re.M):
        declared.update(_declared_names(m2.group(0)))

    missing = [n for n in names if n not in declared]
    # 少数名字可能声明在别处（例如在更早的块里），逐个再确认一次
    really_missing = []
    for n in missing:
        if not re.search(r"(?:\bfunction\s+|\b(?:let|var|const)\s+[^;]*\b)"
                         + re.escape(n) + r"\b", module_scope):
            really_missing.append(n)
    assert not really_missing, (
        "这些名字在模块级**没有声明**，模块级函数引用它们会 ReferenceError：\n"
        f"  {really_missing}\n"
        "（运行时探测只有在真浏览器里才跑得到，静态这一条是它的兜底。）")


def test_no_name_is_declared_twice_at_module_level(ui_code: str):
    """模块级不允许同名声明两次 —— 改代码时最容易犯的错，且症状是"改了没生效"。"""
    init_at = ui_code.index(INIT_SIG)
    module_scope = ui_code[:init_at]
    seen: dict[str, int] = {}
    dupes: list[str] = []
    for ln, stmt in _top_level_statements(module_scope, 0, len(module_scope)):
        if stmt.startswith(("let ", "var ", "const ", "function ", "async function ")):
            for n in _declared_names(stmt):
                if n in seen:
                    dupes.append(f"{n}（第 {seen[n]} 行 与 第 {ln} 行）")
                else:
                    seen[n] = ln
    assert not dupes, "模块级重复声明：\n  " + "\n  ".join(dupes)


def test_init_is_reachable_and_last_statement(ui_code: str):
    """启动序列必须在**文件末尾执行**（否则事件绑定会早于定义）。

    顺带确认它确实是"自执行"的，而不是被包在别的函数里 ——
    后者会让整页什么都不做，而且**不报错**。
    """
    init_at = ui_code.index(INIT_SIG)
    tail = ui_code[init_at:]
    assert tail.rstrip().endswith("})();"), "启动序列没有正常收尾（应为 `})();`）"
    assert ui_code.count(INIT_SIG) == 1, "启动序列出现了多次"


def test_renderer_js_has_no_reference_to_ui_only_names():
    """渲染器（`m3_webgpu.js`）是**独立模块**，不能引用界面里的名字。

    它由 `/api/gpuasset` 单独提供给浏览器，与 `m3_ui.html` 是两个执行环境。
    ⚠️ 若它引用了界面里的 `P` / `busy` / `$` 之类，浏览器里会 ReferenceError ——
    而在 Node 里做语法检查**发现不了**（`new Function()` 只查语法，不查自由变量）。
    """
    js = _blank_noncode(_read_js(JS))
    ui_only = {"$", "P", "busy", "sceneStale", "renderPreview", "applyZoom",
               "toast", "busyBadge", "qs", "logicalSize", "DEFAULTS"}
    used = {w for w in re.findall(r"\b([A-Za-z_$][\w$]*)\b", js)} & ui_only
    assert not used, f"渲染器引用了界面里的名字：{sorted(used)}"
