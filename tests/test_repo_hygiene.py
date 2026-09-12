"""仓库卫生检查 —— 把踩过的坑固化成测试。

这些不是业务逻辑测试，而是"防止同类事故再次发生"的护栏。
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_SKIP_DIRS = {".git", ".venv", "models", "out", ".workbuddy", "__pycache__", ".pytest_cache"}


def _iter_repo_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(ROOT).parts
        if any(part in _SKIP_DIRS for part in parts):
            continue
        yield p


def test_ps1_scripts_are_pure_ascii():
    """⚠️ PowerShell 5.1 在文件没有 UTF-8 BOM 时，按**系统 ANSI 代码页**读取 .ps1。

    在中文 Windows 上就是 GBK —— 脚本里任何中文都会被误解码并直接报语法错误
    （典型症状：``表达式或语句中包含意外的标记"浣跨敤"``）。

    所以 scripts/*.ps1 必须保持纯 ASCII。不要"好心"把它们翻译成中文。
    """
    scripts = sorted((ROOT / "scripts").glob("*.ps1"))
    assert scripts, "scripts/ 下没有找到任何 .ps1"

    for path in scripts:
        raw = path.read_bytes()
        bad = [i for i, b in enumerate(raw) if b > 127]
        if bad:
            line = raw[: bad[0]].count(b"\n") + 1
            pytest.fail(
                f"{path.name} 含 {len(bad)} 个非 ASCII 字节"
                f"（第一处在第 {line} 行, offset={bad[0]}）。"
                "PowerShell 5.1 会因此解析失败，请改回 ASCII。"
            )


def test_no_non_ascii_filenames_in_repo():
    """文件名保持 ASCII，避免跨平台与编码问题。"""
    bad = []
    for p in _iter_repo_files():
        try:
            p.name.encode("ascii")
        except UnicodeEncodeError:
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"发现非 ASCII 文件名: {bad}"


def test_tool_output_paths_are_ascii_literal():
    """tools/ 下的**产物文件名**不要硬编码中文。

    注意只检查字符串字面量里的文件名，不要误伤带中文的 print 提示 ——
    按行粗筛会把 ``print("[ok] ... .png ... 放大")`` 也算进来。
    """
    import re

    pattern = re.compile(r"""['"]([^'"\n]*\.(?:png|webp|gif|jpg))['"]""")
    offenders = []
    for path in (ROOT / "tools").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            name = m.group(1)
            try:
                name.encode("ascii")
            except UnicodeEncodeError:
                lineno = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{lineno} -> {name}")
    assert not offenders, f"产物文件名里出现非 ASCII: {offenders}"


def test_python_sources_are_utf8_decodable():
    """Python 源文件必须能被 UTF-8 解码（注释里的中文是允许的）。"""
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").glob("*.py")) \
            + list((ROOT / "tests").glob("*.py")):
        p.read_text(encoding="utf-8")


def test_ps1_never_passes_a_variable_as_dash_c_script():
    """⚠️ Windows PowerShell 5.1 把参数传给原生 exe 时**不会转义内嵌双引号**。

    所以 ``& $py -c $multiLineScript`` 这种写法会被截断，python 报
    ``SyntaxError: '(' was never closed``。

    自检 / 校验逻辑一律写成 .py 文件，用 ``& $py <file>`` 调用。
    """
    import re

    pattern = re.compile(r"(?<![\w-])-c\s+\$")
    for path in sorted((ROOT / "scripts").glob("*.ps1")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                pytest.fail(
                    f"{path.name}:{lineno} 把变量当作 -c 的脚本参数传入。"
                    "PS 5.1 会截断多行/带引号的参数，请改成调用一个 .py 文件。"
                )
