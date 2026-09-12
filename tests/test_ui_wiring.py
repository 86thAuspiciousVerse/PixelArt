"""界面接线不变量 —— 参数从滑杆到渲染调用，中间有 **5 条**可能断掉的路径。

这个文件专门守住它们，因为每一类都是真实踩过的坑，而且**全都不报错**：

  ① 控件拼错字段名  → 服务端 from_query 逐个字段查，不认识的静默丢弃
  ② 界面 DEFAULTS 少字段 → `qs()` 遍历 Object.keys(P) 就不会带上它
  ③ 复合控件传 row.k（占位名）→ 下游拿它比对具体字段永远不匹配
  ④ 服务端 compose_frame 硬传 scene.* → TuneParams 的 override 从没被调用
  ⑤ 界面 frames[] 不失效 → 播放放的是旧参数渲的帧

其中 ③④⑤ 在 m3_uicheck 里也查，但那些检查不进 pytest；这里补上，
让 `pytest` 这一个入口就能挡住全部五类。

⚠️ 这些检查全部是**静态字符串/CSS 分析**，所以有个必须遵守的前置：
   **查之前要剥掉注释**。解释某个 bug 的注释里往往就写着那行旧代码，
   不剥注释的朴素检查会被自己的说明文字误伤（本仓库已踩过两次）。
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "tools" / "m3_ui.html"
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def html() -> str:
    return UI.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js_only(html: str) -> str:
    """剥掉注释后的 JS（含 HTML 里的 <script> 段）。"""
    m = re.search(r"<script>(.*?)</script>", html, re.S)
    js = m.group(1) if m else ""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    js = re.sub(r"^\s*//.*$", "", js, flags=re.M)
    js = re.sub(r"(?<![:/])//[^\n\"']*$", "", js, flags=re.M)
    return js


@pytest.fixture(scope="module")
def css_only(html: str) -> str:
    """剥掉注释后的 CSS。"""
    m = re.search(r"<style>(.*?)</style>", html, re.S)
    css = m.group(1) if m else ""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _fields() -> set[str]:
    from pixelart.tune import TuneParams

    return {f.name for f in dataclasses.fields(TuneParams)}


# ══════════════════════════════════════════════════════════════════
# ① 控件绑定的字段名必须真实存在
# ══════════════════════════════════════════════════════════════════
def test_every_control_binds_a_real_field(js_only: str):
    """控件写错字段名 → 滑杆能拖、数值会变，但服务端根本不认识，拖了没反应。

    这类 bug 的隐蔽之处：界面完全正常，只是"效果没出来"，
    很容易被误判成"算法不好"或"这个参数本来就影响小"。
    """
    ctrl = set(re.findall(r'\{k:"([A-Za-z0-9_]+)"', js_only))
    composites = {k for k in ctrl if k.startswith("_")}
    sub: set[str] = set()
    for grp in re.findall(r"_c:\s*\[([^\]]*)\]", js_only):
        sub |= set(re.findall(r'"(\w+)"', grp))
    sub |= set(re.findall(r'_auto:\s*"(\w+)"', js_only))

    bound = (ctrl - composites) | sub
    unknown = sorted(bound - _fields())
    assert not unknown, f"控件绑定了服务端不存在的字段（拖了不会有反应）：{unknown}"
    assert len(bound) >= 30, f"只绑定了 {len(bound)} 个字段，控件表可能没解析成功"


# ══════════════════════════════════════════════════════════════════
# ② 界面 DEFAULTS 必须与服务端字段完全一致
# ══════════════════════════════════════════════════════════════════
def test_ui_defaults_cover_every_param(js_only: str):
    """`qs()` 遍历 `Object.keys(P)` 组请求 —— P 里没有的字段就**永远不会被送出**。

    踩过的坑：加了 fog_sat / rays_spread 两个滑杆，却忘了往 DEFAULTS 里加字段。
    而 `range` 的 value 被设为 `undefined` 时浏览器取**量程中点**，
    恰好等于服务端的默认值 —— 于是界面看着完全正确，直到有人改了那个默认值才炸。
    """
    m = re.search(r"const DEFAULTS = \{(.*?)\n\};", js_only, re.S)
    assert m, "没找到界面的 DEFAULTS 对象（改名了？正则要同步）"
    ui_keys = set(re.findall(r"(\w+)\s*:", m.group(1)))

    srv = _fields()
    missing = sorted(srv - ui_keys)
    extra = sorted(ui_keys - srv)
    assert not missing, f"界面 DEFAULTS 缺字段（这些参数永远不会被发送）：{missing}"
    assert not extra, f"界面 DEFAULTS 有多余字段（服务端不认识）：{extra}"


# ══════════════════════════════════════════════════════════════════
# ③ 复合控件必须把真实字段名传给 onParamChange
# ══════════════════════════════════════════════════════════════════
def test_composite_controls_pass_real_field_name(html: str):
    """vec2 / color 这类复合控件的 `row.k` 是占位名（`_light` / `_fogColor`）。

    下游若拿 `row.k` 去比对 `"rays_x"`，**永远不匹配** ——
    实测症状就是"拖动 X/Y 滑杆，画面上的光源标记不动"。
    """
    assert "changedKey" in html, "onParamChange 没有接收真实字段名的参数"
    # vec2 的两个子滑杆必须传出自己的 key
    assert "onParamChange(r, false, key)" in html, "vec2 子滑杆没传真实字段名"
    assert "onParamChange(r, true, key)" in html
    # color 控件同理
    assert "onParamChange(r, false, cr)" in html, "color 控件没传真实字段名"


# ══════════════════════════════════════════════════════════════════
# ④ 服务端渲染入口的每个旋钮都要显式接上来
# ══════════════════════════════════════════════════════════════════
def test_compose_frame_accepts_every_cosmetic_override():
    """`compose_frame` 必须**显式接受**每个会被界面覆盖的化妆参数。

    这一类 bug 出现了 4 次：light_xy / fog_sat / fog_rgb / screen_falloff。
    共同形态都是：`TuneParams` 里字段齐全、`compose_kwargs` 也放进去了，
    但 `compose_frame` **硬用 `scene.*`**，把 override 绕过去 ——
    于是"参数定义了但没接进渲染调用"，拖了毫无反应且不报错。
    """
    import inspect

    from pixelart.pipeline import compose_frame

    sig = inspect.signature(compose_frame)
    for name in ("light_xy", "fog_color_override", "fog_sat", "screen_falloff",
                 "rays_color", "rays_sat"):
        assert name in sig.parameters, f"compose_frame 不接受 `{name}` —— 该旋钮会失效"

    # 而且**默认值不能是覆盖场景值**：None 才表示"用 scene 里的自动估计"
    assert sig.parameters["light_xy"].default is None
    assert sig.parameters["fog_color_override"].default is None


def test_compose_kwargs_carries_every_override():
    """`compose_kwargs` 必须把界面能调的旋钮全部带上，一个不漏。"""
    import numpy as np
    from PIL import Image

    from pixelart.tune import TuneParams, build_scene

    img = Image.fromarray(np.full((48, 64, 3), 90, np.uint8))
    p = TuneParams(grid_long=64, work_long=192)
    sc = build_scene(img, p)
    kw = p.compose_kwargs(sc)
    for name in ("light_xy", "fog_color_override", "fog_sat", "screen_falloff",
                 "rays_color", "rays_sat", "density", "power"):
        assert name in kw, f"compose_kwargs 漏了 `{name}`"

    # 自动模式：雾色/散射色 override 必须是 None（不能塞一个假值进去）
    assert kw["fog_color_override"] is None, "自动模式下不该传手动雾色"
    assert kw["rays_color"] is None, "自动模式下不该传手动散射色"


# ══════════════════════════════════════════════════════════════════
# ⑤ 界面 frames[] 必须随参数失效
# ══════════════════════════════════════════════════════════════════
def test_frames_list_is_invalidated_on_param_change(html: str, js_only: str):
    """用户报障：改了尘埃参数，静止帧变了，**一点播放就跳回预设效果**。

    根因是界面自己维护的 `frames[]` URL 列表只在 `renderPreview` 里构建，
    **参数变化时没有任何机制让它失效** —— 播放时放的是上一组参数渲的帧。
    """
    assert "framesValid" in html, "frames 没有有效性标记"
    assert "invalidateFrames()" in js_only, "参数变化时没有让 frames 失效"
    assert "!frames.length || !framesValid" in html, "播放前没检查 frames 是否对应当前参数"
    assert "!frameCount || !framesValid" in html, "play() 自身没拒绝放失效帧"


# ══════════════════════════════════════════════════════════════════
# 显示尺寸：绝不能回到"由 CSS / 渲染分辨率决定"
# ══════════════════════════════════════════════════════════════════
def test_stage_img_size_is_never_css_driven(css_only: str):
    """用户报障：调参时画面变小、播放时变大（半分辨率图不会被 CSS 放大）。

    ⚠️ 要扫**所有** `.stage img` 规则（含媒体查询里的）——
    残留一条 `.stage img{max-height:52vh}` 就足以在窄窗口下把画面压扁。
    """
    rules = re.findall(r"\.stage(?:box)?\s+img\s*\{([^}]*)\}", css_only, re.S)
    assert rules, "没找到 `.stage img` 规则（改名了？正则要同步）"
    bad = [r.strip() for r in rules if "max-width" in r or "max-height" in r]
    assert not bad, f"这些 img 规则又用 max-* 决定尺寸了：{bad}"


def test_zoom_is_integer_and_fetch_scale_matches_zoom(html: str):
    """像素画的显示倍率必须是**整数**（非整数缩放会让方块参差不齐）。

    并且取图倍率要**等于**显示倍率 —— 否则 PNG 像素与显示不是 1:1，
    要么发虚（拉大）、要么浪费带宽（渲大缩显示）。
    """
    assert "Math.max(1, Math.min(16" in html, "zoom 没有取整/夹紧"
    assert "scale: zoom" in html, "取图倍率不等于显示倍率"
    # ⚠️ /api/preview 也要带同一个 scale：服务端按它预编码帧进缓存，
    #    界面按同样的 scale 取帧才命中；不一致会让缓存全落空、白渲两遍。
    assert "preview_fps: fps, scale: zoom" in html, "preview 请求没带 scale → 缓存落空"


# ══════════════════════════════════════════════════════════════════
# 拖放上传：必须区分「外部文件拖入」与「页面内元素拖动」
# ══════════════════════════════════════════════════════════════════
def test_drag_drop_distinguishes_internal_from_external(html: str):
    """⚠️ 拖动页面内的元素绝不能触发上传。

    用户报障：把预览画面那张成图拖出去用（正常需求），
    页面上却弹出「松手以载入这张图片」的遮罩 —— 一松手就上传，极易误操作。

    三层判据：
      ① `internalDrag` 标记 —— 页面内拖动会触发 `dragstart`，外部拖入不会。
         这个判据**不依赖 `dataTransfer.types`**，最可靠。
      ② `types` 含 `"Files"` —— 外部文件拖入的唯一特征。
      ③ drop 阶段 `files.length > 0` —— 规范里 `dataTransfer.files` 只在 drop 时可读。

    ⚠️ 原实现 dragenter/dragover 判了类型、**drop 没判** —— 这是误上传的直接原因。
    """
    assert "internalDrag" in html, "没有区分内外拖动的标记"
    assert 'addEventListener("dragstart"' in html and "internalDrag = true" in html, \
        "dragstart 没有置位标记（外部拖入不会触发 dragstart，这是最可靠的判据）"
    assert "internalDrag = false" in html, "dragend 没有清除标记"
    assert "isExternalFileDrag" in html, "没有统一的'是不是外部文件'判定函数"


def test_drop_handler_itself_checks_the_type(html: str):
    """⚠️ 检查必须**只看 drop 处理器体**。

    第一版检查写成"全文件里找 `isExternalFileDrag(e)`"—— 而 dragenter/dragover
    里也有同样的字符串，所以**永远通过**。是变异测试把这个问题暴露出来的：
    把 drop 体里的判断改成 `if (false)` 后，检查依然全绿。

    > 教训：范围敏感的检查必须**先把范围切出来**，不能在全文件里搜字符串。
    """
    m = re.search(r'addEventListener\("drop".*?\n  \}\);', html, re.S)
    assert m, "没能定位到 drop 处理器（缩进/结构改了？正则要同步）"
    body = m.group(0)
    assert len(body) > 100, f"drop 体只有 {len(body)} 字符，看着不像完整处理器"
    assert "isExternalFileDrag(e)" in body, "drop 体里没有做类型判定 —— 又会误上传"
    assert "!files.length" in body, "drop 体里没检查 files 是否为空"
    # 内部拖动必须到此为止（不上传）
    assert "if (internalDrag) return;" in body, "drop 体里没有挡住内部拖动"
    # ⚠️ preventDefault 必须排在**分派之前**：不阻止默认行为时浏览器会
    #    按拖入内容导航（内部拖动 → 图片 URL；外部文件 → 打开文件），
    #    两种都会把调参界面冲掉。原先只有部分分支做了，漏一条就白屏。
    assert body.index("e.preventDefault()") < body.index("internalDrag"), \
        "preventDefault 没有排在分派之前 —— 某些分支会漏掉，导致页面被导航掉"


def test_dropzone_uses_timestamp_not_depth_counter(html: str):
    """⚠️ 遮罩显示不能用 `depth++` / `--depth` 计数。

    跨子元素边界时 dragenter/dragleave 的次数并不配平，计数会漂移，
    导致遮罩闪烁或残留。改成 **dragover 刷新时间戳**：每次 dragover 重置一个
    短定时器，停 260ms 没有新事件就自动隐藏 —— 不依赖事件配平，天然免疫。

    ⚠️ 判据要按**变量声明**抓（`let depth`），不能只找 `--depth` / `depth++`
    这两个具体写法 —— 换个名字就绕过去了（本仓库的检查第一版就是这样漏的）。
    """
    body = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert not re.search(r"\b(?:let|var|const)\s+depth\b", body), \
        "又出现了 depth 计数器 —— 遮罩会在跨元素边界时闪烁/残留"
    assert "dzTimer" in html and "showDropzone" in html and "hideDropzone" in html, \
        "没有时间戳式遮罩管理"


# ══════════════════════════════════════════════════════════════════
# 拖放判定的**行为**验证（不是静态检查，而是把真代码跑起来）
# ══════════════════════════════════════════════════════════════════
def test_drag_logic_behaviour_matrix():
    """把界面里拖放那段**真代码**提取出来，用 node 跑事件矩阵。

    静态检查只能证明"代码里写了判据"，证明不了"事件顺序下行为正确"。
    拖放恰恰是**依赖事件顺序**的：dragstart 有没有先到、drop 时
    `dataTransfer.files` 是否可读、遮罩在 drop 前是否亮过 —— 这些都是运行时事实。

    装 Chromium 实测代价太高（~500MB + npm 环境不稳），而这里的判定逻辑
    是纯 JS，提取出来在 node 里跑同样能覆盖，且跑得快（<1s）。

    ⚠️ 关键场景是 **B：内部拖动但 `dataTransfer.types` 意外含 `Files`**。
    真实浏览器拖 `<img>` 时 types 不含 Files，但那是**浏览器行为**、不保证永远如此。
    所以判据不能只靠 types —— `internalDrag`（dragstart 标记）是兜底：
    外部拖入**永远不会**在文档里触发 dragstart。
    """
    import subprocess

    probe = ROOT / "tools" / "drag_logic_probe.py"
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("环境里没有可用的 node")

    r = subprocess.run([sys.executable, str(probe)],
                       capture_output=True, text=True, encoding="utf-8")
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, f"拖放行为矩阵不通过：\n{out[-1500:]}"
    assert "7 / 7 通过" in out, f"场景数不对，可能有场景没跑到：\n{out[-1500:]}"


# ══════════════════════════════════════════════════════════════════
# 元测试：验证"这些检查真的会失败"
# ══════════════════════════════════════════════════════════════════
def test_mutation_actually_breaks_the_behaviour_check():
    """⚠️ 元测试 —— 确认上面的行为检查**不是空转的**。

    一个永远通过的检查比没有检查更糟：它给人虚假的安全感。
    本仓库已经靠变异测试抓出过**两次**假阳性检查：
      · "drop 里判了类型" —— 全文件搜索，而 dragenter/dragover 里也有同样字符串
      · "没用 depth 计数" —— 只找 `--depth`，改成 `let depth = 0` 就绕过去了

    这里注入最致命的一个变异（去掉 dragstart 标记 = 内外拖动分不出来），
    断言行为探针**必须失败**。如果哪天有人把判据改松、或把这段逻辑搬走，
    这个测试会立刻报红。
    """
    import subprocess

    node_ok = True
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        node_ok = False
    if not node_ok:
        pytest.skip("环境里没有可用的 node")

    mutator = ROOT / "tools" / "ui_mutate.py"
    probe = ROOT / "tools" / "drag_logic_probe.py"
    backup = ROOT / "out" / "mutate" / "ui.mutbak"
    original = UI.read_text(encoding="utf-8")
    had_backup = backup.exists()
    prev_backup = backup.read_text(encoding="utf-8") if had_backup else None

    try:
        inj = subprocess.run([sys.executable, str(mutator), "2"],
                             capture_output=True, text=True, encoding="utf-8")
        assert inj.returncode == 0, f"注入变异失败：{inj.stdout}{inj.stderr}"
        assert "已注入" in inj.stdout, inj.stdout

        r = subprocess.run([sys.executable, str(probe)],
                           capture_output=True, text=True, encoding="utf-8")
        out = (r.stdout or "") + (r.stderr or "")
        assert r.returncode != 0, (
            "去掉 dragstart 标记后行为检查竟然还通过 —— 检查是空转的！\n"
            f"探针输出：\n{out[-1200:]}")
    finally:
        # 无论如何都要还原，别把变异留在工作区里
        UI.write_text(original, encoding="utf-8")
        if had_backup:
            backup.write_text(prev_backup, encoding="utf-8")
        else:
            backup.unlink(missing_ok=True)

    # 还原必须彻底
    assert UI.read_text(encoding="utf-8") == original, "变异没被完全还原"
