"""拖放判定逻辑的行为验证 —— 把界面里的**真代码**提取出来跑事件矩阵。

为什么不直接读代码人工确认：拖放判定依赖事件顺序与 `dataTransfer` 的形态，
靠读代码很容易漏掉"某个组合下会误上传"。而装 Chromium 实测的代价
（~500MB 下载、npm 环境不稳）对一个逻辑判定来说太高。

做法：从 `tools/m3_ui.html` 里**原样提取**拖放那段代码（不是抄一遍），
注入一组最小 shim（window / $ / toast / uploadFile），然后用 node 跑场景矩阵。
因为是提取真代码，改了界面它就会跟着变 —— 不会像"抄一份逻辑"那样悄悄漂移。

⚠️ 关键场景是 **B：内部拖动但 `types` 意外含 `Files`**。
真实浏览器里拖动 `<img>` 时 `types` 不含 Files，但那是**浏览器行为**，
不保证永远如此。所以判据不能只靠 types —— `internalDrag` 标记（来自 dragstart）
是兜底：外部拖入**永远不会**在文档里触发 dragstart。
这个测试就是来证明那道兜底真的有效。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "tools" / "m3_ui.html"

#: 拖放段落：从取遮罩元素到 drop 处理器结束
_START = 'const dz = $("dropzone");'
_END = "uploadFile(files[0]);\n  });"

_SHIM = r"""
// ── 最小 shim：只为让提取出来的那段代码能跑 ──
const calls = { upload: [], toast: [], dzOn: false };
const dzEl = {
  classList: {
    add: () => { calls.dzOn = true; },
    remove: () => { calls.dzOn = false; },
  },
};
globalThis.$ = (id) => (id === "dropzone" ? dzEl : { classList: { add(){}, remove(){} } });
globalThis.toast = (m) => { calls.toast.push(m); };
globalThis.uploadFile = (f) => { calls.upload.push(f ? f.name : null); };

const H = {};
globalThis.window = {
  addEventListener: (t, h) => { (H[t] = H[t] || []).push(h); },
};

// 模拟一次事件
function evt(opts) {
  const o = Object.assign({ types: [], files: [], prevented: false }, opts || {});
  return {
    prevented: false,
    preventDefault() { this.prevented = true; },
    dataTransfer: { types: o.types, files: o.files, dropEffect: "" },
  };
}
function fire(type, e) {
  (H[type] || []).forEach((h) => h(e));
  if (type === "drop") lastDrop = e;         // 留着查 preventDefault
  return e;
}
function reset() {
  calls.upload.length = 0; calls.toast.length = 0; calls.dzOn = false;
  dzSeen = false; lastDrop = null;
}

const IMG_TYPES = ["text/uri-list", "text/html", "text/plain"];
const FILE_TYPES = ["Files"];
const fakeFile = (n) => ({ name: n, type: "image/png" });

let dzSeen = false;
let lastDrop = null;
const results = [];
function case_(name, fn) {
  reset();
  let note = "";
  try { note = fn() || ""; } catch (e) { note = "EXC:" + e.message; }
  results.push({ name, uploaded: calls.upload.slice(), toasted: calls.toast.slice(),
                 dzOn: calls.dzOn, dzSeen,
                 prevented: lastDrop ? !!lastDrop.prevented : null, note });
}
"""

_CASES = r"""
// ── A. 拖动画面的成图（页面内拖动）→ 绝不能上传 ──
case_("A 内部拖动 img（types 无 Files）", () => {
  fire("dragstart", evt({ types: IMG_TYPES }));
  fire("dragenter", evt({ types: IMG_TYPES }));
  fire("dragover", evt({ types: IMG_TYPES }));
  dzSeen = calls.dzOn;                       // drop 前采样
  fire("drop", evt({ types: IMG_TYPES }));
  fire("dragend", evt({}));
  return dzSeen ? "⚠️ 遮罩亮了（会误导用户）" : "遮罩未亮、未上传";
});

// ── B. ★核心兜底：内部拖动，但 types 意外含 Files ──
// 真实浏览器拖 img 时 types 不含 Files，但那是浏览器行为、不保证永远如此。
// 判据不能只靠 types —— internalDrag 必须兜住。
case_("B ★内部拖动但 types 含 Files（兜底）", () => {
  fire("dragstart", evt({ types: IMG_TYPES }));
  fire("dragenter", evt({ types: FILE_TYPES }));
  fire("dragover", evt({ types: FILE_TYPES }));
  dzSeen = calls.dzOn;                       // drop 前采样
  fire("drop", evt({ types: FILE_TYPES, files: [fakeFile("inner.png")] }));
  fire("dragend", evt({}));
  return dzSeen ? "⚠️ 遮罩亮了" : "types 像文件也不上传、不亮遮罩";
});

// ── C. 外部拖入真实文件 → 应当上传 ──
case_("C 外部文件拖入", () => {
  fire("dragenter", evt({ types: FILE_TYPES }));
  dzSeen = calls.dzOn;                       // drop 前采样（drop 里会 hide）
  fire("dragover", evt({ types: FILE_TYPES }));
  fire("drop", evt({ types: FILE_TYPES, files: [fakeFile("photo.jpg")] }));
  return dzSeen ? "遮罩已亮" : "⚠️ 遮罩没亮";
});

// ── D. 内部拖动结束后，标记必须被清掉，外部拖入仍能上传 ──
case_("D 内部拖动后再外部拖入（标记要清）", () => {
  fire("dragstart", evt({ types: IMG_TYPES }));
  fire("dragend", evt({}));
  dzSeen = calls.dzOn;
  fire("drop", evt({ types: FILE_TYPES, files: [fakeFile("after.jpg")] }));
  return "标记已清除，外部拖入应当可以上传";
});

// ── E. 从别的网页拖图片链接进来 → 不上传，但要给提示 ──
case_("E 网页图片链接（非文件）", () => {
  fire("dragenter", evt({ types: IMG_TYPES }));
  dzSeen = calls.dzOn;
  fire("drop", evt({ types: IMG_TYPES }));
  return (dzSeen ? "⚠️ 遮罩亮了" : "遮罩未亮") + "，应有提示";
});

// ── F. 外部拖入但 files 为空（规范上不该发生，防御性） ──
case_("F 外部拖入但 files 为空", () => {
  dzSeen = calls.dzOn;
  fire("drop", evt({ types: FILE_TYPES, files: [] }));
  return "不该上传 null";
});

// ── G. 拖入两个文件：只取第一个 ──
case_("G 拖入两个文件", () => {
  dzSeen = calls.dzOn;
  fire("drop", evt({ types: FILE_TYPES, files: [fakeFile("a.png"), fakeFile("b.png")] }));
  return "只应上传第一个文件";
});

console.log(JSON.stringify(results));
"""


def extract_drag_block(html: str) -> str:
    """从界面里原样提取拖放段落。

    原文结构（`bindUpload` 函数体的后半段）::

        function bindUpload(){          ← 不在提取范围内（不需要它的收尾 `}`）
          const dz = $("dropzone");     ← _START
          const internalDrag = false;
          ...
          window.addEventListener("drop", (e) => {
            ...
            uploadFile(files[0]);
          });                            ← _END 的结尾（drop 监听器自闭合）
        }

    ⚠️ 提取结果之后会被 `function wire(){ ... }` 包起来，而 wire 自带一个收尾 `}`。
    这里**不能再补任何括号** —— 试过补 `});`（语法错）和补 `}`（多一个右括号），
    都对不上：改动前先想清楚"哪一层括号由谁收"。
    """
    s = html.index(_START)
    e = html.index(_END, s) + len(_END)
    block = html[s:e]
    # 括号必须配平：block 内部所有监听器都是 `... });` 自闭合，不需要额外收尾
    assert block.count("{") == block.count("}"), (
        f"提取的拖放段落括号不配平（{{={block.count('{')} }}={block.count('}')}）—— "
        "源码结构改了？请同步 _START/_END")
    return block


def main() -> int:
    if not UI.exists():
        print(f"[错误] 找不到 {UI}")
        return 1
    html = UI.read_text(encoding="utf-8")
    try:
        block = extract_drag_block(html)
    except ValueError:
        print("[错误] 没能在界面里定位拖放段落 —— 结构改了？请同步本脚本的 _START/_END")
        return 1

    # 提取的代码是"函数体内的一段"，包一层函数给它作用域
    script = _SHIM + "\nfunction wire(){\n" + block + "\n}\nwire();\n" + _CASES

    proc = subprocess.run(["node", "--input-type=module", "-e", script],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print("[错误] node 执行失败：")
        print((proc.stderr or "")[-1500:])
        return 1

    try:
        rows = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:                                           # noqa: BLE001
        print("[错误] 解析结果失败，原始输出：")
        print(proc.stdout[-1200:])
        return 1

    # 期望：每个场景的 (是否上传, 上传的文件名, 遮罩亮否)
    #   A 不上传 / B 不上传 / C 上传 photo.jpg / D 上传 after.jpg
    #   E 不上传但有提示 / F 不上传 / G 只上传 a.png
    # (应上传的文件, drop 前遮罩应亮?, drop 应被 preventDefault?)
    #   ⚠️ 遮罩必须纳入断言：它正是用户看到的症状（"松手以载入这张图片"浮出来）。
    #      只断言"有没有上传"会漏掉——本仓库有**两道冗余防线**都能拦住上传，
    #      所以单点变异只让遮罩亮、上传仍被拦住，测试却全绿（实测如此）。
    #   ⚠️ preventDefault 也必须断言：drop 不阻止默认行为，浏览器会**导航到图片 URL**
    #      （预览图的 src 是 /api/still?...），整个调参界面被冲掉。
    expect = {
        "A 内部拖动 img（types 无 Files）": ([], False, True),
        "B ★内部拖动但 types 含 Files（兜底）": ([], False, True),
        "C 外部文件拖入": (["photo.jpg"], True, True),
        "D 内部拖动后再外部拖入（标记要清）": (["after.jpg"], False, True),
        "E 网页图片链接（非文件）": ([], False, True),
        "F 外部拖入但 files 为空": ([], False, True),
        "G 拖入两个文件": (["a.png"], False, True),
    }

    print("拖放判定行为矩阵（提取界面真代码，用 node 跑）")
    print()
    print(f"  {'场景':40s} {'上传':>10s} {'遮罩':>5s} {'阻止默认':>7s}  判定")
    print("  " + "-" * 80)
    bad = 0
    for r in rows:
        name = r["name"]
        up, want_dz, want_prev = expect.get(name, ([], False, True))
        problems = []
        if up != up and False:
            pass
        if r["uploaded"] != up:
            problems.append(f"上传应为 {up}，实为 {r['uploaded']}")
        if r["dzSeen"] != want_dz:
            problems.append(f"遮罩应{'亮' if want_dz else '不亮'}")
        if r["prevented"] is not None and r["prevented"] != want_prev:
            problems.append("没阻止默认行为（会导航掉页面）")
        if problems:
            bad += 1
        shown = ",".join(str(x) for x in r["uploaded"]) if r["uploaded"] else "—"
        pv = {True: "是", False: "否", None: "—"}[r["prevented"]]
        print(f"  {name:40s} {shown:>10s} {'是' if r['dzSeen'] else '否':>5s} "
              f"{pv:>7s}  {'OK' if not problems else 'XX'}")
        for pr in problems:
            print(f"      ✗ {pr}")

    print("  " + "-" * 80)
    print(f"  {len(rows) - bad} / {len(rows)} 通过")
    if bad:
        print()
        print("  ⚠️ 有场景不符合预期 —— 拖放判定又出问题了")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
