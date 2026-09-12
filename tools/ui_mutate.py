"""变异测试工具 —— 注入/还原界面代码里的缺陷，用来验证检查"真的会失败"。

⚠️ 踩过的坑：`restore` 从备份文件恢复，而备份可能是**上一轮**留下的 ——
   于是 `mut.py restore` 会把文件回滚成更早的版本，**静默丢掉后来的编辑**
   （本仓库真的发生过：一次"统一 preventDefault"的修改被回滚，
   半天后才发现检查报红其实是文件被还原了，不是代码有问题）。

   修正：变异时**总是先用当前文件刷新备份**，所以 `restore` 一定回到
   "本次变异之前"的状态，不可能跨轮次回滚。备份落 `out/`（已 gitignore）。

用法：
    python tools/ui_mutate.py 123      # 注入变异 1/2/3（自动先备份当前状态）
    python tools/ui_mutate.py restore  # 还原到本次变异之前
"""
import hashlib
import pathlib
import sys

UI = pathlib.Path("tools/m3_ui.html")
BAK = pathlib.Path("out/mutate/ui.mutbak")

MUTS = {
    # ① drop 处理器里不做类型判定（回到"误上传"的写法）
    "1": ('    if (!isExternalFileDrag(e)){\n      // 外部拖入的**不是文件**',
          '    if (false){\n      // 外部拖入的**不是文件**'),
    # ② 去掉 dragstart 标记（内外拖动就分不出来了）
    "2": ('window.addEventListener("dragstart", () => { internalDrag = true; hideDropzone(); }, true);',
          '/* removed */'),
    # ③ 遮罩改回 depth 计数
    "3": ('let dzTimer = null;', 'let dzTimer = null; let depth = 0;'),
    # ④ drop 不检查 files 是否为空
    "4": ('    if (!files || !files.length) return;', '    // removed'),
    # ⑤ 去掉 isExternalFileDrag 里的 internalDrag 兜底
    "5": ('    if (internalDrag) return false;', '    if (false) return false;'),
    # ⑥ drop 里的 preventDefault 挪到分派之后（部分分支会漏掉）
    "6": ('    e.preventDefault();\n    hideDropzone();\n\n    if (internalDrag) return;',
          '    hideDropzone();\n\n    if (internalDrag) return;\n    e.preventDefault();'),
}


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:10]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "restore":
        if not BAK.exists():
            print("没有备份可还原（本工具只还原'本次变异之前'）")
            return 1
        UI.write_text(BAK.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"已还原 -> {md5(UI)}")
        return 0

    BAK.parent.mkdir(parents=True, exist_ok=True)
    s = UI.read_text(encoding="utf-8")
    BAK.write_text(s, encoding="utf-8")               # ⚠️ 总是刷新备份
    for k in mode:
        if k not in MUTS:
            print(f"未知变异 '{k}'")
            return 1
        old, new = MUTS[k]
        if old not in s:
            print(f"[错误] 变异 {k} 未命中 —— 源码结构变了，请同步 MUTS")
            return 1
        s = s.replace(old, new)
    UI.write_text(s, encoding="utf-8")
    print(f"已注入变异 {mode} -> {md5(UI)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
