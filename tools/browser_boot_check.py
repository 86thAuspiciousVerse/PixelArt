"""★ 启动路径体检 —— 打开**普通页面**（不是 `?selftest=1`），验证用户真正走的那条路。

═══ 为什么必须有这个工具 ═══

用户报"画面一直是黑的"。查出来是**作用域**问题：GPU 那套状态与函数被声明在
启动序列 `init()` 里面，而 `renderPreview()` / `applyZoom()` / `stopPlay()`
是模块级的 —— 引用一律 `ReferenceError`，首帧在 `busy = true` 之后抛，
于是**之后每次渲染都被吞掉，画面永远黑**。

而当时 `tools/browser_selftest.py` 全绿。原因很刺眼：

    `?selftest=1` 会在 `init()` 里 `return`，用的全是在作用域内的名字 ——
    **它验的是它自己那条路，不是用户走的那条路。**

所以这个工具补的就是那一块：**不带任何开关打开页面**，等它自己跑完启动流程，
然后把页面回报的"启动进度 / 未捕获异常 / 作用域探测 / 首帧像素统计"拿过来断言。

═══ 判据（每一条都对应一种真实故障）═══

  ① 作用域探测里**没有任何 undefined**  ← 挡住了上面那个 ReferenceError
  ② **没有任何未捕获异常**              ← 静默失败的主要来源
  ③ 启动进度**走到"首图完成"**          ← 挡住"卡在中途"
  ④ 首帧**有内容**（非纯色、不是全黑）  ← 挡住"跑完了但画面是黑的"
  ⑤ 该可见的元素**真的可见**            ← 挡住"渲染对了但显示的是另一个元素"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8770"
for _k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
    os.environ.pop(_k, None)
OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]


def find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    sys.exit("找不到 Edge/Chrome")


def get_state() -> dict | None:
    try:
        raw = OP.open(f"{BASE}/api/params?name=_page_state", timeout=10).read()
        return json.loads(raw).get("params")
    except Exception:                                           # noqa: BLE001
        return None


def run_browser(url: str, wait: float, enable_webgpu: bool, token: str):
    prof = ROOT / "out" / "m3" / f"bootprof-{os.getpid()}"
    prof.mkdir(parents=True, exist_ok=True)
    # ⚠️ 最小参数集 —— 多一个 flag（angle/vulkan/sandbox）反而会让适配器请求
    #    返回 null，见 docs/arch-02-webgpu.md §7.5。
    flags = [find_browser(), "--headless=new", f"--user-data-dir={prof}",
             "--no-first-run", "--no-default-browser-check",
             "--disable-extensions", "--disable-sync",
             "--window-size=1200,800"]
    if enable_webgpu:
        flags += ["--enable-unsafe-webgpu", "--enable-features=Vulkan"]
    flags += [url]

    proc = subprocess.Popen(flags, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    t0 = time.time()
    last = None
    try:
        while time.time() - t0 < wait:
            time.sleep(2.5)
            st = get_state()
            # ⚠️ 只认**本轮 token** 的报告。上一轮的旧报告仍然存在（我们刻意
            #    不删东西 —— 删除是危险操作），不按 token 过滤就会把它当成
            #    这次的结果：典型的"拿错基线"，而这个坑这个项目已经踩过一次。
            if not st or st.get("run") != token:
                continue
            last = st
            steps = st.get("steps") or []
            if steps and steps[-1].get("step") == "首图完成":
                # ⭐ 首帧优先后，**服务端**路径的「首图完成」只是第一步，
                #    还要等后台整段（PAGE_STATE.bg）才叫完整启动；
                #    **GPU 路径没有这一步** —— 帧是按需现渲的，看到首帧即完成。
                bdone = st.get("bg") or (st.get("final") or {}).get("backend") == "gpu"
                if bdone:
                    time.sleep(1.5)          # 让最后的状态落盘
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:                                       # noqa: BLE001
            proc.kill()
    st = get_state()
    return st if (st and st.get("run") == token) else last


def check(st: dict | None, expect_backend: str | None) -> list[str]:
    """返回问题列表（空 = 全过）。每条都说明"这挡住了什么"。"""
    bad: list[str] = []
    if not st:
        return ["没拿到页面报告 —— 页面可能根本没跑起来，或服务端没收到 POST"]

    # ① 作用域
    scope = st.get("scope") or {}
    if not scope:
        bad.append("报告里没有作用域探测（页面版本太旧？）")
    else:
        miss = sorted(k for k, v in scope.items() if v == "undefined")
        thr = {k: v for k, v in scope.items() if str(v).startswith("THROW")}
        if miss:
            bad.append("这些名字**不在**模块作用域里，模块级函数引用它们会 "
                       f"ReferenceError：{miss}")
        if thr:
            bad.append(f"作用域探测本身抛错：{thr}")

    # ② 未捕获异常
    errs = st.get("errors") or []
    if errs:
        bad.append(f"有 {len(errs)} 条未捕获异常：")
        bad += [f"      [{e['kind']}] {e['msg'][:160]}" for e in errs[:4]]

    # ③ 走到"首图完成"
    steps = st.get("steps") or []
    done = [s for s in steps if s.get("step") == "首图完成"]
    if not done:
        last = steps[-1].get("step") if steps else "(一步都没有)"
        bad.append(f"启动没走到「首图完成」，最后一步是：{last}")
    else:
        extra = [s for s in steps if s.get("step") == "initBackend done"]
        if expect_backend == "gpu" and extra and not extra[0]["d"].get("gpuReady"):
            bad.append(f"期望 GPU 后端，但初始化没成功：{extra[0]['d']}")

    # ④ 首帧有内容
    c = st.get("canvas")
    if not c:
        bad.append("没有首帧像素统计（psPaintCheck 没跑到）")
    else:
        if c.get("err"):
            bad.append(f"首帧检查自己报错：{c['err']}")
        if c.get("uniq", 0) <= 1:
            bad.append(f"首帧是**纯色**（唯一色 {c.get('uniq')} 种，"
                       f"均值 {c.get('mean')}）—— 就是「全黑」那个症状")
        if c.get("dark_pct", 100) > 60:
            bad.append(f"首帧 {c['dark_pct']}% 的像素是纯黑（阈值 60%）")
        if c.get("mean", 0) <= 1:
            bad.append(f"首帧均值只有 {c.get('mean')}，基本是黑的")
        # ⑤ 可见性：该显示的那个元素必须在显示
        if expect_backend == "gpu":
            if c.get("canvas_display") == "none":
                bad.append("GPU 模式下画布是 display:none")
            if c.get("img_display") != "none":
                bad.append("GPU 模式下服务端的 <img> 还在显示（两个元素叠着）")
        elif expect_backend == "server":
            if c.get("img_display") == "none":
                bad.append("服务端模式下 <img> 是 display:none")
            if not c.get("img_complete") or not c.get("img_natural", [0])[0]:
                bad.append("服务端模式下 <img> 没有加载出内容")
    # ⑥ 收尾状态（**不能拿渲染中途的快照来判**，见下面注释）
    fin = st.get("final")
    if not fin:
        bad.append("报告里没有收尾状态（renderDone 没跑到？）")
    else:
        if fin.get("busy"):
            bad.append("收尾后 busy 仍为 true —— 之后所有渲染都会被吞掉")
        if fin.get("preview_disabled"):
            bad.append("收尾后「渲染预览」按钮仍是禁用（busy 没收尾）")
        # ⭐ 首帧优先后，「播放」在后台整段就绪**之前**禁用是**正确**的；
        #    体检要看的是 bg.ok 之后它是否亮起。
        bg = st.get("bg")
        if bg is None and fin.get("backend") == "gpu":
            # GPU 路径没有"后台整段"——frames 是按需现渲的，不该要求这个事件
            if fin.get("play_disabled"):
                bad.append("GPU 模式下「播放」按钮是禁用（应当可用）")
        elif not bg:
            bad.append("后台整段没有就绪事件（PAGE_STATE.bg 缺失，超时？）")
        elif not bg.get("ok"):
            bad.append(f"后台整段渲染失败：{bg.get('err')}")
        else:
            if fin.get("play_disabled"):
                bad.append("整段就绪后「播放」按钮仍是禁用")
            if fin.get("n_frames", 0) < 2 and bg.get("n_frames", 0) < 2:
                bad.append(f"整段就绪后帧数异常：{fin.get('n_frames')} / bg {bg.get('n_frames')}")
            print(f"  （后台整段就绪，{bg.get('n_frames')} 帧）")
        if expect_backend and fin.get("backend") != expect_backend:
            bad.append(f"实际后端是 {fin.get('backend')}，期望 {expect_backend}")
    return bad


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
# ⭐ 默认后端现在是**服务端**（2026-09-12 起，GPU 交互没修完之前只作为可选项）。
#    要体检 GPU 路径用 --gpu（页面会带 ?backend=gpu 确定性开启）。
    want_gpu = "--gpu" in sys.argv
    no_webgpu = not want_gpu
    wait = float(argv[0]) if argv else (300.0 if no_webgpu else 90.0)
    url = argv[1] if len(argv) > 1 else BASE + "/"
    expect = "gpu" if want_gpu else "server"

    # 一次性 token：用它把本轮的页面报告与旧报告区分开。
    # 回退路径**用 URL 参数锁定**，不靠"去掉某个浏览器 flag" ——
    # 实测无头 Edge 这版默认就带 WebGPU，靠 flag 会验错对象。
    token = f"{os.getpid()}-{int(time.time())}"
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}bootcheck={token}" + ("&backend=gpu" if want_gpu else "")

    print(f"启动路径体检（期望后端：{expect}）")
    print(f"  URL: {url}   等待上限: {wait:.0f}s")
    st = run_browser(url, wait, enable_webgpu=not no_webgpu, token=token)
    if not st:
        print("\n❌ 没拿到页面报告")
        return 2

    print(f"\n── 启动进度（{len(st.get('steps', []))} 步）──")
    for s in st.get("steps", []):
        extra = f"   {json.dumps(s['d'], ensure_ascii=False)}" if s.get("d") else ""
        print(f"  {s['ms']:7d} ms  {s['step']}{extra}")

    c = st.get("canvas") or {}
    if c:
        print("\n── 首帧 ──")
        print(f"  唯一色 {c.get('uniq')}  纯黑占比 {c.get('dark_pct')}%  "
              f"均值 {c.get('mean')}  范围 [{c.get('min')}, {c.get('max')}]")
        print(f"  显示：画布 {c.get('canvas_display', '—')} / "
              f"图 {c.get('img_display', '—')}")

    bad = check(st, expect)
    print("\n" + "═" * 70)
    if bad:
        print(f"❌ {len(bad)} 项不通过：")
        for b in bad:
            print(f"  · {b}")
        return 1
    print("✅ 启动路径全通（作用域 / 异常 / 进度 / 首帧内容 / 可见性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
