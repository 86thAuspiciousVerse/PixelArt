"""★ 交互探针驱动 —— 把**用户会做的每个交互**在无头浏览器里做一遍。

启动体检（browser_boot_check.py）只证明「首帧能出来」；用户抱怨的是
**交互**：播放不动、滑杆没反应、切图卡住、标记消失。
这个工具打开 `?probe=ui`，页面自己把交互做一遍（判据 = 读回像素有没有变），
把报告 POST 回来，这里逐项断言。

用法：
    python tools/browser_ui_probe.py [等待秒] [--gpu]

不带 --gpu 时体检**默认后端**（现在是服务端）；`--gpu` 用 `?backend=gpu`
确定性开启 GPU —— 不靠浏览器 flag（无头 Chromium 默认就带 WebGPU，靠 flag 会验错对象）。
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


def _get(name: str):
    try:
        raw = OP.open(f"{BASE}/api/params?name={name}", timeout=10).read()
        return json.loads(raw).get("params")
    except Exception:                                           # noqa: BLE001
        return None


def run(url: str, wait: float, token: str):
    prof = ROOT / "out" / "m3" / f"uiprof-{os.getpid()}"
    prof.mkdir(parents=True, exist_ok=True)
    flags = [find_browser(), "--headless=new", f"--user-data-dir={prof}",
             "--no-first-run", "--no-default-browser-check",
             "--disable-extensions", "--disable-sync",
             "--enable-unsafe-webgpu", "--enable-features=Vulkan",
             "--window-size=1200,900", url]
    proc = subprocess.Popen(flags, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        while time.time() - t0 < wait:
            time.sleep(3)
            # ⚠️ 只认**最终**报告 `_ui_probe`（探针结束时直接 await POST 的那份）。
            #    不能看 `_page_state.uiProbe` —— step() 在**第一步**就把半成品
            #    挂上去了，看到非空就收工 = 每次只跑到第四步（实测连坑两次）。
            rep = _get("_ui_probe")
            if rep and rep.get("run") == token:
                time.sleep(1.0)
                return rep, _get("_page_state")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:                                       # noqa: BLE001
            proc.kill()
    return _get("_ui_probe"), _get("_page_state")


def judge(rep: dict, want_gpu: bool) -> list[str]:
    bad: list[str] = []
    by_name = {}
    for s in rep.get("steps", []):
        # ⚠️ 页面里 step() 把字段**合并进 step 顶层**（Object.assign），
        #    不是嵌在 "d" 下 —— 第一版读错键，四个步骤全显示成 {}，
        #    等于白测一遍（好在报告还在服务端，没白跑）。
        by_name.setdefault(s["name"], []).append(
            {k: v for k, v in s.items() if k != "name"})

    if not rep.get("ok"):
        bad.append(f"探针自己没跑完：{rep.get('err')}")
        return bad

    backend = (by_name.get("开始") or [{}])[0].get("backend")
    if want_gpu and backend != "gpu":
        bad.append(f"期望 GPU 后端，实际 {backend}")
    if not want_gpu and backend != "server":
        bad.append(f"期望服务端后端，实际 {backend}")

    for key in ("播放", "播放(改参后)"):
        play = (by_name.get(key) or [{}])[0]
        if not play.get("moving"):
            bad.append(f"{key}没有让画面动起来（btn={play.get('btn')}, "
                       f"playing={play.get('playing')}, err={play.get('err')}）")

    # 循环闭合读数：churn 之后不允许还停在 null（"检测中…"）
    loop = (by_name.get("循环闭合读数") or [{}])[0]
    if loop.get("loop_ok") is None or loop.get("loop_ok") == "无统计":
        bad.append(f"循环闭合读数卡在「检测中…」或无统计：{loop}")

    scrub = (by_name.get("进度条") or [{}])[0]
    if not scrub.get("changed"):
        bad.append("拖进度条画面没有变化")

    # ⚠️ 这几个参数改一步**本来就不一定（立刻）改变画面**，不算坏：
    #    · fog_sat / rays_sat 是「饱和度上限」—— 只在超过阈值时才起作用
    #    · seconds / fps 只改播放时长，同一 t 的画面不变
    #    · dust_count / colors 走素材刷新（260ms debounce + 服务端重算色板
    #      要渲 8 帧，往返 1~2s），探针 500ms 的窗口可能没等到
    #    · **parallax**：它的位移在 t=0 处恒为零（循环闭合要求），而通用滑杆
    #      循环正好停在 t=0 —— 看不到变化是**设计如此**。真正的验收在
    #      「视差(GPU)」那一步（会跳到位移峰值并确认没回退到服务端）。
    #    · rays_shaft / rays_cone_gain：通用循环只加一步（0→0.1 / 1.6→1.7），
    #      变化低于抖动噪声 —— 真正的验收在「光锥+光柱」专项步骤（shaft 1.9）。
    no_visual = {"fog_sat", "rays_sat", "seconds", "fps", "dust_count",
                 "colors", "parallax", "rays_dir", "rays_reach",
                 "rays_shaft", "rays_cone_gain"}
    for name, ds in by_name.items():
        if not name.startswith("滑杆 "):
            continue
        d = ds[0]
        key = name.replace("滑杆 ", "")
        if d.get("err"):
            bad.append(f"{name}：{d['err']}")
        elif d.get("changed") is False and not d.get("scene") and key not in no_visual:
            bad.append(f"{name}：改了一步画面**没有变化**")

    par = (by_name.get("视差(GPU)") or [{}])[0]
    if par:
        if not par.get("changed"):
            bad.append(f"视差在 GPU 上没让画面变化（{par}）")
        if want_gpu and par.get("backend") != "gpu":
            bad.append(f"视差把渲染踢回了 {par.get('backend')}（应当留在 GPU）")

    # 光柱：**锥内必须有强变化**（≥ 8/255 的像素要成片出现）
    par2 = (by_name.get("光锥+光柱") or [{}])[0]
    if par2:
        f2 = par2.get("开光柱后") or {}
        if (f2.get("max") or 0) < 40:
            bad.append(f"光柱几乎不可见（最大通道差 {f2.get('max')}，要求 ≥40）")
        elif (f2.get("strong_pct") or 0) < 1.0:
            bad.append(f"强变化像素只有 {f2.get('strong_pct')}%（光柱太弱或没生效）")
        if want_gpu and par2.get("backend") != "gpu":
            bad.append(f"光锥/光柱把渲染踢回了 {par2.get('backend')}（应当留在 GPU）")

    cone = (by_name.get("光锥") or [{}])[0]
    if cone and not cone.get("changed"):
        bad.append(f"光锥没让画面变化（{cone}）")

    mk = (by_name.get("光源标记") or [{}])[0]
    if not mk.get("visible"):
        bad.append(f"光源标记不可见（{mk}）")

    sw = (by_name.get("切换素材") or [{}])[0]
    ms = sw.get("ms")
    if ms is None:
        bad.append(f"切换素材没有等到画面变化（to={sw.get('to')}）")
    elif ms > 60000:
        bad.append(f"切换素材耗时 {ms/1000:.0f}s（>60s）")
    return bad


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_gpu = "--gpu" in sys.argv
    wait = float(argv[0]) if argv else (600.0 if want_gpu else 420.0)

    token = f"{os.getpid()}-{int(time.time())}"
    sep = "&" if "?" in BASE else "?"
    url = f"{BASE}/?bootcheck={token}&probe=ui" + ("&backend=gpu" if want_gpu else "")

    print(f"交互探针（期望后端：{'gpu' if want_gpu else 'server（默认）'}）")
    print(f"  URL: {url}")
    rep, st = run(url, wait, token)
    if not rep:
        print("\n❌ 没拿到探针报告")
        if st:
            print("   页面状态 steps:",
                  json.dumps(st.get("steps", [])[-4:], ensure_ascii=False))
            print("   页面异常:", json.dumps(st.get("errors", [])[:3],
                                             ensure_ascii=False))
        return 2

    import hashlib
    cur = hashlib.sha256((ROOT / "tools" / "m3_wgsl_uniforms.js").read_bytes()).hexdigest()[:8]
    print(f"\n── 探针步骤（{len(rep.get('steps', []))} 项，后端 {rep.get('backend')}）──")
    print(f"  服务端当前 m3_wgsl_uniforms 指纹: {cur}")
    for s in rep.get("steps", []):
        d = {k: v for k, v in s.items() if k != "name"}
        print(f"  {s['name']:10s} {json.dumps(d, ensure_ascii=False)[:120]}")

    bad = judge(rep, want_gpu)
    print("\n" + "═" * 70)
    if bad:
        print(f"❌ {len(bad)} 项不通过：")
        for b in bad:
            print(f"  · {b}")
        return 1
    print("✅ 交互全部有效（播放 / 进度条 / 各滑杆 / 光源标记 / 切换素材）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
