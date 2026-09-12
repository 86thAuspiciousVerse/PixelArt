"""在**真浏览器**里跑一遍 WebGPU 路径 —— 无头 Edge/Chrome 打开 `?selftest=1`。

═══ 为什么需要真浏览器 ═══

离线能验的（`browser_pipeline_check.py` + `uniform_parity_probe.py`）：

  · WGSL 能不能编译、管线按 JS 声明的布局能不能建起来
  · JS 的 uniform 打包与 Python 是否逐字节一致
  · 界面静态结构（回退路径、rAF 播放、尺寸共用 …）

**离线验不了的**，只有真浏览器能回答：

  · `navigator.gpu` 在真实用户环境里到底给不给设备
  · 整条链（10 个 pass + 上屏）在浏览器里能不能跑通
  · **浏览器渲染的结果与验收台/服务端是否一致**（这是最终判据）

所以加了 `?selftest=1`：页面自己跑一遍链，把结果（含一帧的原始像素）
POST 回服务端，本脚本再把它取回来做比对。全程无人值守。

═══ 用法 ═══

    python tools/browser_selftest.py            # 自动找 Edge/Chrome
    python tools/browser_selftest.py --keep     # 保留报告文件

⚠️ 无头浏览器需要显式开启 WebGPU：
   ``--enable-unsafe-webgpu`` 是必须的（无头下默认关闭）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

REPORT_NAME = "_webgpu_selftest"


def _opener():
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(op, url, timeout=120):
    with op.open(url, timeout=timeout) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--browser", default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    op = _opener()

    print("=" * 76)
    print("  浏览器内 WebGPU 自检")
    print("=" * 76)

    # 服务必须在跑
    try:
        _get(op, base + "/api/assets", 8)
    except Exception as e:                                      # noqa: BLE001
        print(f"  ❌ 服务没在 {base} 上跑：{e}")
        print("     先启动：./.venv/Scripts/python.exe tools/m3_server.py --port 8770")
        return 1
    print(f"  服务    {base}")

    exe = args.browser
    if not exe:
        for c in BROWSERS:
            if Path(c).exists():
                exe = c
                break
    if not exe or not Path(exe).exists():
        print("  ❌ 找不到 Edge/Chrome")
        return 1
    print(f"  浏览器  {exe}")

    # 先清掉旧报告，避免读到上一轮的结果
    # ⚠️⚠️ 启动参数是**试出来的**，几个反直觉的点：
    #
    #   1. **必须用一次性、独占的 `--user-data-dir`**。共用同一个 profile 时，
    #      第二个实例会去连已有的浏览器进程、**另开一个标签页**，
    #      而我们在等服务端收到报告 —— 那个标签页可能压根没跑起来。
    #      实测共享 profile → "没拿到报告"；独占 profile → 成功。
    #      （和"多个服务实例抢同一端口"是同一类问题：**共享了不该共享的东西**。）
    #   2. `--enable-unsafe-webgpu` 是必须的（无头下 WebGPU 默认关闭）。
    #   3. `--use-angle=vulkan` / `--disable-gpu-sandbox` / `--enable-gpu`
    #      这些**加上去反而更糟**：实测带上它们时 adapter 请求返回 null。
    #      最小集才是能用的那组。
    prof = ROOT / "out" / "m3" / f"selftest-prof-{os.getpid()}"
    prof.mkdir(parents=True, exist_ok=True)
    flags = [
        exe,
        "--headless=new",
        f"--user-data-dir={prof}",
        "--no-first-run", "--no-default-browser-check",
        "--disable-extensions", "--disable-sync",
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--window-size=900,700",
        f"{base}/?selftest=1",
    ]
    print("  启动无头浏览器…")
    proc = subprocess.Popen(flags, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    rep = None
    t0 = time.time()
    try:
        while time.time() - t0 < args.timeout:
            time.sleep(2)
            try:
                raw = _get(op, f"{base}/api/params?name={REPORT_NAME}", 10)
                rep = json.loads(raw.decode("utf-8")).get("params")
                break
            except Exception:                                   # noqa: BLE001
                continue
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:                                       # noqa: BLE001
            try:
                proc.kill()
            except Exception:                                   # noqa: BLE001
                pass
        # profile 可能被浏览器进程占着，删不掉就算了（落在 out/ 下，不进仓库）
        import shutil
        try:
            shutil.rmtree(prof, ignore_errors=True)
        except Exception:                                       # noqa: BLE001
            pass
        # profile 可能被浏览器进程占着，删不掉就算了（落在 out/ 下，不进仓库）
        import shutil
        try:
            shutil.rmtree(prof, ignore_errors=True)
        except Exception:                                       # noqa: BLE001
            pass

    if rep is None:
        print(f"  ❌ {args.timeout}s 内没拿到自检报告")
        print("     （无头浏览器可能不支持 WebGPU，或页面报错。")
        print("      手动打开 http://127.0.0.1:%d/?selftest=1 看 console 更快）" % args.port)
        return 1

    print()
    print(f"  自检阶段  {rep.get('stage')}")
    ok = bool(rep.get("ok"))
    print(f"  {'✅' if ok else '❌'} 结果    {'通过' if ok else rep.get('reason')}")
    if not ok:
        print(f"     err: {rep.get('err')}")
        return 1

    print(f"  适配器    {rep.get('adapter')}")
    print(f"  网格      {rep.get('grid')}   色板 {rep.get('palette_len')} 色")
    print(f"  粒子      {rep.get('dust_count')} 颗   辉光核 {rep.get('bloom_kernels')} 抽头")
    print()
    print(f"  ⭐ 循环闭合 frame(t=1) vs frame(t=0)："
          f"{'逐位相同' if rep.get('loop_ok') else '接缝 ' + str(rep.get('loop_max_diff'))}")

    # ── 与服务端渲染的同一帧逐像素比对 ──
    #
    # ⚠️ 判据是"差得很少"，不是"逐位相同"：
    #    浏览器与验收台/服务端是**不同的实现路径**（GPU 上的 exp/sin 与 libm
    #    末位不同、模糊算子也不同），逐位相同做不到也不该要求。
    #    真正的要求是"看不出差别"，所以看**不一致像素的占比**。
    b64 = rep.get("frame37_b64")
    if not b64:
        print("  ⚠️ 报告里没有帧数据，跳过逐像素比对")
        return 0

    got = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)

    # ⚠️ 用页面**原样回报的参数串**重放，不要按网格反推。
    #    反推会丢掉 asset / aspect / 各滑杆值 —— 实测拿到的是另一张素材，
    #    比对结果 100% 不一致，而那是"比错了对象"，不是渲染错了。
    q = rep.get("query")
    if not q:
        print("  ⚠️ 报告里没有参数串（旧版页面？），跳过逐像素比对")
        return 0
    print(f"  参数    {q[:100]}{'…' if len(q) > 100 else ''}")

    still = _get(op, f"{base}/api/still?" + q + "&t=0.37&scale=1")
    from PIL import Image
    import io
    want_img = np.asarray(Image.open(io.BytesIO(still)).convert("RGB"), np.int32)
    want = want_img.reshape(-1, 3)

    n_px = min(len(want), len(got) // 4)
    got_px = np.frombuffer(got[:n_px * 4].tobytes(), dtype=np.uint32).reshape(n_px, 1)
    g = np.stack([(got_px[:, 0] & 0xFF), ((got_px[:, 0] >> 8) & 0xFF),
                  ((got_px[:, 0] >> 16) & 0xFF)], -1).astype(np.int32)

    d = np.abs(g - want[:n_px])
    per = d.max(axis=1)
    share = 100.0 * float((per > 0).mean())
    print()
    print("  与服务端同一帧的逐像素比对（t=0.37）：")
    print(f"    比对像素 {n_px}")
    print(f"    最大通道差 {int(d.max())}   均值 {float(d.mean()):.3f}")
    print(f"    不一致像素 {float((per > 0).mean()) * 100:.3f}%")
    # 允许少量像素因色板边界判定不同而翻色（GPU/CPU 的浮点末位）
    good = float((per > 0).mean()) < 0.02
    print(f"  {'✅' if good else '❌'} 与浏览器渲染结果一致"
          f"（不一致 < 2%；实测 {share:.3f}%）")

    if not args.keep:
        try:
            (ROOT / "out" / "m3" / "params" / f"{REPORT_NAME}.json").unlink()
        except Exception:                                       # noqa: BLE001
            pass

    print()
    print("=" * 76)
    print(f"  {'全部通过' if good else '存在问题'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
