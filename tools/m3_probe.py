"""M3 服务端接口验收 —— 逐个打真实 HTTP 请求。

为什么要写成文件而不是一行 ``-c``：跑本地 HTTP 需要清掉环境里的 http_proxy
（本机有代理会拦截 localhost），而且逻辑一长在 shell 里就容易被引号搞坏
（HANDOFF 里记过这个坑）。

用法::

    python tools/m3_probe.py                 # 默认测 http://127.0.0.1:8770
    python tools/m3_probe.py --port 8899
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

# ⚠️ 必须在建 opener 之前清掉，否则本机代理会把 127.0.0.1 的请求也截走
for _k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
    os.environ.pop(_k, None)

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ASSET = "ref04_lain_room.jpg"
# 预览网格 160 让测试跑得快（真实使用是 240；只影响速度不影响正确性）
BASE_Q = f"asset={ASSET}&grid_long=160&work_long=640&aspect=native&seconds=3&fps=30"


def get(path: str, timeout: float = 240.0):
    with OPENER.open("http://127.0.0.1:" + str(PORT) + path, timeout=timeout) as r:
        body = r.read()
        return r.status, r.headers.get("Content-Type", ""), body


def post(path: str, obj):
    req = urllib.request.Request(
        "http://127.0.0.1:" + str(PORT) + path,
        data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(req, timeout=240.0) as r:
        return r.status, json.loads(r.read())


def show(tag, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {tag:34s} {detail}")
    return ok


def main() -> int:
    global PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    PORT = args.port

    results = []

    print("── 静态与素材 ──")
    st, ct, body = get("/")
    results.append(show("GET /  (界面)", st == 200 and b"<!DOCTYPE html>" in body,
                        f"status={st} {len(body)} bytes"))
    results.append(show("界面含关键控件", all(k in body for k in
                        (b'id="rail"', b'id="frame"', b'id="readout"', b'id="btnExport"'))))

    st, _, body = get("/api/assets")
    d = json.loads(body)
    n = len(d["assets"])
    results.append(show("GET /api/assets", st == 200 and n >= 10, f"{n} 张素材"))

    print("\n── 单帧（拖动滑杆的即时反馈）──")
    t0 = time.perf_counter()
    st, ct, body = get(f"/api/still?{BASE_Q}&t=0&scale=3")
    dt1 = (time.perf_counter() - t0) * 1000
    results.append(show("GET /api/still  首次", st == 200 and ct == "image/png" and body[:4] == b"\x89PNG",
                        f"{dt1:.0f} ms  {len(body)} bytes"))
    t0 = time.perf_counter()
    st, _, body2 = get(f"/api/still?{BASE_Q}&t=0&scale=3")
    dt2 = (time.perf_counter() - t0) * 1000
    results.append(show("GET /api/still  命中缓存", body2 == body, f"{dt2:.0f} ms（缓存应当快很多）"))

    st, _, body3 = get(f"/api/still?{BASE_Q}&t=0.5&scale=3")
    results.append(show("不同 t 给出不同帧", body3 != body))

    print("\n── 整段预览 + 自检统计 ──")
    t0 = time.perf_counter()
    st, _, body = get(f"/api/preview?{BASE_Q}&preview_fps=8")
    dt = time.perf_counter() - t0
    st_ = json.loads(body)
    if "error" in st_:
        results.append(show("GET /api/preview", False, st_["error"]))
    else:
        results.append(show("GET /api/preview", st == 200, f"{dt:.1f} s  {st_['n_frames']} 帧"))
        results.append(show("循环逐位闭合", st_.get("loop_ok") is True,
                            f"最大通道差={st_.get('loop_diff')}"))
        results.append(show("颜色全在共用色板内", st_.get("colors_outside") == 0,
                            f"越界={st_.get('colors_outside')} 色板={st_.get('palette_size')}"))
        results.append(show("预览时长 == 成片时长",
                            abs(st_.get("seconds", 0) - 3.0) < 1e-9,
                            f"seconds={st_.get('seconds')} preview_fps={st_.get('preview_fps')}"))
        m = st_["motion"]
        results.append(show("动画确实在动", m["adj_mean"] > 0.05,
                            f"相邻帧差 {m['adj_min']:.3f}~{m['adj_max']:.3f}"))
        a = st_["amplitude"]
        results.append(show("幅度分档合理", 0 <= a["static"] <= 1 and a["strong"] > 0,
                            f"静止 {a['static']*100:.0f}% 明显 {a['strong']*100:.0f}% max {a['max']}"))
        results.append(show("返回网格/输出尺寸", st_.get("out_size") and st_.get("scale"),
                            f"网格 {st_.get('grid')} ×{st_.get('scale')} -> {st_.get('out_size')}"))
        results.append(show("返回光源与雾色", len(st_.get("light_xy_used", [])) == 2,
                            f"光源 {[round(v,2) for v in st_.get('light_xy_used', [])]} "
                            f"雾色 {[round(v,3) for v in st_.get('fog_color', [])]}"))

    print("\n── 逐帧取图（浏览器端播放）──")
    nf = st_.get("n_frames", 8) if "error" not in st_ else 8
    sizes = []
    for i in (0, 1, max(0, nf // 2), nf - 1, 0):
        st2, ct2, b = get(f"/api/frame?{BASE_Q}&preview_fps=8&i={i}&scale=2")
        sizes.append((i, st2, len(b), b[:4] == b"\x89PNG"))
    results.append(show("GET /api/frame 各帧可用", all(x[1] == 200 and x[3] for x in sizes),
                        " ".join(f"i={i}:{n}B" for i, _, n, _ in sizes)))
    st2, _, b0 = get(f"/api/frame?{BASE_Q}&preview_fps=8&i=0&scale=2")
    st2, _, bN = get(f"/api/frame?{BASE_Q}&preview_fps=8&i={nf}&scale=2")
    results.append(show("frame(0) == frame(N)（图像层）", b0 == bN,
                        "同一张图说明循环无接缝"))

    print("\n── 参数扫描 ──")
    st, ct, body = get(f"/api/sweep?{BASE_Q}&key=density&values=0.4,0.7,1.1")
    results.append(show("GET /api/sweep", st == 200 and body[:4] == b"\x89PNG",
                        f"{len(body)} bytes"))

    print("\n── 参数存取 ──")
    st, d = post("/api/params", {"name": "_m3_probe", "params": {"density": 0.75, "colors": 24}})
    results.append(show("POST /api/params 存", d.get("ok") is True, str(d)))
    st, _, body = get("/api/params?name=_m3_probe")
    d = json.loads(body)
    results.append(show("GET /api/params 取", d.get("params", {}).get("density") == 0.75,
                        json.dumps(d.get("params", {}), ensure_ascii=False)))
    st, _, body = get("/api/params")
    d = json.loads(body)
    results.append(show("GET /api/params 列表", "_m3_probe" in d.get("saved", []),
                        f"{len(d.get('saved', []))} 个存档"))
    try:
        post("/api/params", {"name": "../evil", "params": {}})
        results.append(show("非法参数名被拒", False, "竟然接受了"))
    except urllib.error.HTTPError as e:
        results.append(show("非法参数名被拒", e.code == 400, f"status={e.code}"))

    print("\n── 全量出图（后台任务）──")
    payload = {"params": {"grid_long": 160, "work_long": 640, "aspect": "native",
                          "seconds": 1, "fps": 8, "dust_count": 60},
               "asset": ASSET, "format": "webp", "grid_long": 160, "work_long": 640}
    st, d = post("/api/export", payload)
    jid = d.get("job")
    results.append(show("POST /api/export 起任务", bool(jid), str(d)))
    if jid:
        state = "running"
        last = {}
        for _ in range(240):
            _, _, body = get(f"/api/job?id={jid}")
            last = json.loads(body)
            state = last.get("state")
            if state != "running":
                break
            time.sleep(0.5)
        results.append(show("任务完成", state == "done", last.get("msg", "")))
        files = last.get("files") or []
        results.append(show("产出文件存在", all(Path(f).exists() for f in files) and bool(files),
                            " ".join(Path(f).name for f in files)))
        stt = last.get("stats") or {}
        results.append(show("出图统计带自检", stt.get("loop_ok") is True,
                            f"循环差={stt.get('loop_diff')} 帧={stt.get('n_frames')}"))

    print("\n── 错误处理 ──")
    try:
        get("/api/nope")
        results.append(show("未知路由返回 404", False, "没报错"))
    except urllib.error.HTTPError as e:
        results.append(show("未知路由返回 404", e.code == 404, f"status={e.code}"))
    try:
        # 故意用一个不存在的素材名（中文），验证服务端返回 404 而不是崩掉。
        # ⚠️ 中文用转义写：护栏 `test_tool_output_paths_are_ascii_literal`
        #    会扫 tools/ 下带扩展名的字符串字面量，要求产物文件名是 ASCII。
        get("/api/still?asset=" + urllib.parse.quote("\u4e0d\u5b58\u5728.jpg"))
        results.append(show("素材不存在返回 404", False, "没报错"))
    except urllib.error.HTTPError as e:
        results.append(show("素材不存在返回 404", e.code == 404, f"status={e.code}"))

    ok = sum(1 for r in results if r)
    print("\n" + "=" * 58)
    print(f"  {ok} / {len(results)} 项通过")
    print("=" * 58)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
