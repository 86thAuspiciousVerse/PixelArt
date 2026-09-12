"""M3 调参服务 —— 本地 localhost 的像素风转换器调参台。

设计原则：**这个文件只做 HTTP 胶水。**
参数映射、预览渲染、自检统计全在 ``pixelart.tune`` 里（可被单元测试覆盖）。
这里负责：路由、场景/帧缓存、后台全量出图任务、编码。

───── 两个关键的性能设计 ─────

1. **场景缓存**：``build_scene`` 含深度推理（约 1.1 s），但只依赖
   网格/预处理那几项设置。按这些项做 key 缓存，拖雾浓度/动画滑杆时就不用重跑深度。
2. **两级预览节奏**：
   - **拖动中** → 只渲 **单帧**（`/api/still`，约 120 ms），给即时反馈
   - **松手后** → 渲整段循环（`/api/preview`，约 3 s，另加 `/api/frame` 逐帧取图）
   若拖动中就渲整段，滑杆会完全卡死。

启动::

    .\\.venv\\Scripts\\python.exe tools\\m3_server.py
    # 然后打开 http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
from pixelart.analyze import DepthEstimator  # noqa: E402
from pixelart.debugview import depth_view, luma_view, vol_view  # noqa: E402
from pixelart.encode import save_animation, save_still  # noqa: E402
from pixelart.paths import INPUT, OUT, out  # noqa: E402
from pixelart.palette import palette_from_frames  # noqa: E402
from pixelart.pipeline import compose_frame  # noqa: E402
from pixelart.tune import (  # noqa: E402
    TuneParams,
    aspect_value,
    build_scene,
    display_scale,
    render_preview,
    render_still,
    render_sweep,
)

UI_PATH = Path(__file__).resolve().parent / "m3_ui.html"
#: WGSL 着色器 + 浏览器端 JS。**由服务端按名字白名单提供**，不暴露目录。
#  ⚠️ 让浏览器加载**与验收台相同的文件**是刻意的：tools/wgsl_lab.py 验证过的
#     就是这几个文件的内容，所以"验过了"对浏览器同样成立。若把 WGSL 复制一份
#     塞进 JS 字符串里，验证与运行就会是两份东西 —— 那验收台就白做了。
ASSET_DIR = Path(__file__).resolve().parent
WGSL_DIR = ASSET_DIR / "wgsl"
#: 允许通过 /api/wgsl 与 /api/gpuasset 取的文件（白名单，防目录穿越）
WGSL_NAMES = ("fog", "scatter", "volumetric", "bloom_bright", "bloom_blur",
              "bloom_combine", "dust_splat", "dust_apply", "tail", "present",
    # M5 分层视差（前向 splatting + u32 定点原子累加）
    "parallax_splat", "parallax_apply",
)
JS_NAMES = ("m3_wgsl_uniforms", "m3_webgpu")
PARAM_DIR = OUT / "m3" / "params"
#: 用户上传的素材落这里（out/ 已 gitignore，不会污染仓库）
UPLOAD_DIR = OUT / "m3" / "uploads"
UPLOAD_MAX_BYTES = 24 * 1024 * 1024          # 24 MB，够 8K 单图了
PIL_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF", "AVIF"}

# ── 缓存（场景贵、帧便宜但数量多，各用一个 LRU）─────────────────────────
_SCENE_CACHE: "OrderedDict[str, object]" = OrderedDict()
_SCENE_MAX = 4
_FRAME_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_FRAME_MAX = 400
_DEBUG_CACHE: dict[str, bytes] = {}          # 调试图 PNG（LRU，见 api_debug）
_DEBUG_MAX = 48
_PAL_CACHE: "OrderedDict[str, object]" = OrderedDict()
_PAL_MAX = 12
_RENDER_LOCK = threading.Lock()          # 渲染是 CPU 密集，串行化免得互相拖慢
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


# ----------------------------------------------------------------- 工具
def scene_key(p: TuneParams, asset: str) -> str:
    """场景只依赖网格与预处理（含天空替换——它改 base）—— 其余滑杆不会让缓存失效。"""
    raw = (f"{asset}|{p.grid_long}|{p.work_long}|{p.aspect}|{p.levels}|{p.sharpen}|"
           f"{p.var_gain}|{p.detail}|{p.sky_on}|{p.sky_mode}|{p.sky_stars}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def params_key(p: TuneParams) -> str:
    return hashlib.sha1(p.to_query().encode()).hexdigest()[:16]


def get_scene(p: TuneParams, asset: str):
    key = scene_key(p, asset)
    if key in _SCENE_CACHE:
        _SCENE_CACHE.move_to_end(key)
        return _SCENE_CACHE[key]
    path = find_asset(asset)
    if path is None:
        raise FileNotFoundError(f"素材不存在: {asset}")
    with _RENDER_LOCK:
        sc = build_scene(Image.open(path), p, depth_estimator=DepthEstimator())
    _SCENE_CACHE[key] = sc
    while len(_SCENE_CACHE) > _SCENE_MAX:
        _SCENE_CACHE.popitem(last=False)
    return sc


def png_bytes(u8: np.ndarray, scale: int) -> bytes:
    im = Image.fromarray(np.ascontiguousarray(u8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


#: 求色板时固定的采样时刻（占一个周期的比例）。
#
# ⚠️⚠️ **必须固定，不能取决于"预览帧数"。**
#
#    原实现是 ``t = i / n_frames``（在 0..1 上等分 n_frames 个点里抽 8 个）——
#    于是**同一场景同一参数、只因为 n_frames 不同就会得到两块不同的色板**。
#    后果实测：浏览器侧的场景包用 3 帧求色板、``/api/still`` 用另一个帧数，
#    两边色板不一致 → **51.7% 的像素吸附到不同颜色**，
#    看起来像"移植错了"，其实是**色板根本不是同一块**。
#
#    更根本的问题是语义：色板是**全片共用**并且应当**确定**的（时序铁律 1）。
#    "预览多少帧"是展示层的量，不该影响它。固定 8 个等分点之后，
#    预览 / 单帧 / 场景包 / 导出**必然共用同一块色板**。
_PALETTE_T = tuple(k / 8.0 for k in range(8))


#: 求色板时固定的采样时刻（占一个周期的比例）。
#
# ⚠️⚠️ **必须固定，不能取决于"预览帧数"。**
#
#    原实现是 ``t = i / n_frames``（在 0..1 上等分 n_frames 个点里抽 8 个）——
#    于是**同一场景同一参数、只因为 n_frames 不同就会得到两块不同的色板**。
#    后果实测：浏览器侧的场景包用 3 帧求色板、``/api/still`` 用另一个帧数，
#    两边色板不一致 → **51.7% 的像素吸附到不同颜色**，
#    看起来像"移植错了"，其实是**色板根本不是同一块**。
#
#    更根本的问题是语义：色板是**全片共用**并且应当**确定**的（时序铁律 1）。
#    "预览多少帧"是展示层的量，不该影响它。固定 8 个等分点之后，
#    预览 / 单帧 / 场景包 / 导出**必然共用同一块色板**。
_PALETTE_T = tuple(k / 8.0 for k in range(8))


def get_preview_palette(p: TuneParams, asset: str, n_frames: int | None = None):
    """取这次参数下的**全片共用色板**（时序铁律 1）。

    ⚠️ 单帧接口（``/api/still`` / ``/api/frame``）、场景包、导出**必须**用这块。

    第一版这里踩过坑：单帧接口各自调 ``render_still(palette=None)``，
    也就是**每帧各求一块色板**，浏览器播放时相邻帧色板漂移 ——
    正好制造出我们花大力气防的"整片闪烁"。

    第二个坑（更隐蔽）：采样点原来写成 ``t = i / n_frames``，
    于是色板**取决于 n_frames**。那是个展示层的量，不该影响色彩。
    现在改成固定采样时刻 ``_PALETTE_T``（见那里的说明）。

    ``n_frames`` 参数保留只为兼容旧调用点，**不再参与计算**。
    """
    key = f"pal|{scene_key(p, asset)}|{params_key(p)}"
    if key in _PAL_CACHE:
        _PAL_CACHE.move_to_end(key)
        return _PAL_CACHE[key]
    sc = get_scene(p, asset)
    kw = p.compose_kwargs(sc)
    with _RENDER_LOCK:
        sampled = [compose_frame(sc, t=float(t), **kw) for t in _PALETTE_T]
        pal = palette_from_frames(sampled, n_colors=int(p.colors),
                                  max_samples=len(sampled))
    _PAL_CACHE[key] = pal
    while len(_PAL_CACHE) > 12:
        _PAL_CACHE.popitem(last=False)
    return pal


def param_from_query(qs: dict, asset: str) -> tuple[TuneParams, str]:
    """从查询串取参数。``asset`` 单独取（它不属于 TuneParams）。"""
    p = TuneParams.from_query(qs)
    a = qs.get("asset", [""])[0] if qs.get("asset") else ""
    if not a:
        files = sorted(f.name for f in INPUT.glob("*.jpg"))
        a = files[0] if files else ""
    return p, a


def find_asset(name: str) -> Path | None:
    """在「内置样例」与「用户上传」两处找素材。

    ⚠️ 上传的名字经过 sanitize（见 ``sanitize_name``），所以这里不含路径穿越风险；
    但仍然只按**文件名**查找，绝不把用户提供的字符串当路径拼。
    """
    if not name:
        return None
    safe = Path(name).name
    for d in (INPUT, UPLOAD_DIR):
        p = d / safe
        if p.is_file():
            return p
    return None


def sanitize_name(raw: str) -> str:
    """把用户提供的文件名压成安全的 ASCII 文件名（保留扩展名）。

    ⚠️ 产物文件名一律 ASCII 是本仓库的既有约定（``tests/test_repo_hygiene.py``
    有护栏）。中文/空格/特殊字符会造成路径与编码问题，所以统一转写成
    ``upload-<hash>.<ext>`` —— 既安全、又天然去重（同名文件不会互相覆盖）。
    """
    p = Path(raw or "image")
    ext = p.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"):
        ext = ".png"
    stem = p.stem
    ascii_stem = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "-"
                         for c in stem)[:40].strip("-_")
    digest = hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:8]
    return f"{ascii_stem or 'img'}-{digest}{ext}"


def pack_segments(segs: dict, meta: dict) -> bytes:
    """把若干二进制段打成一个包：``u32 头部长度`` + JSON 头部 + 负载。

    ⚠️ 自己定格式而不是用 multipart / base64：
    这里是**二进制浮点数据**，base64 会白涨 33% 且多一次编解码。
    头部里带每一段的 dtype / shape / 偏移，浏览器侧解起来只有几行。
    """
    # ⚠️⚠️ **每一段都必须 4 字节对齐**（头部本身也要补到 4 的倍数）。
    #
    #    否则浏览器侧 `new Float32Array(u8.buffer, u8.byteOffset, n)` 会直接抛
    #    `RangeError: start offset of Float32Array should be a multiple of 4`
    #    —— 因为 JSON 头部的长度是任意的，它后面的第一段几乎必然不对齐。
    #    实测就是这么暴露出来的：诊断代码试图把段视图成 Float32Array 时报错。
    #
    #    另一种"修法"是让浏览器每次先复制一份对齐的，但那等于每个场景包都白拷
    #    几百 KB。**在生产者一侧补齐**更省、也更不容易忘。
    def _pad4(b: bytes) -> bytes:
        r = len(b) % 4
        return b if r == 0 else b + b"\x00" * (4 - r)

    offset = 0
    payload: list[bytes] = []
    head = {"meta": meta, "segments": {}}
    for name, (dtype, shape, data) in segs.items():
        head["segments"][name] = {"dtype": dtype, "shape": [int(v) for v in shape],
                                  "offset": int(offset), "nbytes": int(len(data))}
        padded = _pad4(data)
        payload.append(padded)
        offset += len(padded)

    hb = json.dumps(head, ensure_ascii=False).encode("utf-8")
    # ⚠️ 头部补齐必须用**空格**，不能用 NUL —— NUL 会让 JSON 解析器报
    #    "Extra data"（Python 的 json.loads 与浏览器的 JSON.parse 都会）。
    #    空格是合法的尾随空白，两边都容忍。
    r = len(hb) % 4
    if r:
        hb += b" " * (4 - r)
    return len(hb).to_bytes(4, "little") + hb + b"".join(payload)


def build_scene_pack(p: "TuneParams", asset: str, parts: str) -> bytes:
    """组装发给浏览器的场景包。

    ⚠️ **必须是模块级函数，而不是 Handler 的方法**。
    第一版把这段逻辑写在 ``api_scene`` 里，于是测试只能"照着重写一遍"来构造
    期望值 —— 那就成了"测我自己抄的副本"，服务端真正的代码路径一行都没跑到
    （改了 handler 而忘了改测试，测试照样绿）。
    提出来之后 handler 与测试调的是**同一个函数**。

    段与含义：

      ``scene``   ``base`` (Gh,Gw,3) 与 ``far`` (Gh,Gw) —— 就是
                  ``compose_frame`` 的输入。它们**已经是网格分辨率**
                  （``prepare_scene`` 里降过了），所以数据量很小。
      ``assets``  ``palette`` / ``kernels`` / ``dust`` + 一串常量。
                  这些是浏览器**不自己算**的东西：色板要服务端合成帧来求，
                  核系数与粒子 hash 参数在 CPU 上有权威实现。
    """
    from pixelart.animate import DEFAULT_FLICKER_FREQS, hash01
    from pixelart.compose import BLOOM_RADII
    from pixelart.webgpu import (
        TWO_PI, bloom_kernels, dust_fixed_point_scale, dust_particle_params,
        fog_harmonics,
    )

    want = {x.strip() for x in (parts or "").split(",") if x.strip()}
    sc = get_scene(p, asset)

    segs: dict[str, tuple[str, list[int], bytes]] = {}
    meta: dict = {
        "grid": [int(sc.grid[0]), int(sc.grid[1])],
        "scale": int(sc.scale),
        "out_size": [int(sc.out_size[0]), int(sc.out_size[1])],
        # light_xy = 打包那一刻 params 解析出的位置（自检面板展示用）；
        # light_xy_auto = **场景检测值**（只依赖 scene，与参数无关）——
        # GPU 侧自动模式每帧用它、手动模式用滑杆，两者不得混用。
        "light_xy": [float(v) for v in p.light_xy(sc)],
        "light_xy_auto": [float(v) for v in sc.light_xy],
        "asset": asset,
    }

    if "scene" in want:
        segs["base"] = ("f32", [sc.grid[1], sc.grid[0], 3],
                        np.ascontiguousarray(sc.base, np.float32).tobytes())
        segs["far"] = ("f32", [sc.grid[1], sc.grid[0]],
                       np.ascontiguousarray(sc.far, np.float32).tobytes())
        meta["fog_color_raw"] = [float(v) for v in sc.fog_color]
        # ── 分层视差的静态量：层边界由 CPU 算一次（WGSL 没有 quantile），
        #    其余是常量。浏览器只做"数一数有几个边界 ≤ far"。
        from pixelart.parallax import (DRIFT, FAR_GAIN, LAYERS, NEAR_GAIN,
                                       PIVOT, SPLAT_SCALE, layer_edges)
        from pixelart.webgpu import parallax_layout
        _ed = layer_edges(sc.far, LAYERS)
        segs["par_edges"] = ("f32", [int(_ed.size)],
                             np.ascontiguousarray(_ed, np.float32).tobytes())
        meta["par_n_edges"] = int(_ed.size)
        meta["par_near_gain"] = float(NEAR_GAIN)
        meta["par_far_gain"] = float(FAR_GAIN)
        meta["par_drift"] = float(DRIFT)
        meta["par_pivot"] = [float(PIVOT[0]), float(PIVOT[1])]
        meta["par_scale"] = int(SPLAT_SCALE)
        meta["par_layout"] = parallax_layout(int(sc.grid[0] * sc.grid[1]))
        meta["depth_q"] = [float(v) for v in sc.stats.get("depth_q", [])]

    if "assets" in want:
        pal = get_preview_palette(p, asset, 3)
        segs["palette"] = ("f32", [int(len(pal)), 3],
                           np.ascontiguousarray(pal, np.float32).tobytes())
        meta["palette_len"] = int(len(pal))

        kflat, table = bloom_kernels(BLOOM_RADII)
        segs["kernels"] = ("f32", [int(len(kflat))],
                           kflat.astype(np.float32).tobytes())
        meta["kbuf_table"] = [[int(o), int(c), int(hh), float(wt)]
                              for (o, c, hh, wt) in table]
        meta["bloom_kernels_total"] = int(len(kflat))
        # ⚠️ 辉光的阈值/软阈值/权重和也必须给 —— 浏览器侧的 uniform 要用它们。
        #    第一版漏了这三个，于是 JS 里 `A.bloom_threshold` 是 undefined，
        #    写进 uniform 就成了 NaN，高光 mask 全 0 → `bright` 缓冲全 0
        #    → 辉光和之后的一切都是 0 → 最终输出纯黑。
        #    **而且不报任何错**（NaN 在 GPU 上是合法值）。
        #    这就是"两边各取一次同一个概念"的又一处 —— 现在由
        #    tests/test_webgpu_serving.py 里那条"meta 必须覆盖 JS 读的键"守住。
        from pixelart.compose import BLOOM_KNEE, BLOOM_THRESHOLD
        meta["bloom_threshold"] = float(BLOOM_THRESHOLD)
        meta["bloom_knee"] = float(BLOOM_KNEE)
        meta["bloom_wsum"] = float(sum(1.0 / (i + 1)
                                       for i in range(len(BLOOM_RADII))))

        count = int(p.dust_count)
        par = dust_particle_params(count, seed=11)
        segs["dust"] = ("f32", [int(par.shape[0]), int(par.shape[1])],
                        np.ascontiguousarray(par, np.float32).tobytes())
        meta["dust_count"] = count
        meta["dust_scale"] = int(dust_fixed_point_scale(count))
        meta["dust_stride"] = 11

        # 静态常量：谐波系数 + 闪烁相位。
        # 浏览器不重新算它们 —— 谐波要 hash01（splitmix64，WebGPU 没有 i64），
        # 闪烁相位同理。**算一次、传过去**，就不会两边漂移。
        meta["fog_harmonics"] = [
            [float(hh["fx"]), float(hh["fy"]), int(hh["ft"]),
             float(hh["phase"]), float(hh["amp"])] for hh in fog_harmonics()
        ]
        meta["flicker_freqs"] = [int(f) for f in DEFAULT_FLICKER_FREQS]
        meta["flicker_phases"] = [float(TWO_PI * hash01(0, i))
                                  for i in range(len(DEFAULT_FLICKER_FREQS))]

    return pack_segments(segs, meta)


def list_assets() -> list[dict]:
    out_list = []
    for d, origin in ((INPUT, "sample"), (UPLOAD_DIR, "upload")):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*")):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp",
                                       ".tif", ".tiff", ".gif", ".avif"):
                continue
            try:
                with Image.open(f) as im:
                    w, h = im.size
            except Exception:                          # noqa: BLE001
                continue                               # 坏文件直接跳过，不要让列表 500
            out_list.append({"name": f.name, "w": w, "h": h,
                             "ratio": round(w / max(h, 1), 3), "origin": origin})
    return out_list


# ----------------------------------------------------------------- 路由
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pixelart-m3"

    def log_message(self, fmt, *args):        # 静音默认日志（太吵）
        pass

    # ---- 响应助手 ----
    def _send(self, body: bytes, ctype: str, code: int = 200, cache: bool = False,
              download: str | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        if download:
            # 走浏览器下载通道：文件落到用户的下载目录，而不是只躺在服务端 out/
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    def _err(self, msg: str, code: int = 500):
        self._json({"error": msg}, code)

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            route = u.path
            if route in ("/", "/index.html"):
                return self._send(UI_PATH.read_bytes(), "text/html; charset=utf-8")
            if route == "/api/assets":
                return self._json({"assets": list_assets()})
            if route == "/api/meta":
                return self.api_meta(qs)
            if route == "/api/debug":
                return self.api_debug(qs)
            if route == "/api/still":
                return self.api_still(qs)
            if route == "/api/frame":
                return self.api_frame(qs)
            if route == "/api/preview":
                return self.api_preview(qs)
            if route == "/api/sweep":
                return self.api_sweep(qs)
            if route == "/api/job":
                return self.api_job(qs)
            if route == "/api/export_file":
                return self.api_export_file(qs)
            if route == "/api/params":
                return self.api_params_get(qs)
            if route == "/api/wgsl":
                return self.api_wgsl(qs)
            if route == "/api/gpuasset":
                return self.api_gpuasset(qs)
            if route == "/api/scene":
                return self.api_scene(qs)
            return self._err(f"未知路由 {route}", 404)
        except FileNotFoundError as e:
            return self._err(str(e), 404)
        except Exception as e:                      # noqa: BLE001
            traceback.print_exc()
            return self._err(f"{type(e).__name__}: {e}")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/upload":
            return self.api_upload()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return self._err("请求体不是合法 JSON", 400)
        if u.path == "/api/export":
            return self.api_export(payload)
        if u.path == "/api/params":
            return self.api_params_post(payload)
        return self._err(f"未知路由 {u.path}", 404)

    def api_upload(self):
        """接收用户上传的图片（原始字节，不是 multipart）—— 让管线不限于内置样例。

        设计取舍：
        - **不用 multipart**：解析 multipart 要额外依赖，而前端用
          ``fetch(url, {body: file})`` 直接发原始字节更简单可靠。
        - **必须校验真是图片**：用 Pillow 打开并 ``verify()``。
          否则任意文件都能塞进来，之后在渲染时才炸。
        - **文件名转写成 ASCII**：本仓库约定产物名一律 ASCII
          （``tests/test_repo_hygiene.py`` 有护栏），且哈希后缀天然去重。
        """
        raw_len = int(self.headers.get("Content-Length") or 0)
        if raw_len <= 0:
            return self._err("请求体为空", 400)
        if raw_len > UPLOAD_MAX_BYTES:
            return self._err(f"文件过大（上限 {UPLOAD_MAX_BYTES // 1024 // 1024} MB）", 413)

        data = self.rfile.read(raw_len)
        # 文件名走 URL 查询串（避免 multipart）；前端用 encodeURIComponent 传
        qs = parse_qs(urlparse(self.path).query)
        raw_name = (qs.get("name", ["image"])[0] or "image")

        try:
            im = Image.open(io.BytesIO(data))
            im.verify()                                     # 校验确实是图片
            fmt = (im.format or "").upper()
            im2 = Image.open(io.BytesIO(data))
            w, h = im2.size
        except Exception as e:                              # noqa: BLE001
            return self._err(f"不是有效的图片文件：{type(e).__name__}", 400)

        if fmt not in PIL_FORMATS:
            return self._err(f"不支持的图片格式：{fmt or '未知'}", 400)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / sanitize_name(raw_name)
        target.write_bytes(data)
        return self._json({"ok": True, "name": target.name, "w": w, "h": h,
                           "format": fmt, "bytes": len(data)})

    # ---- 接口实现 ----
    def api_meta(self, qs):
        """只返回**元信息**（不渲染任何帧）—— 给界面用来定显示尺寸。

        存在的理由：界面需要知道"逻辑输出尺寸"才能把取景框设成正确大小，
        而显示尺寸**不能依赖拉回来的图的实际像素**（那正是"画面跳变"的根因）。
        只渲染一帧也能拿到尺寸，但那样首屏会先空一下。
        这个接口只读缓存的 scene，很快就返回。
        """
        p, asset = param_from_query(qs, "asset")
        sc = get_scene(p, asset)
        return self._json({
            "asset": asset,
            "grid": [int(sc.grid[0]), int(sc.grid[1])],
            "scale": int(sc.scale),
            "out_size": [int(sc.out_size[0]), int(sc.out_size[1])],
            "src_size": [int(sc.src_size[0]), int(sc.src_size[1])],
            "display_scale": display_scale(sc),
            "light_xy_used": [float(v) for v in p.light_xy(sc)],
            "fog_color": [float(v) for v in sc.fog_color],
            "depth_q": [float(v) for v in np.percentile(sc.far, [5, 25, 50, 75, 95])],
        })

    # ---- WebGPU：静态资源 + 场景包 ----
    def api_wgsl(self, qs):
        """提供 WGSL 源码（白名单）。"""
        name = (qs.get("name", [""])[0] or "").strip()
        if name not in WGSL_NAMES:
            return self._err(f"未允许的着色器 {name!r}", 404)
        f = WGSL_DIR / f"{name}.wgsl"
        if not f.exists():
            return self._err(f"缺少 {f.name}", 404)
        # ⚠️ 不缓存（no-store）：着色器在开发期**频繁修改**，而"改文件名"的
        #    前提并不存在 —— 路径是固定的。缓存一小时 = 用户浏览器里跑着
        #    一小时前的旧着色器，症状是"改了没生效"。
        return self._send(f.read_bytes(), "text/plain; charset=utf-8", cache=False)

    def api_gpuasset(self, qs):
        """提供浏览器端 JS（白名单）。"""
        name = (qs.get("name", [""])[0] or "").strip()
        if name not in JS_NAMES:
            return self._err(f"未允许的脚本 {name!r}", 404)
        f = ASSET_DIR / f"{name}.js"
        if not f.exists():
            return self._err(f"缺少 {f.name}", 404)
        # ⚠️ 不缓存（no-store）：这是"GPU 模式看不到新效果"的根因 ——
        #    max-age=3600 让浏览器跑一小时前的旧 JS。旧 JS 渲染其余部分
        #    全部正常，**唯独没有新加的 pass**，极难从画面上判断出来。
        return self._send(f.read_bytes(), "application/javascript; charset=utf-8",
                          cache=False)

    def api_scene(self, qs):
        """场景包：base + far（**网格分辨率**）+ 素材。见 :func:`build_scene_pack`。

        ``parts`` 用逗号分隔：``scene``（base/far/场景元信息）与
        ``assets``（色板 + 辉光核 + 粒子参数 + 着色器常量）。
        """
        p, asset = param_from_query(qs, "asset")
        parts = (qs.get("parts", ["scene,assets"])[0] or "")
        return self._send(build_scene_pack(p, asset, parts),
                          "application/octet-stream")

    def api_debug(self, qs):
        """调试图 PNG：depth（深度）/ luma（等亮线）/ vol（体积光形状）。

        ⚠️ 缓存键必须**包含 params_key（全量）**，哪怕 depth/luma 只依赖场景：
           图上画的光源十字用的是 ``p.light_xy(sc)``（自动=检测、手动=滑杆），
           拖滑杆必须换新图。成本 ~20ms 且浏览器对相同 URL 自带缓存，无所谓。
        """
        p, asset = param_from_query(qs, "asset")
        kind = (qs.get("kind", ["depth"])[0] or "depth").strip()
        sc = get_scene(p, asset)
        a = p.light_xy(sc)
        lx = int(np.clip(a[0], 0.0, 1.0) * (sc.far.shape[1] - 1))
        ly = int(np.clip(a[1], 0.0, 1.0) * (sc.far.shape[0] - 1))
        key = f"dbg|{kind}|{scene_key(p, asset)}|{params_key(p)}"
        body = _DEBUG_CACHE.get(key)
        if body is None:
            if kind == "depth":
                im = depth_view(sc, lx, ly)
            elif kind == "luma":
                im = luma_view(sc, lx, ly)
            elif kind == "vol":
                im = vol_view(sc, density=p.density, power=p.power,
                              fog_tint=p.fog_tint, screen_falloff=p.rays_spread,
                              cone_angle=p.rays_cone, cone_dir_deg=p.rays_dir,
                              cone_reach=p.rays_reach, shaft=p.rays_shaft,
                              lx=lx, ly=ly, cone_x=p.cone_x, cone_y=p.cone_y)
            else:
                return self._err(f"未知调试图 {kind}", 404)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=False, compress_level=3)
            body = buf.getvalue()
            _DEBUG_CACHE[key] = body
            while len(_DEBUG_CACHE) > _DEBUG_MAX:
                _DEBUG_CACHE.pop(next(iter(_DEBUG_CACHE)))
        return self._send(body, "image/png", cache=True)

    def api_still(self, qs):
        """单帧 PNG —— 拖动滑杆时的即时反馈。

        ⚠️ 用全片共用色板，不是单帧自己的色板 —— 否则拖滑杆时看到的颜色
        会和成片不一致，而且会制造"色板漂移"的假象。
        """
        p, asset = param_from_query(qs, "asset")
        scale = max(1, int(qs.get("scale", ["0"])[0] or 0))
        t = float(qs.get("t", ["0"])[0] or 0.0)
        n = max(1, p.preview_frames(get_scene(p, asset)))
        key = f"t|{scene_key(p, asset)}|{params_key(p)}|{t:.4f}|{scale}|{n}"
        if key in _FRAME_CACHE:
            _FRAME_CACHE.move_to_end(key)
            return self._send(_FRAME_CACHE[key], "image/png", cache=True)
        sc = get_scene(p, asset)
        if scale == 0:
            scale = display_scale(sc)
        pal = get_preview_palette(p, asset, n)
        with _RENDER_LOCK:
            u8, _ = render_still(sc, p, t=t, palette=pal)
            body = png_bytes(u8, scale)
        _FRAME_CACHE[key] = body
        while len(_FRAME_CACHE) > _FRAME_MAX:
            _FRAME_CACHE.popitem(last=False)
        return self._send(body, "image/png", cache=True)

    def api_frame(self, qs):
        """循环预览里的第 i 帧 PNG（用于浏览器端播放）。

        ⚠️ **必须与整段预览共用同一块色板**，否则播放时逐帧色板漂移 =
        人为造出"整片闪烁"。实测这个坑：第一版这里用了单帧色板，
        ``frame(0)`` 与 ``frame(N)`` 渲染出的 PNG 都不相等（本该逐位相同）。
        """
        p, asset = param_from_query(qs, "asset")
        scale = max(1, int(qs.get("scale", ["0"])[0] or 0))
        i = int(qs.get("i", ["0"])[0] or 0)
        fps = max(2, int(qs.get("preview_fps", ["10"])[0] or 10))
        sc = get_scene(p, asset)
        n = p.preview_frames(sc, fps)
        key = f"f|{scene_key(p, asset)}|{params_key(p)}|{i % n}|{'s' + str(scale)}|{n}"
        if key in _FRAME_CACHE:
            _FRAME_CACHE.move_to_end(key)
            return self._send(_FRAME_CACHE[key], "image/png", cache=True)
        if scale == 0:
            scale = display_scale(sc)
        pal = get_preview_palette(p, asset, n)
        with _RENDER_LOCK:
            u8, _ = render_still(sc, p, t=(i % n) / n, palette=pal)
            body = png_bytes(u8, scale)
        _FRAME_CACHE[key] = body
        while len(_FRAME_CACHE) > _FRAME_MAX:
            _FRAME_CACHE.popitem(last=False)
        return self._send(body, "image/png", cache=True)

    def api_preview(self, qs):
        """渲染整段预览循环，返回自检统计 + 帧数（帧本身走 /api/frame）。

        ⚠️ **这里渲染的帧会被写进 `_FRAME_CACHE`。**
        第一版只返回统计、把渲好的帧丢掉，然后界面逐帧来取时
        `/api/frame` **又从头渲了一遍** —— 整段渲染做了两次。

        实测（240 网格、60 帧）：
            服务端渲 60 帧（算统计）   3.19 s
            逐帧取图 60 次（重渲）      3.13 s
        合计 6.3 s，正是用户体感的"8~10 秒"的主体。
        现在统计与图像共用同一次渲染，直接省掉后半段。
        """
        p, asset = param_from_query(qs, "asset")
        fps = max(2, int(qs.get("preview_fps", ["10"])[0] or 10))
        sc = get_scene(p, asset)
        n = p.preview_frames(sc, fps)
        scale_q = int(qs.get("scale", ["0"])[0] or 0)
        scale = scale_q if scale_q > 0 else display_scale(sc)
        pal = get_preview_palette(p, asset, n)
        with _RENDER_LOCK:
            frames, _, stats = render_preview(sc, p, preview_fps=fps,
                                             include_last=True, palette=pal)
            # 顺手把帧编码好塞进缓存 —— 界面随后来取就直接命中
            sres = "s" + str(scale)
            for i, u8 in enumerate(frames):
                k = f"f|{scene_key(p, asset)}|{params_key(p)}|{i % n}|{sres}|{n}"
                if k not in _FRAME_CACHE:
                    _FRAME_CACHE[k] = png_bytes(u8, scale)
            while len(_FRAME_CACHE) > _FRAME_MAX:
                _FRAME_CACHE.popitem(last=False)
        stats["preview_fps"] = fps
        stats["display_scale"] = scale
        stats["asset"] = asset
        stats["seconds"] = p.seconds
        return self._json(stats)

    def api_sweep(self, qs):
        """参数扫描：一个参数取多个值，拼成一张对照图。"""
        p, asset = param_from_query(qs, "asset")
        key = (qs.get("key", ["density"])[0] or "density")
        raw = (qs.get("values", ["0.4,0.7,1.1"])[0] or "")
        vals = [float(v) for v in raw.split(",") if v.strip()]
        vals = vals[:8]
        if not vals:
            return self._err("values 为空", 400)
        sc = get_scene(p, asset)
        with _RENDER_LOCK:
            items = render_sweep(sc, p, key, vals, t=float(qs.get("t", ["0"])[0] or 0))

        scale = display_scale(sc, target=300)
        tiles = [Image.fromarray(u8).resize((u8.shape[1] * scale, u8.shape[0] * scale),
                                           Image.Resampling.NEAREST)
                 for _, u8 in items]
        cols = min(4, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        cw = max(t.width for t in tiles) + 16
        ch = max(t.height for t in tiles) + 46
        sheet = Image.new("RGB", (cw * cols, ch * rows), (14, 14, 17))
        d = ImageDraw.Draw(sheet)
        for i, ((v, _), t) in enumerate(zip(items, tiles)):
            bx, by = (i % cols) * cw, (i // cols) * ch
            sheet.paste(t, (bx + 8, by + 36))
            d.text((bx + 10, by + 10), f"{key} = {v:g}", font=C.font(22), fill=(220, 220, 215))
        buf = io.BytesIO()
        sheet.save(buf, format="PNG", compress_level=3)
        return self._send(buf.getvalue(), "image/png")

    # ---- 全量出图（后台任务）----
    def api_export(self, payload: dict):
        qs = {k: [str(v)] for k, v in (payload.get("params") or {}).items()}
        asset = str(payload.get("asset") or "")
        fmt = str(payload.get("format") or "webp")
        full_grid = int(payload.get("grid_long") or 480)
        full_work = int(payload.get("work_long") or 1920)
        if not asset:
            return self._err("缺少 asset", 400)

        p = TuneParams.from_query(qs)
        p.grid_long = full_grid
        p.work_long = full_work
        # 全量渲染用成片的 fps（预览的降帧只属于预览）
        jid = hashlib.sha1(f"{asset}|{params_key(p)}|{fmt}".encode()).hexdigest()[:12]
        with _JOBS_LOCK:
            if _JOBS.get(jid, {}).get("state") == "running":
                return self._json({"job": jid, "state": "running"})
            _JOBS[jid] = {"state": "running", "done": 0, "total": 0, "msg": "准备中"}
        threading.Thread(target=_export_worker, args=(jid, p, asset, fmt), daemon=True).start()
        return self._json({"job": jid, "state": "running"})

    def api_job(self, qs):
        jid = (qs.get("id", [""])[0] or "")
        with _JOBS_LOCK:
            job = dict(_JOBS.get(jid) or {})
        if not job:
            return self._err("任务不存在", 404)
        return self._json(job)

    def api_export_file(self, qs):
        """把全量出图的结果**通过浏览器下载通道**交给用户（attachment）。"""
        jid = (qs.get("job", [""])[0] or "")
        try:
            idx = int(qs.get("idx", ["0"])[0] or 0)
        except ValueError:
            idx = 0
        with _JOBS_LOCK:
            job = dict(_JOBS.get(jid) or {})
        files = job.get("files") or []
        if job.get("state") != "done" or idx >= len(files):
            return self._err("导出文件不存在（任务未完成或编号越界）", 404)
        fp = Path(files[idx])
        if not fp.is_file():
            return self._err("导出文件已被清理", 404)
        ctype = "image/webp" if fp.suffix.lower() == ".webp" else "image/png"
        self._send(fp.read_bytes(), ctype, download=fp.name)

    # ---- 参数存取 ----
    def api_params_get(self, qs):
        name = (qs.get("name", [""])[0] or "").strip()
        if name:
            f = PARAM_DIR / f"{name}.json"
            if not f.exists():
                return self._err("参数文件不存在", 404)
            return self._json({"name": name, "params": json.loads(f.read_text("utf-8"))})
        PARAM_DIR.mkdir(parents=True, exist_ok=True)
        return self._json({"saved": sorted(f.stem for f in PARAM_DIR.glob("*.json"))})

    def api_params_post(self, payload: dict):
        name = str(payload.get("name") or "").strip()
        if not name or any(c in name for c in '\\/:*?"<>|'):
            return self._err("参数名非法", 400)
        PARAM_DIR.mkdir(parents=True, exist_ok=True)
        (PARAM_DIR / f"{name}.json").write_text(
            json.dumps(payload.get("params") or {}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return self._json({"ok": True, "name": name})


def _export_worker(jid: str, p: TuneParams, asset: str, fmt: str):
    """后台全量出图。进度写进 _JOBS，前端轮询。"""
    def setj(**kw):
        with _JOBS_LOCK:
            _JOBS[jid].update(kw)

    try:
        sc = get_scene(p, asset)
        n = max(2, int(round(p.seconds * max(1, p.fps))))
        setj(total=n + 1, msg=f"渲染 {n} 帧 @ 网格 {sc.grid[0]}x{sc.grid[1]}")

        with _RENDER_LOCK:
            def prog(i, tot):
                setj(done=i, total=tot)

            frames, pal, stats = render_preview(
                sc, p, preview_fps=int(p.fps), include_last=True, on_progress=prog)
            from pixelart.pipeline import upscale
            big = [upscale(f, sc.scale) for f in frames]
            stem = Path(asset).stem
            wp = save_animation(big, out("m3", f"{stem}_tuned.{fmt}"),
                               duration_ms=int(round(1000 / max(1, p.fps))), lossless=True)
            st = save_still(big[0], out("m3", f"{stem}_tuned.png"))

        setj(state="done", msg=f"{wp.name}（{wp.stat().st_size / 1e6:.1f} MB）",
             files=[str(wp), str(st)], stats=stats)
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        setj(state="error", msg=f"{type(e).__name__}: {e}")


def _port_busy(host: str, port: int) -> int | None:
    """端口上是否已有别的进程在监听？返回那个 PID（拿不到就返回 0）。

    ⚠️ 为什么必须查：Python 的 ``HTTPServer.allow_reuse_address`` 在 Windows 上
    允许**多个进程同时绑定同一端口**。实测这台机器上 8770 曾同时有 4 个在听，
    请求由 OS 随便挑一个处理 —— 于是"改了代码却没生效"，
    而报错信息完全看不出是这个原因（排查了一整轮）。

    所以启动前先查一次；有别人在听就报出来，别悄悄又起一个。
    """
    import subprocess
    try:
        r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        for line in (r.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "LISTENING" \
                    and parts[1].endswith(f":{port}"):
                return int(parts[-1])
    except Exception:                                          # noqa: BLE001
        return 0
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="只绑本机，不要暴露到 0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()

    if not UI_PATH.exists():
        print(f"[错误] 找不到界面文件 {UI_PATH}")
        return 1

    busy = _port_busy(args.host, args.port)
    if busy:
        print(f"[错误] 端口 {args.port} 上已有进程在监听（PID {busy}）。")
        print("       本服务不允许多实例 —— Windows 上多个进程可以绑同一端口，")
        print("       而请求会被 OS 随机分给其中一个，导致'改了代码没生效'。")
        print(f"       先停掉它：taskkill /PID {busy} /F")
        print(f"       查看占用：netstat -ano | findstr {args.port}")
        return 1
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    n_assets = len(list_assets())
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("=" * 62)
    print("  pixelart 调参台 (M3)")
    print("=" * 62)
    print(f"  地址     http://{args.host}:{args.port}")
    print(f"  素材     {n_assets} 张  (内置 {INPUT.name}/ + 上传 {UPLOAD_DIR.name}/)")
    print(f"  参数存档 {PARAM_DIR}")
    print(f"  上传目录 {UPLOAD_DIR}")
    print()
    print("  预览渲染在服务端进行。调参只渲首帧（约 50ms）；")
    print("  整段循环仅在「渲染预览」或点「播放」时才渲。按 Ctrl+C 退出。")
    print("=" * 62)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
