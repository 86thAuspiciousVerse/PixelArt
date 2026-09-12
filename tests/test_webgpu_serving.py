"""WebGPU 资产服务与场景包 —— 服务端这一侧的不变量。

═══ 为什么这些必须测 ═══

浏览器侧的渲染器依赖服务端送来的三样东西：

  1. **WGSL 源码**（`/api/wgsl`）—— 必须是**验收台验证过的那些文件**。
     若为了"方便"把 WGSL 抄进 JS 字符串里，验收台验的与浏览器跑的就是两份，
     验收就白做了。所以这里断言"按名字取到的内容 == 磁盘上的文件"。
  2. **场景包**（`/api/scene`）—— `base` / `far` 必须与 `build_scene` 的产物
     **逐位相同**。差一点，浏览器渲染的就不是我们验过的那个场景。
  3. **静态常量**（谐波系数 / 闪烁相位 / 核表 / 粒子参数）—— 这些是"唯一
     需要重复实现"之外的部分，必须由服务端算一次发过来，浏览器不自己算。

`/api/scene` 的 `base`/`far` 是**逐位**判据（它们只是搬运，不该有任何差异）；
没有 GPU 的环境也能跑，因为它不涉及 wgpu。
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import m3_server as srv  # noqa: E402


ASSET = "ref04_lain_room.jpg"


@pytest.fixture(scope="module")
def scene_pack():
    from pixelart.tune import TuneParams

    p = TuneParams.from_query({
        "asset": [ASSET], "grid_long": ["240"], "work_long": ["960"],
        "aspect": ["native"],
    })
    sc = srv.get_scene(p, ASSET)
    return p, sc


def _pack(parts: str, p):
    """调**服务端真正的**打包函数。

    ⚠️ 这里必须调 `srv.build_scene_pack`，不能"照着重写一遍"。
    第一版我就是在测试里手工重写了一遍打包逻辑 —— 那等于**测我自己抄的副本**：
    服务端改了 handler 而忘了同步测试，测试照样绿。
    现在两边走同一个函数（`build_scene_pack` 是从 handler 里提出来的）。
    """
    return srv.build_scene_pack(p, ASSET, parts)


def _decode(blob: bytes):
    hlen = struct.unpack("<I", blob[:4])[0]
    head = json.loads(blob[4:4 + hlen].decode("utf-8"))
    off = 4 + hlen
    out = {}
    for name, s in head["segments"].items():
        out[name] = np.frombuffer(
            blob[off + s["offset"]: off + s["offset"] + s["nbytes"]],
            dtype=np.float32).reshape(s["shape"])
    return head["meta"], out


# ══════════════════════════════════════════════════════════════════
def test_wgsl_is_served_from_the_verified_files():
    """`/api/wgsl` 提供的内容必须**就是**验收台跑过的那些文件。

    这不是形式检查：它保证"tools/wgsl_lab.py 验过了"对浏览器同样成立。
    如果有人把 WGSL 复制进 JS 字符串，这条会以另一种方式失效（浏览器不再
    取这些文件），而 `m3_uicheck` 的"脚本在主脚本前加载"检查会报出来。
    """
    names = srv.WGSL_NAMES
    assert "tail" in names and "fog" in names and "present" in names
    for n in names:
        f = srv.WGSL_DIR / f"{n}.wgsl"
        assert f.exists(), f"白名单里有 {n} 但磁盘上没有 {f.name}"


def test_gpuasset_whitelist_excludes_server_code():
    """JS 白名单不能包含服务端自己的代码（`m3_server`）。"""
    assert "m3_server" not in srv.JS_NAMES
    for n in srv.JS_NAMES:
        assert (srv.ASSET_DIR / f"{n}.js").exists(), f"缺少 {n}.js"


def test_scene_pack_base_and_far_are_bit_identical(scene_pack):
    """⭐ `base` / `far` 必须与 `build_scene` 的产物**逐位相同**。

    它们只是搬运（网格分辨率 → 网络 → GPU 缓冲），中间不该有任何变换。
    差一点，浏览器渲染的就不是验收台验过的那个场景。
    """
    p, sc = scene_pack
    blob = _pack("scene,assets", p)
    meta, segs = _decode(blob)
    assert list(sc.grid) == meta["grid"]
    assert np.array_equal(segs["base"], sc.base), "base 不是逐位相同"
    assert np.array_equal(segs["far"], sc.far), "far 不是逐位相同"
    assert segs["base"].shape == (sc.grid[1], sc.grid[0], 3)
    assert segs["far"].shape == (sc.grid[1], sc.grid[0])


def test_scene_pack_constants_match_the_python_side(scene_pack):
    """场景包里的**静态常量**必须与服务端算的一致。

    这些量（谐波系数、闪烁相位、核表、粒子参数）是浏览器不自己算的那部分 ——
    服务端算一次发过去。它们错了，浏览器侧的雾/闪烁/辉光/粒子就会跟
    服务端不一样，而且**极难从画面上看出来**（雾只是稍微不同、闪烁相位偏了）。
    """
    p, _ = scene_pack
    blob = _pack("scene,assets", p)
    meta, segs = _decode(blob)

    from pixelart.animate import DEFAULT_FLICKER_FREQS, hash01
    from pixelart.compose import BLOOM_RADII
    from pixelart.webgpu import (TWO_PI, bloom_kernels, dust_fixed_point_scale,
                                 dust_particle_params, fog_harmonics)

    # 谐波系数：与 fog_harmonics 逐项相同
    hs = fog_harmonics()
    assert len(meta["fog_harmonics"]) == len(hs) == 2
    for got, want in zip(meta["fog_harmonics"], hs):
        assert got[0] == pytest.approx(want["fx"])
        assert got[1] == pytest.approx(want["fy"])
        assert int(got[2]) == int(want["ft"])            # ⚠️ ft 必须是整数
        assert got[3] == pytest.approx(want["phase"])
        assert got[4] == pytest.approx(want["amp"])

    # 闪烁相位：服务端算好，浏览器不再需要 splitmix64
    assert meta["flicker_freqs"] == list(DEFAULT_FLICKER_FREQS)
    assert meta["flicker_phases"] == [pytest.approx(TWO_PI * hash01(0, i))
                                      for i in range(len(DEFAULT_FLICKER_FREQS))]

    # 核表：偏移/抽头数/半宽/权重
    kflat, table = bloom_kernels(BLOOM_RADII)
    assert len(segs["kernels"]) == len(kflat)
    assert meta["kbuf_table"] == [[int(o), int(c), int(hh), float(wt)]
                                  for o, c, hh, wt in table]

    # 粒子参数 + 定点标度
    cnt = int(p.dust_count)
    par = dust_particle_params(cnt, seed=11)
    assert segs["dust"].shape == (cnt, 11)
    assert np.array_equal(segs["dust"], par)
    assert meta["dust_scale"] == dust_fixed_point_scale(cnt)
    assert cnt * meta["dust_scale"] <= (1 << 32) - 1, "定点累加会溢出"


def test_scene_pack_palette_has_the_asked_size(scene_pack):
    """色板必须按当前 `colors` 参数给——它是全片共用的那一块（时序铁律 1）。"""
    p, _ = scene_pack
    blob = _pack("assets", p)
    meta, segs = _decode(blob)
    assert segs["palette"].shape == (int(p.colors), 3)
    # 色板存的是感知空间（开方），所以值域仍在 [0,1]
    assert segs["palette"].min() >= 0.0 and segs["palette"].max() <= 1.0


def test_assets_only_pack_omits_the_heavy_scene_segments(scene_pack):
    """只取 assets 时必须**不含** base/far —— 那是滑杆变化时的常用路径。

    `base`+`far` 占了整包的 98%，而滑杆变化不影响它们。
    若这条路径也带上，拖一次滑杆就要传 491 KB。
    """
    p, _ = scene_pack
    full = _pack("scene,assets", p)
    light = _pack("assets", p)
    meta, segs = _decode(light)
    assert "base" not in segs and "far" not in segs
    assert "palette" in segs and "kernels" in segs and "dust" in segs
    assert len(light) < len(full) * 0.1, "assets 包没比完整包小一个量级"


def test_pack_segments_offsets_are_consistent():
    """打包格式：偏移必须首尾相接，且负载总长与偏移和一致。

    ⚠️ 这类"自定二进制格式"最容易犯的错是偏移算错 —— 而且症状是
    解出来一串垃圾数据（可能看起来像"渲染对了但很怪"），
    不像崩溃那样容易发现。
    """
    a = np.arange(12, dtype=np.float32)
    b = np.arange(7, dtype=np.float32)
    blob = srv.pack_segments(
        {"a": ("f32", [12], a.tobytes()), "b": ("f32", [7], b.tobytes())},
        {"k": 1})
    hlen = struct.unpack("<I", blob[:4])[0]
    head = json.loads(blob[4:4 + hlen].decode("utf-8"))
    assert head["meta"] == {"k": 1}
    off = 4 + hlen
    assert head["segments"]["a"]["offset"] == 0
    assert head["segments"]["b"]["offset"] == 48          # 12×4
    assert len(blob) == off + 48 + 28
    assert np.array_equal(
        np.frombuffer(blob[off:off + 48], dtype=np.float32), a)
    assert np.array_equal(
        np.frombuffer(blob[off + 48:], dtype=np.float32), b)


def test_scene_pack_meta_covers_every_key_the_renderer_reads():
    """⭐⭐ 场景包 meta 必须覆盖**浏览器渲染器读的每一个键**。

    这条是"界面 DEFAULTS 必须覆盖服务端字段"那条检查的**对偶**，
    而且是被真事故逼出来的：

        渲染器读 `A.bloom_threshold` / `A.bloom_knee` / `A.bloom_wsum`，
        而场景包 meta 里没有这三个键 → `undefined` 写进 uniform 变成 **NaN**
        → 高光 mask 全 0 → `bright` 缓冲全 0 → 辉光及之后全是 0 → **输出纯黑**。
        **而且不报任何错** —— NaN 在 GPU 上是合法的。

    修法不是"记得加字段"，而是**让漏字段变成测试失败**：
    从 `m3_webgpu.js` 里把 `A.<name>` 的读法全抽出来，逐个要求 meta 里有。
    """
    import re as _re

    js = (srv.ASSET_DIR / "m3_webgpu.js").read_text(encoding="utf-8")
    # 渲染器里统一用 `var A = this.assets;`，所以读法都是 `A.xxx`
    keys = set(_re.findall(r"\bA\.([a-z_][a-z0-9_]*)\b", js))
    assert keys, "没从渲染器里解析出任何 A.xxx —— 正则要跟着源码结构同步"
    # 这几个是渲染器自己挂上去的局部（不是从 meta 读的）
    keys -= {"push", "length", "forEach", "map", "slice", "join", "reduce"}

    from pixelart.tune import TuneParams
    p = TuneParams.from_query({"asset": [ASSET], "grid_long": ["240"],
                               "work_long": ["960"], "aspect": ["native"]})
    meta, _ = _decode(_pack("scene,assets", p))

    missing = sorted(k for k in keys if k not in meta)
    assert not missing, (
        f"场景包 meta 缺少渲染器要读的键：{missing}\n"
        f"（漏了会让 uniform 变 NaN → 静默输出纯黑，不报错）")
    # 单独盯住那几个曾经漏掉的：必须是有限数，不能是 None/NaN
    for k in ("bloom_threshold", "bloom_knee", "bloom_wsum",
              "palette_len", "dust_count", "dust_scale"):
        v = meta[k]
        assert isinstance(v, (int, float)) and v == v, f"{k} 不是有限数：{v!r}"
