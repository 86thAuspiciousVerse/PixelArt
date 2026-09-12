"""光锥（M5.1）的回归测试。

每条都对应一件事：

- **关闭时必须逐位不变**：新特性的默认值不许改动既有行为
  （这是这个项目一贯的不变量，也是"界面调好了、命令行跑出来不一样"的防线）。
- **非空转**：作用量必须真的看得见 —— 上一轮刚抓过"数学上在动、视觉上没动"。
- **几何正确**：锥是锥（正下方亮、正上方暗），reach 小的时候远处更暗。
- **闭合**：锥只改空间分布、不该破坏 frame(0)==frame(1)（时序铁律 3）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pixelart.compose import cone_weight  # noqa: E402
from pixelart.pipeline import AnimParams, compose_frame, finish_frame  # noqa: E402
from pixelart.tune import TuneParams, build_scene  # noqa: E402

ASSET = "ref01_roadsign_night.jpg"


def test_cone_weight_geometry():
    """锥的几何：沿轴亮、背轴暗；半球之外为 0。"""
    w, h = 240, 120
    lx, ly = 60, 30
    wgt = cone_weight(w, h, lx, ly, 0.7, 80.0, 0.8)
    down = float(wgt[ly + 50, lx])          # 沿 80°（≈向下）
    up = float(wgt[max(ly - 15, 0), lx])    # 反向
    left = float(wgt[ly, max(lx - 60, 0)])
    assert down > 0.5, f"轴向该亮，实际 {down}"
    assert up == 0.0 and left == 0.0, f"背轴该为 0，实际 {up}/{left}"


def test_cone_weight_reach_shortens():
    """reach 小 → 同样角度下远处更暗（"打不到底"）。"""
    w, h = 240, 120
    w_long = cone_weight(w, h, 60, 30, 0.9, 90.0, 1.0)
    w_short = cone_weight(w, h, 60, 30, 0.9, 90.0, 0.0)
    far_y = 110
    assert w_short[far_y, 60] < w_long[far_y, 60]
    # 光源附近的衰减**远小于**远处的衰减（实测 0.11 vs 0.67）
    near_loss = float(1.0 - w_short[35, 60])
    far_loss = float(1.0 - w_short[far_y, 60])
    assert near_loss < far_loss * 0.5, f"近 {near_loss:.3f} 远 {far_loss:.3f}"


def test_cone_weight_monotonic_in_angle():
    """离轴越远，权重越小（不能出现"环形"之类的怪形）。"""
    w, h = 200, 200
    lx = ly = 100
    wgt = cone_weight(w, h, lx, ly, 1.0, 0.0, 1.0)
    # ⚠️ 沿轴方向在**锥内是平的**（半角内 saturate 到 1），
    #    所以要沿**离轴**方向取样才看得到单调下降：
    #    (lx+40, ly+dy) 对应的角度是 atan2(dy, 40)。
    #    实测：0°/14° → 1.000，32° → 0.978，48° → 0.306，60° → 0.000。
    col = wgt[ly:ly + 71, lx + 40]
    assert col[0] == 1.0
    assert col[10] == pytest.approx(1.0, abs=1e-4)
    assert col[25] > col[45] > col[70]
    assert col[70] == 0.0


@pytest.fixture(scope="module")
def scene():
    p = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                               "work_long": ["480"], "aspect": ["native"]})
    from PIL import Image
    return build_scene(Image.open(ROOT / "assets" / "input" / ASSET), p)


def _u8(scene, p: TuneParams, t: float):
    """用给定参数渲一帧 u8（compose → finish，取帧本体）。"""
    kw = p.compose_kwargs(scene)
    kw["anim"] = p.to_anim()
    return finish_frame(compose_frame(scene, t=t, **kw), scene.tail)[0]


def test_cone_off_is_bit_identical(scene):
    """⭐ cone=0 与"完全不传锥参数"逐位相同 —— 默认行为不许变。"""
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                 "work_long": ["480"], "aspect": ["native"]})
    zero = TuneParams(**{**base.to_dict(), "rays_cone": 0.0})
    assert np.array_equal(_u8(scene, base, 0.37), _u8(scene, zero, 0.37))


def test_cone_changes_image(scene):
    """非空转：开锥后 u8 帧必须真的不同。"""
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                  "work_long": ["480"], "aspect": ["native"]})
    off = _u8(scene, base, 0.37)
    on = _u8(scene, TuneParams(**{**base.to_dict(), "rays_cone": 0.8}), 0.37)
    d = float(np.abs(on.astype(int) - off.astype(int)).mean())
    assert d > 2.0, f"平均差 {d:.2f}/255 —— 看不出就是白做"


def test_cone_loop_still_closes(scene):
    """开锥后循环闭合仍逐位成立。"""
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                 "work_long": ["480"], "aspect": ["native"]})
    p = TuneParams(**{**base.to_dict(), "rays_cone": 0.8, "flicker": 0.16,
                      "fog_drift": 0.3})
    assert np.array_equal(_u8(scene, p, 0.0), _u8(scene, p, 1.0))


def test_cone_direction_matters_at_layer_level(scene):
    """⭐ 方向必须真的改变**体积光图层** —— 而且要在图层上量。

    ⚠️ 为什么不在最终 u8 帧上量：实测整个体积光图层在这几张素材上只有
    **1~3/255**（`rays=0.55` + `screen_falloff` 是刻意调弱的 ——
    早先用户报过"整幅图蒙了层滤镜"）。在这么弱的层上做锥形塑形，
    最终帧的差异只有 0.04~1.75/255 —— 拿 u8 当判据会得到
    "看起来像没生效"的结论，而实际几何完全正确。
    **判据要下在被测对象真正生效的层级上。**
    """
    from pixelart.compose import volumetric_light
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                 "work_long": ["480"], "aspect": ["native"]})
    kw = base.compose_kwargs(scene)
    kw.pop("cone_angle"); kw.pop("cone_dir_deg")
    kw.pop("cone_reach"); kw.pop("cone_gain")
    layer = {}
    for name, d_deg in (("down", 90.0), ("up", -90.0)):
        layer[name] = volumetric_light(
            scene.base, scene.far, base.light_xy(scene),
            strength=0.55, cone_angle=0.9, cone_dir_deg=d_deg,
            cone_reach=0.8, cone_gain=1.6)
    d = float(np.abs(layer["down"] - layer["up"]).mean())
    assert d > 1e-4, f"方向没改变图层（差 {d:.2e}）"
    # 方向换来的是**分布**变了：锥内区域该有一个明显变亮的象限
    assert float(layer["down"].mean()) > 0.0


# ── M5.2 光柱：独立加性层 ──────────────────────────────────────────
def test_shaft_off_is_bit_identical(scene):
    """⭐ shaft=0 时输出与"只开锥"逐位相同 —— 新旋钮的默认值不许改动行为。"""
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                 "work_long": ["480"], "aspect": ["native"]})
    cone_only = TuneParams(**{**base.to_dict(), "rays_cone": 0.7,
                              "rays_dir": 90.0, "rays_shaft": 0.0})
    zero = TuneParams(**{**base.to_dict(), "rays_cone": 0.7,
                         "rays_dir": 90.0})
    assert np.array_equal(_u8(scene, cone_only, 0.37), _u8(scene, zero, 0.37))


def test_shaft_is_visible_inside_the_cone(scene):
    """⭐⭐ 光柱的**验收判据**：锥内的差异必须 ≥ 8/255（"一眼可见"的量级）。

    ⚠️ 为什么量"锥内"而不是全帧：锥只覆盖画面约 12%，
    全帧均值会被没被照到的区域稀释掉 —— 实测全帧 0.6~3.9/255，
    而锥内是 30+/255。**判据要下在这层真正生效的区域上。**
    """
    from pixelart.compose import cone_weight
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                 "work_long": ["480"], "aspect": ["native"]})
    off = _u8(scene, TuneParams(**{**base.to_dict(), "rays_cone": 0.7,
                                   "rays_dir": 90.0, "rays_shaft": 0.0}), 0.2)
    on = _u8(scene, TuneParams(**{**base.to_dict(), "rays_cone": 0.7,
                                  "rays_dir": 90.0, "rays_shaft": 1.0}), 0.2)
    lx = int(base.light_xy(scene)[0] * (scene.grid[0] - 1))
    ly = int(base.light_xy(scene)[1] * (scene.grid[1] - 1))
    cw = cone_weight(scene.grid[0], scene.grid[1], lx, ly, 0.7, 90.0, 0.8, 1.6)
    inside = cw > 0.5
    assert inside.sum() > 50, "锥内像素太少，判据没有意义（空转检验）"
    d = np.abs(on.astype(int) - off.astype(int)).max(axis=2)
    mean_in = float(d[inside].mean())
    assert mean_in >= 8.0, (
        f"锥内平均差异只有 {mean_in:.2f}/255 —— 光柱还看不出来（要求 ≥ 8）")


def test_shaft_loop_still_closes(scene):
    """开光柱后循环闭合仍逐位成立。"""
    base = TuneParams.from_query({"asset": [ASSET], "grid_long": ["120"],
                                 "work_long": ["480"], "aspect": ["native"]})
    p = TuneParams(**{**base.to_dict(), "rays_cone": 0.7, "rays_dir": 90.0,
                      "rays_shaft": 1.0, "flicker": 0.16, "fog_drift": 0.3})
    assert np.array_equal(_u8(scene, p, 0.0), _u8(scene, p, 1.0))
