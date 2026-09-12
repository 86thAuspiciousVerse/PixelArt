"""WGSL 移植的数值验收 —— 用 wgpu 无头执行着色器，与 numpy 参考实现比对。

═══ 为什么这个测试必须存在 ═══

把渲染从 Python 搬到 WebGPU 是个大搬家。搬错了不会崩，只会"看起来差不多
但某处不对"：雾的指数曲线差 2%、色板吸附在边界翻了个色、抖动相位偏一格 ——
这些都不会让人觉得"坏了"，只会让人觉得"好像不如以前好看"。

**所以验收必须是数值比对，不是看图。**

═══ 为什么能这么做 ═══

`wgpu-py` 暴露的是与 WebGPU 同一套 API，而本机可用后端里有 **Vulkan 和 D3D12**
—— 这正是浏览器在 Windows 上会走的两个后端。所以这里跑出来的结果，
和用户浏览器里的是同一段驱动代码，不是模拟。

═══ ⚠️ 这类测试最大的风险是"测了个假东西" ═══

本仓库在写这些验收时踩过**三次"假象"**，每次都白追了一轮：

  1. 拿 t=1.0 的结果去比 t=0.37 的基线，然后宣称"循环没闭合"。
  2. 调 numpy 参考时**漏传一个参数**（strength），numpy 用默认值 0.85、
     GPU 用 0.55，于是把"最大差 2.2e-3"当成移植误差去追。
  3. 测 threshold/knee，但那两个值在 numpy 侧是**写死**的、根本不接受传参 ——
     拿 GPU 的 0.0 去比 numpy 的 0.48，差 1.7e-2 纯属自造。

**结论：参考实现的每一个参数都必须显式传全；比对的两个基线必须来自同一组输入。**
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "tools" / "wgsl_lab.py"

#: 全部阶段。端到端（chain）连起 6 个 pass 跑，最终 u8 必须逐位一致。
STAGES = ("fog", "volumetric", "bloom", "dust", "tail", "chain")


def _wgpu_available() -> tuple[bool, str]:
    try:
        import wgpu  # noqa: F401
    except ImportError:
        return False, "wgpu 未安装（pip install wgpu）"
    try:
        r = subprocess.run([sys.executable, "-c",
                            "import wgpu; wgpu.gpu.request_adapter_sync()"],
                           capture_output=True, text=True, timeout=90)
    except Exception as e:                                      # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if r.returncode != 0:
        return False, (r.stderr or "").strip()[-200:]
    return True, ""


@pytest.fixture(scope="module")
def gpu_ok() -> bool:
    ok, why = _wgpu_available()
    if not ok:
        pytest.skip(f"没有可用的 WGSL 执行环境：{why}")
    return True


@pytest.mark.parametrize("stage", STAGES)
def test_wgsl_stage_matches_numpy(stage: str, gpu_ok: bool):
    """每个阶段（以及端到端链路）都必须与 numpy 参考实现一致。

    判据由各阶段自己定，但原则是：
      · 纯比较 / 纯查表的阶段（色板吸附、抖动、边缘压暗）—— **必须逐位相同**，
        那些地方任何差异都是逻辑错误，不是精度问题。
      · 含超越函数（exp / pow / sin）的阶段 —— 用容差，
        但也应当落在 1e-5 量级（f32 的精度极限就在 1e-7 附近）。

    ⚠️ 把第二类用容差"放过"是危险的：真 bug 也会被放过。
      所以容差定得比必要值紧，并且单阶段之外还有一条**端到端 u8 逐位**的判据兜底。
    """
    r = subprocess.run([sys.executable, "-u", str(LAB), "--stage", stage],
                       capture_output=True, text=True, encoding="utf-8",
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, f"[{stage}] 验收不通过：\n{out[-2500:]}"
    assert "全部通过" in out, f"[{stage}] 输出里没有'全部通过'：\n{out[-2000:]}"


def test_chain_final_output_is_bit_identical(gpu_ok: bool):
    """⭐ 端到端：**最终 u8 输出必须逐位相同**。

    这是整个移植工作的核心判据。单阶段各自一致还不够 ——
    误差会累积，而最终产物是 32 色的 uint8 图，中间任何偏差都可能
    让某个像素吸附到另一个颜色上（那就是肉眼可见的"某个地方颜色变了"）。

    中间量允许差约 1 个 f32 ulp（fog 的 exp/pow 决定的），
    但**量化之后必须完全相同** —— 色板吸附本身就是最好的误差吸收器。
    """
    r = subprocess.run([sys.executable, "-u", str(LAB), "--stage", "chain"],
                       capture_output=True, text=True, encoding="utf-8",
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, f"端到端验收不通过：\n{out[-2500:]}"
    assert "逐位不一致 0/" in out, f"端到端 u8 不是逐位一致：\n{out[-2000:]}"


def test_chain_matches_production_server_closely(gpu_ok: bool):
    """⚠️ 这条不是"必须逐位"，而是**盯住那个数**。

    GPU 侧的辉光是真高斯，而现在的服务端用 PIL 的三次盒式近似 + 中途降 uint8，
    两者**本来就不会逐位相同**（见 ``compose.blur_float`` 的说明）。

    但这个数正是"用户换成浏览器渲染后会不会觉得画面变了"的答案，
    所以必须被盯住：一旦哪天它涨上去（比如有人改了参数、或者 PAL 换了实现），
    测试会立刻报出来 —— 而不是等用户说"怎么不一样了"。

    实测当前：0.03%（2/6144 像素），且都是落在色板边界上的像素。
    """
    r = subprocess.run([sys.executable, "-u", str(LAB), "--stage", "chain"],
                       capture_output=True, text=True, encoding="utf-8",
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out[-2000:]
    assert "B. 与现在的服务端" in out, "端到端报告里没有与服务端的对比"
    # 从报告里抓那个百分比
    import re as _re
    m = _re.search(r"不一致 (\d+)/(\d+) \(([\d.]+)%\)", out)
    assert m, f"没解析到服务端差异比例：\n{out[-1200:]}"
    share = float(m.group(3))
    assert share < 2.0, (
        f"GPU 链路与现在的服务端差了 {share}% —— 超过 2% 就要复核了。\n"
        f"可能原因：改了 bloom 的参数、换了模糊算子、或色板生成变了。\n"
        f"{out[-1500:]}")


def test_lab_reports_which_stages_exist(gpu_ok: bool):
    """验收台必须能列出阶段 —— 防止阶段被悄悄改名/删除而测试还在"通过"。"""
    r = subprocess.run([sys.executable, str(LAB), "--list"],
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0
    for s in STAGES:
        assert s in r.stdout, f"验收台里没有阶段 {s}"
