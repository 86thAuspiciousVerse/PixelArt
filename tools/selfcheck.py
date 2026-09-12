"""环境自检。

检查四件事：
  1. 必需依赖是否齐全
  2. 可选依赖（有则更好）
  3. 深度模型权重是否就位
  4. 端到端冒烟测试：用一张合成小图跑通一次 render_frame

设计成独立脚本而不是 ``python -c "..."``，是因为
Windows PowerShell 5.1 在把参数传给原生 exe 时不会正确转义内嵌双引号，
多行 + 引号的 ``-c`` 参数会被截断（症状：``SyntaxError: '(' was never closed``）。

失败时以非零退出码结束，便于脚本调用。

用法::

    .\\.venv\\Scripts\\python.exe tools\\selfcheck.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REQUIRED = ("numpy", "PIL", "cv2", "onnxruntime", "pixelart")
OPTIONAL = ("pytest", "imageio_ffmpeg")
MIN_MODEL_MB = 50.0
BAR = "=" * 64


def main() -> int:
    ok = True
    print(BAR)
    print("pixelart self-check")
    print(BAR)

    print(f"\npython : {sys.version.split()[0]}")
    print(f"exe    : {sys.executable}")
    if sys.version_info < (3, 11):
        ok = False
        print("  FAIL  need Python >= 3.11")

    print("\n[1] required packages")
    for name in REQUIRED:
        try:
            m = importlib.import_module(name)
            print(f"  ok    {name:<16s} {getattr(m, '__version__', '')}")
        except Exception as e:
            ok = False
            print(f"  FAIL  {name:<16s} {type(e).__name__}: {e}")

    print("\n[2] optional packages")
    for name in OPTIONAL:
        try:
            m = importlib.import_module(name)
            print(f"  ok    {name:<16s} {getattr(m, '__version__', '')}")
        except Exception:
            print(f"  --    {name:<16s} not installed (optional)")

    print("\n[3] model weights")
    try:
        from pixelart.paths import DEPTH_MODEL_ONNX

        if DEPTH_MODEL_ONNX.exists():
            mb = DEPTH_MODEL_ONNX.stat().st_size / 1e6
            if mb < MIN_MODEL_MB:
                ok = False
                print(f"  FAIL  {DEPTH_MODEL_ONNX.name} only {mb:.1f} MB (expected ~99 MB)")
                print("        re-run: python tools/fetch_models.py --force")
            else:
                print(f"  ok    depth-anything-v2-small  {mb:.1f} MB")
        else:
            ok = False
            print(f"  FAIL  missing: {DEPTH_MODEL_ONNX}")
            print("        run: python tools/fetch_models.py")
    except Exception as e:
        ok = False
        print(f"  FAIL  {type(e).__name__}: {e}")

    print("\n[4] end-to-end smoke test (synthetic image)")
    try:
        import numpy as np
        from PIL import Image

        from pixelart.pixelate import PixelTail, render_frame

        yy, xx = np.mgrid[0:180, 0:320]
        arr = np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8)
        out, pal = render_frame(Image.fromarray(arr), (80, 45), PixelTail(n_colors=16))
        if out.size != (80, 45):
            raise AssertionError(f"unexpected output size {out.size}")
        if pal.ndim != 2 or pal.shape[1] != 3:
            raise AssertionError(f"unexpected palette shape {pal.shape}")
        print(f"  ok    render_frame -> {out.size[0]}x{out.size[1]}, palette {len(pal)} colors")
    except Exception as e:
        ok = False
        print(f"  FAIL  {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n" + BAR)
    print("RESULT: " + ("PASS" if ok else "FAIL"))
    print(BAR)
    if not ok:
        print("\nHints:")
        print("  missing packages -> pip install numpy pillow opencv-python-headless onnxruntime")
        print("  missing model    -> python tools/fetch_models.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
