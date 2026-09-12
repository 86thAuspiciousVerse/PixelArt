"""项目路径解析。所有脚本都从这里取路径，不要自己拼相对路径。"""

from pathlib import Path

# src/pixelart/paths.py -> 仓库根目录
ROOT = Path(__file__).resolve().parents[2]

ASSETS = ROOT / "assets"
INPUT = ASSETS / "input"
MODELS = ROOT / "models"
OUT = ROOT / "out"
DOCS = ROOT / "docs"

DEPTH_MODEL_DIR = MODELS / "depth-anything-v2-small"
DEPTH_MODEL_ONNX = DEPTH_MODEL_DIR / "onnx" / "model.onnx"


def ensure_dir(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def out(*parts: str) -> Path:
    """在 out/ 下建档并返回路径, 例如 out("probe", "a.png")。"""
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
