"""分析阶段：单目深度（后续会加语义 / 发光遮罩）。

这是整条管线里**唯一"重"的一段**，结果必须缓存到 ``out/`` 或 ``cache/``，
调参时绝不要重跑。

模型：Depth Anything V2 Small（ONNX，来自 onnx-community/depth-anything-v2-small）。
预处理遵循 DPTImageProcessor 约定：保持宽高比缩放到 518 内、边长为 14 的倍数、
rescale 1/255、ImageNet mean/std 归一化。
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .paths import DEPTH_MODEL_ONNX

__all__ = ["DepthEstimator", "colorize_depth", "DEPTH_MODEL_HINT"]

DPT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DPT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ENSURE_MULTIPLE_OF = 14
TARGET_SIDE = 518

#: 该模型的输出是 **视差类** 量：值越大 = 越近。
#: 因此 ``predict()`` 归一化后 **1.0 = 最近, 0.0 = 最远**。
#: 若下游约定 0=近，请自行 ``1 - d``（见 :meth:`DepthEstimator.predict_far`）。
DEPTH_MODEL_HINT = "relative/disparity-like: output larger == closer"

#: 高亮提示：模型权重不入库，需先跑 tools/fetch_models.py
DEPTH_MODEL_MISSING = (
    f"找不到深度模型: {DEPTH_MODEL_ONNX}\n"
    "请先运行:  python tools/fetch_models.py"
)


class DepthEstimator:
    """单目深度估计。构造一次、复用多次（会话初始化有开销）。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        threads: int | None = None,
    ) -> None:
        path = Path(model_path) if model_path else DEPTH_MODEL_ONNX
        if not path.exists():
            raise FileNotFoundError(DEPTH_MODEL_MISSING)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = int(threads)

        self.session = ort.InferenceSession(
            str(path), sess_options=opts,
            providers=providers or ort.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # ------------------------------------------------------------------ #
    def _preprocess(self, img: Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        scale = TARGET_SIDE / max(w, h)
        nw = max(ENSURE_MULTIPLE_OF, int(round(w * scale / ENSURE_MULTIPLE_OF)) * ENSURE_MULTIPLE_OF)
        nh = max(ENSURE_MULTIPLE_OF, int(round(h * scale / ENSURE_MULTIPLE_OF)) * ENSURE_MULTIPLE_OF)

        small = img.resize((nw, nh), Image.Resampling.BICUBIC)
        a = np.asarray(small, dtype=np.float32) / 255.0
        a = (a - DPT_MEAN) / DPT_STD
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None]), (nw, nh)

    def predict(self, img: Image.Image) -> np.ndarray:
        """返回 (H, W) float32 in [0, 1]，**1.0 = 最近**，尺寸与原图一致。"""
        x, (nw, nh) = self._preprocess(img)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        d = np.asarray(out, dtype=np.float32).reshape(nh, nw)

        # 回到原图尺寸
        d = np.asarray(
            Image.fromarray(d, mode="F").resize(img.size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        lo, hi = float(d.min()), float(d.max())
        return (d - lo) / max(hi - lo, 1e-6)

    def predict_far(self, img: Image.Image) -> np.ndarray:
        """返回 (H, W) float32 in [0, 1]，**0.0 = 最近，1.0 = 最远**（下游合成用的约定）。"""
        return 1.0 - self.predict(img)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DepthEstimator in={self.input_name} out={self.output_name}>"


# ---------------------------------------------------------------------- #
# 伪彩渲染：优先用 OpenCV 的 TURBO，没有 cv2 时退化为内置的 numpy 近似实现。
# 这样深度检查这条路径不强制依赖 opencv。
_TURBO_ANCHORS = [
    (0.00, (0.19, 0.07, 0.23)),
    (0.13, (0.27, 0.45, 0.94)),
    (0.25, (0.10, 0.72, 0.83)),
    (0.38, (0.15, 0.90, 0.56)),
    (0.50, (0.51, 0.98, 0.32)),
    (0.63, (0.85, 0.94, 0.21)),
    (0.75, (0.99, 0.75, 0.17)),
    (0.88, (0.95, 0.42, 0.10)),
    (1.00, (0.48, 0.02, 0.01)),
]


def _turbo_numpy(x: np.ndarray) -> np.ndarray:
    """TURBO 色图的纯 numpy 近似（锚点线性插值）。x in [0,1] → uint8 (...,3)。"""
    xs = np.asarray(x, dtype=np.float32)
    knots = np.array([a for a, _ in _TURBO_ANCHORS], dtype=np.float32)
    out = np.empty(xs.shape + (3,), dtype=np.float32)
    for c in range(3):
        vals = np.array([v[c] for _, v in _TURBO_ANCHORS], dtype=np.float32)
        out[..., c] = np.interp(xs, knots, vals)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def colorize_depth(value01: np.ndarray, invert: bool = False) -> np.ndarray:
    """把标量图渲染成 TURBO 伪彩，便于目视检查。返回 uint8 (H, W, 3)。

    约定：**输入值越大 → 颜色越暖（红/黄）**。语义由调用方决定，例如
    ``colorize_depth(near01)`` 得到「暖=近」，``colorize_depth(near01, invert=True)``
    得到「暖=远」。

    优先用 OpenCV 的 TURBO；没有 cv2 时退化为内置的 numpy 近似。
    """
    x = np.clip(np.asarray(value01, dtype=np.float32), 0.0, 1.0)
    if invert:
        x = 1.0 - x
    try:
        import cv2

        u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)[:, :, ::-1]
    except ImportError:  # pragma: no cover - 取决于运行环境
        return _turbo_numpy(x)
