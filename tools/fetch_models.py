"""从镜像下载模型权重（不入库，体积大）。

huggingface.co 在部分网络环境下不可达；``hf-mirror.com`` 是其完整镜像。
可用环境变量 ``HF_ENDPOINT`` 覆盖。

用法::

    python tools/fetch_models.py
    python tools/fetch_models.py --model segformer-b0     # 预留
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelart.paths import DEPTH_MODEL_DIR, MODELS  # noqa: E402

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

MODELS_Spec = {
    "depth-anything-v2-small": {
        "repo": "onnx-community/depth-anything-v2-small",
        "files": ["config.json", "preprocessor_config.json", "onnx/model.onnx"],
        "dest": DEPTH_MODEL_DIR,
    },
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print(f"  [ok  ] {dest.relative_to(MODELS.parent)}  {dest.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="depth-anything-v2-small", choices=sorted(MODELS_Spec))
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = ap.parse_args()

    spec = MODELS_Spec[args.model]
    base = f"{args.endpoint.rstrip('/')}/{spec['repo']}/resolve/main"
    print(f"model : {args.model}\nrepo  : {spec['repo']}\nfrom  : {base}\n")
    for rel in spec["files"]:
        download(f"{base}/{rel}", spec["dest"] / rel)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
