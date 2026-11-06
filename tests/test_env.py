#!/usr/bin/env python
"""Step 0 驗收：環境冒煙測試。

檢查項：
1. lerobot 可 import（PYTHONPATH 指向 lerobot-main/src）
2. torch 可用 CUDA 且 2 張 GPU 可見
3. ffmpeg / ffprobe 可用
4. pyav / pyarrow 可 import

用法（必須用 embody 環境）：
    PYTHONPATH=/data/home/cloudsway/lerobot-main/src \
        /data/home/cloudsway/.conda/envs/embody/bin/python tests/test_env.py
"""
import shutil
import sys

FAILURES = []


def check(name: str, fn):
    try:
        detail = fn()
        print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        FAILURES.append(name)
        print(f"[FAIL] {name} — {type(e).__name__}: {e}")


def _lerobot():
    import lerobot  # noqa: F401

    return f"loaded from {lerobot.__file__}"


def _torch_cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    assert n == 2, f"expected 2 GPUs, got {n}"
    return f"{n} x {names[0]}"


def _ffmpeg():
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    assert ff and fp, f"ffmpeg={ff} ffprobe={fp}"
    return f"{ff}, {fp}"


def _pyav_pyarrow():
    import av  # noqa: F401
    import pyarrow  # noqa: F401

    import av

    return f"av {av.__version__}, pyarrow {pyarrow.__version__}"


def main():
    check("import lerobot", _lerobot)
    check("torch + 2x CUDA GPU", _torch_cuda)
    check("ffmpeg/ffprobe", _ffmpeg)
    check("pyav + pyarrow", _pyav_pyarrow)

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("\nAll environment checks passed.")


if __name__ == "__main__":
    main()
