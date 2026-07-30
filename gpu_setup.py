"""
Puts the pip-installed CUDA/cuDNN runtime DLLs (nvidia-cudnn-cu12,
nvidia-cublas-cu12, nvidia-cuda-runtime-cu12) on PATH so onnxruntime-gpu's
CUDAExecutionProvider can load them.

Windows has no rpath equivalent — onnxruntime's LoadLibrary call for
onnxruntime_providers_cuda.dll fails silently (falls back to CPU) unless
these directories are already on PATH *before* onnxruntime is imported.
Must be imported before `import onnxruntime` (directly, or transitively via
kokoro_onnx / moonshine_onnx) happens anywhere in the process.
"""

import os
from pathlib import Path

_SITE_PACKAGES = Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "nvidia"

_DLL_DIRS = [
    _SITE_PACKAGES / "cudnn" / "bin",
    _SITE_PACKAGES / "cublas" / "bin",
    _SITE_PACKAGES / "cuda_runtime" / "bin",
    _SITE_PACKAGES / "cuda_nvrtc" / "bin",
    _SITE_PACKAGES / "cufft" / "bin",
    _SITE_PACKAGES / "nvjitlink" / "bin",
]

_done = False


def setup_cuda_path() -> None:
    global _done
    if _done:
        return
    for d in _DLL_DIRS:
        if d.exists():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    _done = True


setup_cuda_path()
