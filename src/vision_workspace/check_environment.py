from __future__ import annotations

import argparse
import shutil
import sys
from importlib import metadata
from pathlib import Path


REQUIRED_DISTRIBUTIONS: list[tuple[str, str]] = [
    ("ultralytics", "YOLO framework"),
    ("opencv-python", "OpenCV image processing"),
    ("numpy", "Numerical computing"),
    ("Pillow", "Image library"),
    ("PyYAML", "YAML configuration"),
    ("tqdm", "Progress bars"),
    ("onnx", "ONNX model format"),
    ("onnxruntime", "ONNX Runtime inference"),
]

OPTIONAL_DISTRIBUTIONS: list[tuple[str, str]] = [
    ("torch", "PyTorch deep learning"),
    ("torchvision", "PyTorch vision utilities"),
    ("torchaudio", "PyTorch audio utilities"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local environment prerequisites.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path used for disk space check (default: current directory).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=10.0,
        help="Minimum free disk space in GB (default: 10).",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU/CUDA checks (avoids importing torch).",
    )
    return parser.parse_args(argv)


def check_python_version(min_major: int = 3, min_minor: int = 10) -> bool:
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")

    if (version.major, version.minor) >= (min_major, min_minor):
        print(f"  Python version >= {min_major}.{min_minor}")
        return True

    print(f"  Python version < {min_major}.{min_minor}", file=sys.stderr)
    return False


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_distributions(items: list[tuple[str, str]], *, title: str) -> bool:
    print(f"{title}:")
    ok = True
    for dist_name, description in items:
        version = distribution_version(dist_name)
        if version is None:
            print(f"  {dist_name:15s} NOT INSTALLED - {description}", file=sys.stderr)
            ok = False
        else:
            print(f"  {dist_name:15s} {version:10s} - {description}")
    return ok


def check_disk_space(path: Path, min_free_gb: float) -> bool:
    try:
        total, used, free = shutil.disk_usage(path)
    except OSError as exc:
        print(f"Disk space: failed to query ({exc})")
        return True

    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)

    print("Disk space:")
    print(f"  Path: {path}")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used:  {used_gb:.1f} GB")
    print(f"  Free:  {free_gb:.1f} GB")

    if free_gb < min_free_gb:
        print(f"  WARNING: Less than {min_free_gb:.1f} GB free.", file=sys.stderr)
        return False
    print(f"  Free space >= {min_free_gb:.1f} GB")
    return True


def check_torch_gpu() -> bool:
    try:
        import torch
    except ImportError:
        print("GPU/CUDA:")
        print("  torch is not installed; skipping GPU checks.")
        return True

    print("GPU/CUDA:")
    print(f"  torch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if not cuda_available:
        return True

    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU memory: {total_mem:.1f} GB")
    return True


def check_onnxruntime_providers() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime:")
        print("  onnxruntime is not installed.", file=sys.stderr)
        return False

    providers = ort.get_available_providers()
    print("ONNX Runtime:")
    print(f"  Providers: {', '.join(providers)}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    results: dict[str, bool] = {}

    results["python"] = check_python_version()
    print()

    results["required"] = check_distributions(REQUIRED_DISTRIBUTIONS, title="Required packages")
    print()

    _ = check_distributions(OPTIONAL_DISTRIBUTIONS, title="Optional packages")
    print()

    results["onnx_providers"] = check_onnxruntime_providers()
    print()

    if args.skip_gpu:
        print("GPU/CUDA:")
        print("  Skipped.")
        results["gpu"] = True
    else:
        results["gpu"] = check_torch_gpu()
    print()

    results["disk"] = check_disk_space(args.path, args.min_free_gb)
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)

    critical = ["python", "required", "onnx_providers"]
    failed = [name for name in critical if not results.get(name, False)]
    if not failed:
        print("All critical checks passed.")
        return 0

    print("Critical checks failed:", file=sys.stderr)
    for name in failed:
        print(f"  - {name}", file=sys.stderr)
    print("Install dependencies with:", file=sys.stderr)
    print("  uv pip install -e .", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

