"""Print basic environment information for this project."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())

    try:
        import torch

        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
        if hasattr(torch.backends, "mps"):
            print("MPS available:", torch.backends.mps.is_available())
    except ImportError:
        print("PyTorch: not installed")

    try:
        import transformers

        print("Transformers:", transformers.__version__)
    except ImportError:
        print("Transformers: not installed")

    try:
        import datasets

        print("Datasets:", datasets.__version__)
    except ImportError:
        print("Datasets: not installed")


if __name__ == "__main__":
    main()
