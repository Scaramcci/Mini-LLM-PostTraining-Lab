"""Print cleanup guidance for generated outputs.

This script intentionally avoids deleting files automatically. In the first
month of learning, explicit cleanup is safer than hidden cleanup.
"""

from __future__ import annotations


def main() -> None:
    print("Generated outputs to review manually:")
    print("- 01_nanogpt_pretrain/out*/")
    print("- checkpoints/")
    print("- outputs/")
    print("- runs/")
    print("- wandb/")
    print("Delete only after important results are recorded in reports/experiment_log.md.")


if __name__ == "__main__":
    main()
