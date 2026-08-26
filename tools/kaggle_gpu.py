#!/usr/bin/env python3
"""VS Code task helper for submitting this project to a Kaggle GPU notebook."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
KERNEL_DIR = WORKSPACE / "kaggle_gpu"
METADATA_PATH = KERNEL_DIR / "kernel-metadata.json"
OUTPUT_DIR = KERNEL_DIR / "output"
KERNEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*$")


def read_metadata() -> dict[str, object]:
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing {METADATA_PATH.relative_to(WORKSPACE)}.") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {METADATA_PATH.relative_to(WORKSPACE)}: {exc}") from exc


def get_kernel_id(metadata: dict[str, object]) -> str:
    kernel_id = metadata.get("id")
    if (
        not isinstance(kernel_id, str)
        or kernel_id.startswith("YOUR_KAGGLE_USERNAME/")
        or not KERNEL_ID_PATTERN.fullmatch(kernel_id)
    ):
        raise SystemExit(
            "Configure a real Kaggle notebook ID first: run the VS Code task "
            "'Kaggle: Configure Notebook ID'."
        )
    return kernel_id


def kaggle_command(*args: str) -> list[str]:
    return [str(Path(sys.executable).with_name("kaggle")), *args]


def run_kaggle(*args: str) -> None:
    completed = subprocess.run(kaggle_command(*args), cwd=WORKSPACE, check=False)
    raise SystemExit(completed.returncode)


def configure(kernel_id: str) -> None:
    if not KERNEL_ID_PATTERN.fullmatch(kernel_id):
        raise SystemExit("Notebook ID must have the form username/notebook-slug.")

    metadata = read_metadata()
    metadata["id"] = kernel_id
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Configured remote Kaggle notebook: {kernel_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    configure_parser = subcommands.add_parser("configure", help="set the Kaggle username/notebook slug")
    configure_parser.add_argument("--kernel", required=True, help="username/notebook-slug")
    subcommands.add_parser("push", help="submit the notebook to a Kaggle T4 GPU")
    subcommands.add_parser("status", help="show the latest Kaggle run status")
    subcommands.add_parser("logs", help="stream logs from the Kaggle run")
    subcommands.add_parser("output", help="download latest Kaggle outputs")
    subcommands.add_parser("quota", help="show remaining Kaggle accelerator quota")
    args = parser.parse_args()

    if args.command == "configure":
        configure(args.kernel)
        return
    if args.command == "quota":
        run_kaggle("quota")
        return

    metadata = read_metadata()
    kernel_id = get_kernel_id(metadata)
    if args.command == "push":
        run_kaggle(
            "kernels",
            "push",
            "-p",
            str(KERNEL_DIR),
            "--accelerator",
            "NvidiaTeslaT4",
        )
    elif args.command == "status":
        run_kaggle("kernels", "status", kernel_id)
    elif args.command == "logs":
        run_kaggle("kernels", "logs", "--follow", kernel_id)
    elif args.command == "output":
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_kaggle("kernels", "output", kernel_id, "-p", str(OUTPUT_DIR), "--force")


if __name__ == "__main__":
    main()
