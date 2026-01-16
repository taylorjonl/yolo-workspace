from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a new vision project skeleton."
    )
    parser.add_argument(
        "name",
        help="Project name (letters, numbers, hyphen, underscore).",
    )
    return parser.parse_args(argv)


def validate_name(name: str) -> None:
    if not VALID_NAME.fullmatch(name):
        raise ValueError(
            "Invalid project name. Use letters, numbers, hyphen, or underscore."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_name(args.name)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    project_root = repo_root / "projects" / args.name

    if project_root.exists():
        print(f"Project already exists: {project_root}", file=sys.stderr)
        return 1

    directories = [
        project_root / "artifacts" / "incoming",
        project_root / "artifacts" / "archive",
        project_root / "datasets",
        project_root / "experiments",
        project_root / "models" / "checkpoints",
        project_root / "models" / "exports",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)

    print(f"Created project skeleton at {project_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
