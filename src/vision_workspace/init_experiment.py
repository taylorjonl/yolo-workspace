from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from vision_workspace.cli_helpers import (
    resolve_dataset_dir,
    resolve_dataset_dir_from_cwd,
    resolve_project_name,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a training experiment.")
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument(
        "--experiment", type=str, help="Experiment name (exp_001_baseline)"
    )
    parser.add_argument("--dataset", type=str, help="Dataset version or path")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Model checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def resolve_experiment_name(project_root: Path, experiment: str | None) -> str | None:
    if experiment:
        return experiment

    cwd = Path.cwd().resolve()
    if cwd.parent.name == "experiments" and cwd.parent.parent == project_root:
        return cwd.name

    if not sys.stdin.isatty():
        return None

    while True:
        raw = input("Experiment name: ").strip()
        if raw:
            return raw
        print("Experiment name is required.", file=sys.stderr)


def load_class_names(dataset_dir: Path) -> list[str] | None:
    data_yaml_path = dataset_dir / "data.yaml"
    if data_yaml_path.exists():
        try:
            data_config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            data_config = None
        if isinstance(data_config, dict):
            names = data_config.get("names")
            if isinstance(names, list):
                return [str(name) for name in names]

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    categories = manifest.get("categories") if isinstance(manifest, dict) else None
    if not isinstance(categories, list):
        return None

    names_by_index: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        index = category.get("index")
        name = category.get("name")
        if isinstance(index, int) and name is not None:
            names_by_index[index] = str(name)

    if not names_by_index:
        return None

    max_index = max(names_by_index)
    return [names_by_index.get(i, f"Class {i}") for i in range(max_index + 1)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir_from_cwd = resolve_dataset_dir_from_cwd(repo_root)
    project_name = resolve_project_name(
        args.project, repo_root, context_dir=dataset_dir_from_cwd
    )
    if not project_name:
        print(
            "Project not specified. Run from projects/<name> or pass the project name.",
            file=sys.stderr,
        )
        return 2

    project_root = repo_root / "projects" / project_name
    if not project_root.exists():
        print(f"Project not found: {project_root}", file=sys.stderr)
        return 1

    experiment_name = resolve_experiment_name(project_root, args.experiment)
    if not experiment_name:
        print("Experiment not specified.", file=sys.stderr)
        return 2

    if args.dataset is None and dataset_dir_from_cwd is not None:
        dataset_dir = dataset_dir_from_cwd
    else:
        dataset_dir = resolve_dataset_dir(project_root, args.dataset)
    if dataset_dir is None:
        return 1

    class_names = load_class_names(dataset_dir)
    if class_names is None:
        print(f"No class names found in {dataset_dir}.", file=sys.stderr)
        return 1

    experiment_dir = project_root / "experiments" / experiment_name
    if experiment_dir.exists():
        print(f"Experiment already exists: {experiment_dir}", file=sys.stderr)
        return 1

    experiment_dir.mkdir(parents=True, exist_ok=False)

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    (experiment_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    train_yaml = {
        "model": args.model,
        "data": "data.yaml",
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
    }
    (experiment_dir / "train.yaml").write_text(
        yaml.safe_dump(train_yaml, sort_keys=False), encoding="utf-8"
    )

    print(f"Initialized experiment at {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
