from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml


DATASET_DIR_PATTERN = re.compile(r"^dataset_v(\d+)$")


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


def extract_project_from_path(path: Path, repo_root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(repo_root)
    except ValueError:
        return None
    parts = relative.parts
    if "projects" in parts:
        index = parts.index("projects")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def list_project_dirs(projects_dir: Path) -> list[Path]:
    if not projects_dir.exists():
        return []
    project_dirs = [path for path in projects_dir.iterdir() if path.is_dir()]
    return sorted(project_dirs, key=lambda path: path.name)


def select_project_name(repo_root: Path) -> str | None:
    projects_dir = repo_root / "projects"
    project_dirs = list_project_dirs(projects_dir)
    if not project_dirs:
        return None

    if len(project_dirs) == 1:
        return project_dirs[0].name

    if not sys.stdin.isatty():
        print("Multiple projects found; pass the project name.", file=sys.stderr)
        return None

    print("Available projects:")
    for idx, project_dir in enumerate(project_dirs, start=1):
        print(f"  [{idx}] {project_dir.name}")

    default_index = len(project_dirs)
    while True:
        raw = input(
            f"Select project [{project_dirs[default_index - 1].name}]: "
        ).strip()
        if not raw:
            return project_dirs[default_index - 1].name
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a project number.", file=sys.stderr)
            continue
        if 1 <= choice <= len(project_dirs):
            return project_dirs[choice - 1].name
        print("Selection out of range.", file=sys.stderr)


def resolve_project_name(
    project: str | None,
    repo_root: Path,
    dataset_dir: Path | None,
) -> str | None:
    if project:
        return project

    if dataset_dir:
        project_name = extract_project_from_path(dataset_dir, repo_root)
        if project_name:
            return project_name

    cwd = Path.cwd().resolve()
    project_name = extract_project_from_path(cwd, repo_root)
    if project_name:
        return project_name

    return select_project_name(repo_root)


def dataset_sort_key(dataset_dir: Path) -> tuple[int, int | str]:
    match = DATASET_DIR_PATTERN.match(dataset_dir.name)
    if match:
        return (0, int(match.group(1)))
    return (1, dataset_dir.name)


def list_dataset_dirs(datasets_dir: Path) -> list[Path]:
    if not datasets_dir.exists():
        return []
    dataset_dirs = [path for path in datasets_dir.iterdir() if path.is_dir()]
    return sorted(dataset_dirs, key=dataset_sort_key)


def select_dataset_dir(datasets_dir: Path) -> Path | None:
    dataset_dirs = list_dataset_dirs(datasets_dir)
    if not dataset_dirs:
        print(f"No datasets found in {datasets_dir}", file=sys.stderr)
        return None

    if len(dataset_dirs) == 1:
        return dataset_dirs[0]

    if not sys.stdin.isatty():
        print("Multiple datasets found; pass --dataset to select one.", file=sys.stderr)
        return None

    print("Available datasets:")
    for idx, dataset_dir in enumerate(dataset_dirs, start=1):
        print(f"  [{idx}] {dataset_dir.name}")

    default_index = len(dataset_dirs)
    while True:
        raw = input(
            f"Select dataset [{dataset_dirs[default_index - 1].name}]: "
        ).strip()
        if not raw:
            return dataset_dirs[default_index - 1]
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a dataset number.", file=sys.stderr)
            continue
        if 1 <= choice <= len(dataset_dirs):
            return dataset_dirs[choice - 1]
        print("Selection out of range.", file=sys.stderr)


def resolve_dataset_dir(project_root: Path, dataset_arg: str | None) -> Path | None:
    if dataset_arg:
        candidate = Path(dataset_arg)
        if candidate.exists():
            if candidate.is_dir():
                return candidate
            print(f"Dataset path is not a directory: {candidate}", file=sys.stderr)
            return None
        candidate = project_root / "datasets" / dataset_arg
        if candidate.exists():
            if candidate.is_dir():
                return candidate
            print(f"Dataset path is not a directory: {candidate}", file=sys.stderr)
            return None
        print(f"Dataset not found: {dataset_arg}", file=sys.stderr)
        return None

    return select_dataset_dir(project_root / "datasets")


def resolve_dataset_dir_from_cwd(repo_root: Path) -> Path | None:
    current = Path.cwd().resolve()
    while True:
        if current.parent.name == "datasets" and DATASET_DIR_PATTERN.match(
            current.name
        ):
            return current
        if current == repo_root or current.parent == current:
            break
        current = current.parent
    return None


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
    project_name = resolve_project_name(args.project, repo_root, dataset_dir_from_cwd)
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
