from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import yaml
from ultralytics import YOLO  # type: ignore[attr-defined]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model.")
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--config", type=str, help="Path to experiment train.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pt)")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save metrics to experiment dir"
    )
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
    config_path: Path | None,
    experiment_dir: Path | None,
) -> str | None:
    if project:
        return project

    if config_path:
        project_name = extract_project_from_path(config_path, repo_root)
        if project_name:
            return project_name

    if experiment_dir:
        project_name = extract_project_from_path(experiment_dir, repo_root)
        if project_name:
            return project_name

    cwd = Path.cwd().resolve()
    project_name = extract_project_from_path(cwd, repo_root)
    if project_name:
        return project_name

    return select_project_name(repo_root)


def find_experiment_dir_from_cwd(repo_root: Path) -> Path | None:
    current = Path.cwd().resolve()
    while True:
        if current.parent.name == "experiments":
            return current
        if current == repo_root or current.parent == current:
            break
        current = current.parent
    return None


def list_experiment_dirs(experiments_dir: Path) -> list[Path]:
    if not experiments_dir.exists():
        return []
    experiment_dirs = [path for path in experiments_dir.iterdir() if path.is_dir()]
    return sorted(experiment_dirs, key=lambda path: path.name)


def select_experiment_dir(experiments_dir: Path) -> Path | None:
    experiment_dirs = list_experiment_dirs(experiments_dir)
    if not experiment_dirs:
        print(f"No experiments found in {experiments_dir}", file=sys.stderr)
        return None

    if len(experiment_dirs) == 1:
        return experiment_dirs[0]

    if not sys.stdin.isatty():
        print(
            "Multiple experiments found; pass --experiment or --config.",
            file=sys.stderr,
        )
        return None

    print("Available experiments:")
    for idx, experiment_dir in enumerate(experiment_dirs, start=1):
        print(f"  [{idx}] {experiment_dir.name}")

    default_index = len(experiment_dirs)
    while True:
        raw = input(
            f"Select experiment [{experiment_dirs[default_index - 1].name}]: "
        ).strip()
        if not raw:
            return experiment_dirs[default_index - 1]
        try:
            choice = int(raw)
        except ValueError:
            print("Enter an experiment number.", file=sys.stderr)
            continue
        if 1 <= choice <= len(experiment_dirs):
            return experiment_dirs[choice - 1]
        print("Selection out of range.", file=sys.stderr)


def resolve_experiment_dir(
    project_root: Path,
    experiment: str | None,
    config_path: Path | None,
    cwd_experiment_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    if config_path:
        return config_path.parent, config_path

    if experiment:
        experiment_dir = project_root / "experiments" / experiment
        return experiment_dir, experiment_dir / "train.yaml"

    if cwd_experiment_dir:
        return cwd_experiment_dir, cwd_experiment_dir / "train.yaml"

    return None, None


def load_data_yaml_from_train(train_yaml: Path) -> Path | None:
    if not train_yaml.exists():
        return None

    try:
        config = yaml.safe_load(train_yaml.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None

    if isinstance(config, dict) and isinstance(config.get("data"), str):
        data_path = Path(config["data"])
        if not data_path.is_absolute():
            data_path = (train_yaml.parent / data_path).resolve()
        if data_path.is_dir():
            data_path = data_path / "data.yaml"
        return data_path

    return None


def resolve_data_yaml(
    experiment_dir: Path | None,
    data_arg: str | None,
    train_yaml: Path | None,
) -> Path | None:
    if data_arg:
        candidate = Path(data_arg)
        if candidate.is_dir():
            candidate = candidate / "data.yaml"
        return candidate

    if train_yaml:
        data_path = load_data_yaml_from_train(train_yaml)
        if data_path:
            return data_path

    if experiment_dir:
        candidate = experiment_dir / "data.yaml"
        if candidate.exists():
            return candidate

    return None


def resolve_checkpoint_path(
    experiment_dir: Path | None, checkpoint_arg: str | None
) -> Path | None:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    if not experiment_dir:
        return None

    preferred = experiment_dir / "runs" / "detect" / "weights" / "best.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(
        experiment_dir.rglob("best.pt"), key=lambda path: path.stat().st_mtime
    )
    if candidates:
        return candidates[-1]

    candidates = sorted(
        experiment_dir.rglob("last.pt"), key=lambda path: path.stat().st_mtime
    )
    if candidates:
        return candidates[-1]

    return None


def prompt_for_path(label: str) -> Path | None:
    if not sys.stdin.isatty():
        return None

    while True:
        raw = input(f"{label}: ").strip()
        if raw:
            return Path(raw)
        print(f"{label} is required.", file=sys.stderr)


def evaluate(
    checkpoint_path: Path, data_yaml: Path, split: str, experiment_dir: Path
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    print(f"Loading model: {checkpoint_path}")
    print(f"Dataset config: {data_yaml}")
    print(f"Evaluating on: {split} split\n")

    model = YOLO(str(checkpoint_path))

    print("Running evaluation...")
    print("=" * 60)
    results = model.val(
        data=str(data_yaml),
        split=split,
        project=str(experiment_dir / "eval"),
        name=split,
    )
    print("=" * 60)

    results_dict = getattr(results, "results_dict", {}) or {}
    metrics = dict(results_dict)
    if not metrics and hasattr(results, "maps"):
        metrics = {
            "mAP50": results.box.map50 if hasattr(results.box, "map50") else None,
            "mAP50-95": results.box.map if hasattr(results.box, "map") else None,
            "precision": results.box.mp if hasattr(results.box, "mp") else None,
            "recall": results.box.mr if hasattr(results.box, "mr") else None,
        }

    return metrics


def save_metrics(metrics: dict, experiment_dir: Path, split: str) -> Path:
    output_file = experiment_dir / f"metrics_{split}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_file


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve() if args.config else None
    cwd_experiment_dir = find_experiment_dir_from_cwd(repo_root)

    project_name = resolve_project_name(
        args.project, repo_root, config_path, cwd_experiment_dir
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

    experiment_dir, train_yaml = resolve_experiment_dir(
        project_root, args.experiment, config_path, cwd_experiment_dir
    )
    if experiment_dir is None or train_yaml is None:
        experiment_dir = select_experiment_dir(project_root / "experiments")
        if experiment_dir is None:
            return 2
        train_yaml = experiment_dir / "train.yaml"

    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}", file=sys.stderr)
        return 1

    data_yaml = resolve_data_yaml(experiment_dir, args.data, train_yaml)
    if data_yaml is None:
        data_yaml = prompt_for_path("data.yaml path")
    if data_yaml is None:
        print("data.yaml not specified.", file=sys.stderr)
        return 2

    checkpoint_path = resolve_checkpoint_path(experiment_dir, args.checkpoint)
    if checkpoint_path is None:
        checkpoint_path = prompt_for_path("Checkpoint path")
    if checkpoint_path is None:
        print("Checkpoint not specified.", file=sys.stderr)
        return 2

    try:
        metrics = evaluate(checkpoint_path, data_yaml, args.split, experiment_dir)
    except Exception as exc:
        print(f"\n✗ Error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)

    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float)):
            print(f"  {key:20s}: {value:.4f}")
        elif value is not None:
            print(f"  {key:20s}: {value}")

    if args.save:
        output_file = save_metrics(metrics, experiment_dir, args.split)
        print(f"\n✓ Metrics saved to: {output_file}")

    print("\n✓ Evaluation complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
