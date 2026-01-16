from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO  # type: ignore[attr-defined]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument(
        "--experiment", type=str, help="Experiment name (exp_001_baseline)"
    )
    parser.add_argument("--config", type=str, help="Path to training config YAML")
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
    return extract_project_from_path(cwd, repo_root)


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


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if isinstance(config, dict) and isinstance(config.get("data"), str):
        data_path = Path(config["data"])
        if not data_path.is_absolute():
            config["data"] = str((config_path.parent / data_path).resolve())

    if isinstance(config, dict) and isinstance(config.get("project"), str):
        project_path = Path(config["project"])
        if not project_path.is_absolute():
            config["project"] = str((config_path.parent / project_path).resolve())

    return config or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_model(config: dict, experiment_name: str, experiment_dir: Path) -> dict:
    print(f"\n{'=' * 60}")
    print(f"Training Experiment: {experiment_name}")
    print(f"{'=' * 60}\n")

    if "seed" in config:
        print(f"Setting random seed: {config['seed']}")
        set_seed(int(config["seed"]))

    config.setdefault("project", str(experiment_dir / "runs"))
    config.setdefault("name", "detect")

    model_name = config.get("model", "yolov8n.pt")
    print(f"Initializing model: {model_name}")
    model = YOLO(model_name)

    print("\nStarting training...")
    print(f"  Dataset: {config.get('data')}")
    print(f"  Epochs: {config.get('epochs')}")
    print(f"  Batch size: {config.get('batch')}")
    print(f"  Image size: {config.get('imgsz')}\n")

    results = model.train(**config)

    results_dict = getattr(results, "results_dict", {}) or {}
    metrics = {
        "experiment": experiment_name,
        "model": model_name,
        "epochs": config.get("epochs"),
        "final_metrics": {
            "map50": float(results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(results_dict.get("metrics/mAP50-95(B)", 0)),
        },
    }

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    save_dir = getattr(results, "save_dir", None)
    print(f"\nCheckpoints saved to: {save_dir if save_dir else 'runs/'}")

    print("\nREMINDER:")
    print("  - Checkpoints (.pt) are INTERMEDIATE artifacts")
    print("  - Export to ONNX: export-onnx --experiment", experiment_name)
    print("  - Validate ONNX: validate-onnx --experiment", experiment_name)

    return metrics


def save_metrics(metrics: dict, experiment_dir: Path) -> None:
    metrics_path = experiment_dir / "metrics_train.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            existing.update(metrics)
            metrics = existing

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")


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

    experiment_dir, config_path = resolve_experiment_dir(
        project_root, args.experiment, config_path, cwd_experiment_dir
    )
    if experiment_dir is None or config_path is None:
        experiment_dir = select_experiment_dir(project_root / "experiments")
        if experiment_dir is None:
            return 2
        config_path = experiment_dir / "train.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}", file=sys.stderr)
        return 1

    experiment_name = experiment_dir.name

    try:
        config = load_config(config_path)
    except (OSError, yaml.YAMLError) as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1

    try:
        metrics = train_model(config, experiment_name, experiment_dir)
        save_metrics(metrics, experiment_dir)
        return 0
    except Exception as exc:  # pragma: no cover - surface errors in CLI
        print(f"\nERROR: Training failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
