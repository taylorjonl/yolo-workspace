from __future__ import annotations

import re
import sys
from pathlib import Path


DATASET_DIR_PATTERN = re.compile(r"^dataset_v(\d+)$")


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
    config_path: Path | None = None,
    context_dir: Path | None = None,
) -> str | None:
    if project:
        return project

    if config_path:
        project_name = extract_project_from_path(config_path, repo_root)
        if project_name:
            return project_name

    if context_dir:
        project_name = extract_project_from_path(context_dir, repo_root)
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
        images_dir = current / "images"
        labels_dir = current / "labels"
        if images_dir.exists() and labels_dir.exists():
            return current
        if current == repo_root or current.parent == current:
            break
        current = current.parent
    return None


def resolve_checkpoint_path(
    experiment_dir: Path | None, checkpoint_arg: str | None
) -> Path | None:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    if not experiment_dir:
        return None

    preferred = experiment_dir / "weights" / "best.pt"
    if preferred.exists():
        return preferred

    if experiment_dir.parent.name == "experiments":
        project_root = experiment_dir.parent.parent
        preferred = (
            project_root
            / "models"
            / "checkpoints"
            / experiment_dir.name
            / "weights"
            / "best.pt"
        )
        if preferred.exists():
            return preferred

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
