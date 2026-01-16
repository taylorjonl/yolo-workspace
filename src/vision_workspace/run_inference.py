from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO  # type: ignore[attr-defined]

from vision_workspace.cli_helpers import (
    find_experiment_dir_from_cwd,
    prompt_for_path,
    resolve_experiment_dir,
    resolve_project_name,
    select_experiment_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images.")
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--config", type=str, help="Path to experiment train.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--source", type=str, help="Image or folder to run inference on"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--save-txt", action="store_true", help="Save YOLO-format labels"
    )
    parser.add_argument("--save-crop", action="store_true", help="Save detected crops")
    parser.add_argument("--name", type=str, help="Output run name")
    return parser.parse_args(argv)


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


def run_inference(
    checkpoint_path: Path,
    source: Path,
    experiment_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    save_txt: bool,
    save_crop: bool,
    run_name: str,
) -> None:
    model = YOLO(str(checkpoint_path))
    model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        project=str(experiment_dir / "predictions"),
        name=run_name,
        save=True,
        save_txt=save_txt,
        save_crop=save_crop,
        exist_ok=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve() if args.config else None
    cwd_experiment_dir = find_experiment_dir_from_cwd(repo_root)

    project_name = resolve_project_name(
        args.project, repo_root, config_path=config_path, context_dir=cwd_experiment_dir
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

    experiment_dir, _ = resolve_experiment_dir(
        project_root, args.experiment, config_path, cwd_experiment_dir
    )
    if experiment_dir is None:
        experiment_dir = select_experiment_dir(project_root / "experiments")
        if experiment_dir is None:
            return 2

    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}", file=sys.stderr)
        return 1

    checkpoint_path = resolve_checkpoint_path(experiment_dir, args.checkpoint)
    if checkpoint_path is None:
        checkpoint_path = prompt_for_path("Checkpoint path")
    if checkpoint_path is None:
        print("Checkpoint not specified.", file=sys.stderr)
        return 2

    source = Path(args.source) if args.source else prompt_for_path("Source path")
    if source is None:
        print("Source not specified.", file=sys.stderr)
        return 2

    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    run_name = args.name or datetime.now().strftime("predict_%Y%m%d_%H%M%S")

    run_inference(
        checkpoint_path,
        source,
        experiment_dir,
        args.conf,
        args.iou,
        args.imgsz,
        args.save_txt,
        args.save_crop,
        run_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
