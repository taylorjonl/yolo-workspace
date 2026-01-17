from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO  # type: ignore[attr-defined]

from vision_workspace.cli_helpers import (
    find_experiment_dir_from_cwd,
    prompt_for_path,
    resolve_checkpoint_path,
    resolve_experiment_dir,
    resolve_project_name,
    select_experiment_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX.")
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--config", type=str, help="Path to experiment train.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (.pt)")
    parser.add_argument("--output", type=str, help="Output directory for ONNX file")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes")
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify the ONNX graph"
    )
    return parser.parse_args(argv)


def resolve_export_path(checkpoint_path: Path, export_result: object) -> Path | None:
    if isinstance(export_result, (str, Path)):
        candidate = Path(export_result)
        if candidate.exists():
            return candidate

    candidates = sorted(
        checkpoint_path.parent.rglob("*.onnx"), key=lambda path: path.stat().st_mtime
    )
    if candidates:
        return candidates[-1]

    return None


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

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else None
    if output_dir is None:
        output_dir = project_root / "models" / "exports" / experiment_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(checkpoint_path))
    export_result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )

    export_path = resolve_export_path(checkpoint_path, export_result)
    if export_path is None:
        print("ONNX export failed: file not found.", file=sys.stderr)
        return 1

    destination = output_dir / f"{experiment_dir.name}.onnx"
    if export_path.resolve() != destination.resolve():
        shutil.copy2(export_path, destination)

    print(f"Exported ONNX to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
