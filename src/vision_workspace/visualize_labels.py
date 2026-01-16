from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

DATASET_DIR_PATTERN = re.compile(r"^dataset_v(\d+)$")

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 165, 0),
    (0, 128, 255),
    (128, 255, 0),
    (255, 128, 0),
    (0, 255, 128),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images.")
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument("--dataset", type=str, help="Dataset version or path")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Which split to visualize",
    )
    parser.add_argument("--image", type=str, help="Single image to visualize")
    parser.add_argument("--label", type=str, help="Corresponding label file")
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of samples to visualize (dataset mode)",
    )
    parser.add_argument(
        "--output", type=str, help="Output directory to save visualizations"
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress details")
    return parser.parse_args(argv)


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    annotations: list[tuple[int, float, float, float, float]] = []

    if not label_path.exists():
        return annotations

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Error reading {label_path}: {exc}", file=sys.stderr)
        return annotations

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def yolo_to_xyxy(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = bbox

    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    width_px = width * image_width
    height_px = height * image_height

    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)

    return (x1, y1, x2, y2)


def draw_yolo_boxes(
    image_path: Path,
    label_path: Path,
    class_names: list[str] | None = None,
) -> Any:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = image.shape[:2]
    annotations = parse_yolo_label(label_path)

    if not annotations:
        return image

    for class_id, x_center, y_center, bbox_width, bbox_height in annotations:
        x1, y1, x2, y2 = yolo_to_xyxy(
            (x_center, y_center, bbox_width, bbox_height), width, height
        )
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y1 = max(y1 - text_height - baseline - 5, 0)
        label_y2 = label_y1 + text_height + baseline + 5

        cv2.rectangle(image, (x1, label_y1), (x1 + text_width, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def visualize_dataset_samples(
    dataset_dir: Path,
    split: str,
    num_samples: int,
    class_names: list[str] | None,
    output_dir: Path | None,
    verbose: bool,
) -> None:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
        return
    if not labels_dir.exists():
        print(f"Labels directory does not exist: {labels_dir}")
        return

    image_files = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    num_samples = min(num_samples, len(image_files))
    sampled_images = random.sample(image_files, num_samples)

    if verbose:
        print(f"Visualizing {num_samples} samples from {split} split...\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    show_images = output_dir is None

    for index, image_path in enumerate(sampled_images, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if verbose:
            print(f"[{index}/{num_samples}] {image_path.name}")

        if not label_path.exists():
            if verbose:
                print("  Warning: No label file found")
            annotated = cv2.imread(str(image_path))
        else:
            annotations = parse_yolo_label(label_path)
            if verbose:
                print(f"  Objects: {len(annotations)}")
            annotated = draw_yolo_boxes(image_path, label_path, class_names)

        if annotated is None:
            if verbose:
                print("  Error: Failed to load image")
                print()
            continue

        annotated_image: Any = annotated

        if output_dir:
            output_path = output_dir / f"{image_path.stem}_vis{image_path.suffix}"
            cv2.imwrite(str(output_path), annotated_image)
            if verbose:
                print(f"  Saved: {output_path}")

        if show_images:
            cv2.imshow(f"Visualization - {image_path.name}", annotated_image)
            key = cv2.waitKey(0)
            if key == 27:
                if verbose:
                    print("\nStopped by user")
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()

        if verbose:
            print()

    if output_dir and verbose:
        print(f"✓ Visualizations saved to {output_dir}")


def load_class_names(dataset_dir: Path, verbose: bool) -> list[str] | None:
    data_yaml_path = dataset_dir / "data.yaml"
    if data_yaml_path.exists():
        try:
            data_config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            if verbose:
                print(f"Warning: Failed to load class names: {exc}", file=sys.stderr)
        else:
            names = data_config.get("names") if isinstance(data_config, dict) else None
            if isinstance(names, list):
                return [str(name) for name in names]

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if verbose:
            print(f"Warning: Failed to load class names: {exc}", file=sys.stderr)
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
    project: str | None, repo_root: Path, dataset_dir: Path | None
) -> str | None:
    if project:
        return project

    if dataset_dir:
        try:
            relative = dataset_dir.resolve().relative_to(repo_root)
        except ValueError:
            relative = None
        if relative and "projects" in relative.parts:
            index = relative.parts.index("projects")
            if index + 1 < len(relative.parts):
                return relative.parts[index + 1]

    cwd = Path.cwd().resolve()
    try:
        relative = cwd.relative_to(repo_root)
    except ValueError:
        relative = None

    if relative:
        parts = relative.parts
        if "projects" in parts:
            index = parts.index("projects")
            if index + 1 < len(parts):
                return parts[index + 1]

    return select_project_name(repo_root)


def find_data_yaml(start_dir: Path, stop_dir: Path) -> Path | None:
    current = start_dir
    while True:
        candidate = current / "data.yaml"
        if candidate.exists():
            return candidate
        if current == stop_dir or current.parent == current:
            break
        current = current.parent
    return None


def find_manifest_json(start_dir: Path, stop_dir: Path) -> Path | None:
    current = start_dir
    while True:
        candidate = current / "manifest.json"
        if candidate.exists():
            return candidate
        if current == stop_dir or current.parent == current:
            break
        current = current.parent
    return None


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

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            return 1

        label_path = Path(args.label) if args.label else image_path.with_suffix(".txt")

        class_names: list[str] | None = None
        dataset_dir = None
        if args.dataset:
            dataset_dir = resolve_dataset_dir(project_root, args.dataset)
        if dataset_dir is None:
            dataset_dir = dataset_dir_from_cwd
        if dataset_dir:
            class_names = load_class_names(dataset_dir, args.verbose)
        if class_names is None:
            data_yaml = find_data_yaml(image_path.parent, repo_root)
            if data_yaml:
                class_names = load_class_names(data_yaml.parent, args.verbose)
        if class_names is None:
            manifest_json = find_manifest_json(image_path.parent, repo_root)
            if manifest_json:
                class_names = load_class_names(manifest_json.parent, args.verbose)

        if class_names and args.verbose:
            print(f"Loaded {len(class_names)} class names from data.yaml\n")

        if not label_path.exists():
            if args.verbose:
                print(f"Warning: Label not found: {label_path}")
                print("Displaying image without annotations...")
            annotated = cv2.imread(str(image_path))
        else:
            annotations = parse_yolo_label(label_path)
            if args.verbose:
                print(f"Image: {image_path}")
                print(f"Label: {label_path}")
                print(f"Objects: {len(annotations)}\n")
            annotated = draw_yolo_boxes(image_path, label_path, class_names)

        if annotated is None:
            print("Error: Failed to load image", file=sys.stderr)
            return 1

        annotated_image: Any = annotated
        show_images = args.output is None

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_vis{image_path.suffix}"
            cv2.imwrite(str(output_path), annotated_image)
            if args.verbose:
                print(f"Saved: {output_path}")

        if show_images:
            cv2.imshow("Visualization", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0

    if args.dataset is None and dataset_dir_from_cwd is not None:
        dataset_dir = dataset_dir_from_cwd
    else:
        dataset_dir = resolve_dataset_dir(project_root, args.dataset)
    if dataset_dir is None:
        return 1

    class_names = load_class_names(dataset_dir, args.verbose)
    if class_names and args.verbose:
        print(f"Loaded {len(class_names)} class names from data.yaml\n")

    output_dir = Path(args.output) if args.output else None
    visualize_dataset_samples(
        dataset_dir,
        args.split,
        args.num,
        class_names,
        output_dir,
        args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
