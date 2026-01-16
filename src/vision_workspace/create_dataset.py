from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from vision_workspace.cli_helpers import DATASET_DIR_PATTERN, resolve_project_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset version from CVAT COCO exports in YOLO format."
    )
    parser.add_argument("project", nargs="?", help="Project name under projects/.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Train split ratio (0-1). If omitted, you'll be prompted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all incoming ZIP artifacts without prompting.",
    )
    return parser.parse_args(argv)


def prompt_train_ratio(default: float = 0.8) -> float:
    if not sys.stdin.isatty():
        return default

    while True:
        raw = input(f"Train split ratio (0-1) [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number between 0 and 1.", file=sys.stderr)
            continue
        if 0.0 < value < 1.0:
            return value
        print("Enter a number between 0 and 1.", file=sys.stderr)


def parse_selection(raw: str, max_index: int) -> list[int]:
    selection: list[int] = []
    raw = raw.strip().lower()
    if raw in {"", "all", "a", "*"}:
        return list(range(1, max_index + 1))

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            if not start_text or not end_text:
                raise ValueError("Invalid range selection.")
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError("Invalid range selection.")
            selection.extend(range(start, end + 1))
        else:
            selection.append(int(part))

    for index in selection:
        if index < 1 or index > max_index:
            raise ValueError("Selection is out of range.")

    return selection


def select_artifacts(artifacts: list[Path], use_all: bool) -> list[Path]:
    if not artifacts:
        raise ValueError("No incoming artifacts were found.")
    if use_all or len(artifacts) == 1 or not sys.stdin.isatty():
        return artifacts

    print("Available artifacts:")
    for idx, path in enumerate(artifacts, start=1):
        print(f"  [{idx}] {path.name}")

    while True:
        raw = input("Select artifacts (e.g., 1,3-5 or 'all') [all]: ")
        try:
            indices = parse_selection(raw, len(artifacts))
        except (ValueError, TypeError):
            print("Invalid selection.", file=sys.stderr)
            continue
        return [artifacts[i - 1] for i in indices]


def next_dataset_version(datasets_dir: Path) -> str:
    max_version = 0
    if datasets_dir.exists():
        for child in datasets_dir.iterdir():
            if not child.is_dir():
                continue
            match = DATASET_DIR_PATTERN.match(child.name)
            if match:
                max_version = max(max_version, int(match.group(1)))
    return f"dataset_v{max_version + 1}"


def find_annotation_path(names: Iterable[str]) -> str:
    names = list(names)
    preferred = "annotations/instances_default.json"
    if preferred in names:
        return preferred
    candidates = [
        name
        for name in names
        if name.startswith("annotations/") and name.endswith(".json")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("No COCO annotations JSON found in artifact.")
    raise ValueError("Multiple annotation JSON files found; expected one.")


def build_image_entry_map(names: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        if not name.startswith("images/") or name.endswith("/"):
            continue
        base = Path(name).name
        if base in mapping:
            raise ValueError(f"Duplicate image name in artifact: {base}")
        mapping[base] = name
    return mapping


def load_coco_from_zip(zip_path: Path) -> tuple[dict, dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zip_file:
        names = zip_file.namelist()
        annotation_path = find_annotation_path(names)
        image_entry_map = build_image_entry_map(names)
        data = json.loads(zip_file.read(annotation_path))
    return data, image_entry_map


def combine_coco(
    artifacts: list[Path],
) -> tuple[list[dict], list[dict], dict[int, tuple[Path, str]], dict]:
    combined_images: list[dict] = []
    combined_annotations: list[dict] = []
    image_sources: dict[int, tuple[Path, str]] = {}
    seen_names: set[str] = set()

    base_metadata: dict | None = None
    image_id_counter = 1
    annotation_id_counter = 1

    for zip_path in artifacts:
        coco, image_entry_map = load_coco_from_zip(zip_path)
        if base_metadata is None:
            base_metadata = {
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "categories": coco.get("categories", []),
            }
        else:
            if coco.get("categories", []) != base_metadata.get("categories", []):
                raise ValueError("Category lists do not match across artifacts.")

        image_id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            base_name = Path(image.get("file_name", "")).name
            if not base_name:
                raise ValueError("Image entry missing file_name.")
            if base_name in seen_names:
                raise ValueError(f"Duplicate image name across artifacts: {base_name}")
            if base_name not in image_entry_map:
                raise ValueError(f"Image not found in artifact: {base_name}")
            seen_names.add(base_name)
            new_image = dict(image)
            new_image["id"] = image_id_counter
            new_image["file_name"] = base_name
            combined_images.append(new_image)
            image_id_map[int(image["id"])] = image_id_counter
            image_sources[image_id_counter] = (zip_path, image_entry_map[base_name])
            image_id_counter += 1

        for annotation in coco.get("annotations", []):
            old_image_id = int(annotation.get("image_id"))
            if old_image_id not in image_id_map:
                raise ValueError("Annotation references unknown image id.")
            new_annotation = dict(annotation)
            new_annotation["id"] = annotation_id_counter
            new_annotation["image_id"] = image_id_map[old_image_id]
            combined_annotations.append(new_annotation)
            annotation_id_counter += 1

    if base_metadata is None:
        raise ValueError("No COCO data found in artifacts.")
    return combined_images, combined_annotations, image_sources, base_metadata


def split_ids(
    ids: list[int], train_ratio: float, seed: int
) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    rng.shuffle(ids)
    total = len(ids)
    train_count = int(total * train_ratio)
    if total > 1:
        train_count = max(1, min(total - 1, train_count))
    train_ids = set(ids[:train_count])
    val_ids = set(ids[train_count:])
    return train_ids, val_ids


def build_category_index(categories: list[dict]) -> dict[int, int]:
    if not categories:
        raise ValueError("No categories found in annotations.")
    sorted_categories = sorted(categories, key=lambda item: int(item.get("id", 0)))
    mapping: dict[int, int] = {}
    for idx, category in enumerate(sorted_categories):
        category_id = category.get("id")
        if category_id is None:
            raise ValueError("Category entry missing id.")
        if category_id in mapping:
            raise ValueError("Duplicate category id detected.")
        mapping[int(category_id)] = idx
    return mapping


def write_manifest(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    project_name = resolve_project_name(args.project, repo_root)
    if not project_name:
        print(
            "Project not specified. Run from projects/<name> or pass the project name.",
            file=sys.stderr,
        )
        return 2
    project_root = repo_root / "projects" / project_name

    incoming_dir = project_root / "artifacts" / "incoming"
    archive_dir = project_root / "artifacts" / "archive"
    datasets_dir = project_root / "datasets"

    if not project_root.exists():
        print(f"Project not found: {project_root}", file=sys.stderr)
        return 1

    artifacts = sorted(incoming_dir.glob("*.zip"))
    if not artifacts:
        print("No incoming artifacts were found.", file=sys.stderr)
        return 1

    train_ratio = args.train_ratio
    if train_ratio is None:
        train_ratio = prompt_train_ratio()
    if not (0.0 < train_ratio < 1.0):
        print("Train ratio must be between 0 and 1.", file=sys.stderr)
        return 2

    try:
        selected_artifacts = select_artifacts(artifacts, args.all)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dataset_version = next_dataset_version(datasets_dir)

    dataset_root = datasets_dir / dataset_version
    if dataset_root.exists():
        print(f"Dataset already exists: {dataset_root}", file=sys.stderr)
        return 1

    try:
        images, annotations, image_sources, metadata = combine_coco(selected_artifacts)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dataset_images_dir = dataset_root / "images"
    dataset_labels_dir = dataset_root / "labels"
    (dataset_images_dir / "train").mkdir(parents=True, exist_ok=False)
    (dataset_images_dir / "val").mkdir(parents=True, exist_ok=False)
    (dataset_labels_dir / "train").mkdir(parents=True, exist_ok=False)
    (dataset_labels_dir / "val").mkdir(parents=True, exist_ok=False)

    image_ids = [image["id"] for image in images]
    train_ids, val_ids = split_ids(image_ids, train_ratio, args.seed)

    images_by_id = {image["id"]: image for image in images}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)

    category_index = build_category_index(metadata.get("categories", []))

    try:
        grouped: dict[Path, list[tuple[int, str]]] = defaultdict(list)
        for image_id in image_ids:
            zip_path, zip_entry = image_sources[image_id]
            grouped[zip_path].append((image_id, zip_entry))

        for zip_path, entries in grouped.items():
            with zipfile.ZipFile(zip_path) as zip_file:
                for image_id, zip_entry in entries:
                    split = "train" if image_id in train_ids else "val"
                    base_name = Path(zip_entry).name
                    destination = dataset_images_dir / split / base_name
                    with (
                        zip_file.open(zip_entry) as source,
                        destination.open("wb") as dest,
                    ):
                        dest.write(source.read())
                    image = images_by_id[image_id]
                    width = image.get("width")
                    height = image.get("height")
                    if not width or not height:
                        raise ValueError("Image entry missing width/height.")
                    label_path = (
                        dataset_labels_dir / split / f"{Path(base_name).stem}.txt"
                    )
                    lines: list[str] = []
                    for annotation in annotations_by_image.get(image_id, []):
                        bbox = annotation.get("bbox")
                        if not bbox or len(bbox) != 4:
                            raise ValueError("Annotation missing bbox.")
                        x, y, w, h = bbox
                        class_id = annotation.get("category_id")
                        if class_id is None:
                            raise ValueError("Annotation missing category id.")
                        class_id = int(class_id)
                        if class_id not in category_index:
                            raise ValueError(
                                "Annotation references unknown category id."
                            )
                        x_center = (x + w / 2.0) / width
                        y_center = (y + h / 2.0) / height
                        w_norm = w / width
                        h_norm = h / height
                        lines.append(
                            f"{category_index[class_id]} "
                            f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                        )
                    label_path.write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"Failed to extract images: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    archive_target = archive_dir / dataset_version
    archive_target.mkdir(parents=True, exist_ok=True)
    for artifact in selected_artifacts:
        artifact.rename(archive_target / artifact.name)

    categories_sorted = sorted(
        metadata.get("categories", []), key=lambda item: int(item.get("id", 0))
    )
    manifest = {
        "dataset_version": dataset_version,
        "project": args.project,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [artifact.name for artifact in selected_artifacts],
        "split": {
            "train_ratio": train_ratio,
            "seed": args.seed,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
        },
        "images_total": len(images),
        "annotations_total": len(annotations),
        "labels_format": "yolo",
        "labels": {
            "train": "labels/train/*.txt",
            "val": "labels/val/*.txt",
        },
        "categories": [
            {
                "id": int(category.get("id"))
                if category.get("id") is not None
                else None,
                "name": category.get("name"),
                "index": idx,
            }
            for idx, category in enumerate(categories_sorted)
        ],
    }
    write_manifest(dataset_root / "manifest.json", manifest)

    print(f"Created dataset {dataset_version} at {dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
