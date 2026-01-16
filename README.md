# Vision Workspace

## Purpose

This repository defines a structured workspace for developing, training, and managing computer vision models (e.g., YOLO-based detectors) using annotated data produced in external tools such as CVAT.

The primary goals of this workspace are:

- Provide a clear, repeatable structure for managing multiple vision projects within a single repository
- Enforce immutability of datasets and training artifacts once they are created
- Support reproducible experiments by clearly separating data, configuration, and results
- Serve as a stable foundation for automation by humans and agents

This repository is opinionated about **structure and workflow**, but intentionally light on implementation details.

---

## Core Concepts

See `docs/CORE_CONCEPTS.md` for definitions of the workspace's core concepts.

---

## Repository Structure

At a high level:

```
workspace/
├── scripts/                # Executable scripts (dataset prep, training, evaluation)
├── tools/                  # Shared utilities and helper modules
│
├── projects/
│   ├── <project_name>/
│   │   ├── artifacts/
│   │   │   ├── incoming/   # Newly added external artifacts
│   │   │   └── archive/    # Archived artifacts organized by dataset version
│   │   │
│   │   ├── datasets/
│   │   │   └── dataset_vN/ # Immutable, training-ready datasets
│   │   │
│   │   ├── experiments/
│   │   │   └── exp_XXX/    # Reproducible training runs
│   │   │
│   │   └── models/
│   │       ├── checkpoints/
│   │       └── exports/
│   │
│   └── ...
│
└── models/
    └── base/               # Shared pretrained model weights
```

### Key structural principles

- Shared code lives at the repository root
- Projects are isolated from each other
- Datasets and artifacts are versioned and immutable
- Training outputs never overwrite inputs

---

## Typical Workflows

## Environment Setup

Create and activate a local virtual environment, then install the workspace
in editable mode to register the repo's script commands in this venv.

Using `uv`:

```
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
check-environment
```

Using Python:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
check-environment
```

CPU-only (no CUDA):

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
check-environment --skip-gpu
```

or

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
check-environment --skip-gpu
```

### Creating a New Project

A new project represents a new vision task with its own label schema and datasets.

The recommended workflow for creating a new project is:

1. Create the project skeleton using the initialization command:

```
init-project <project_name>
```

2. Verify the generated structure under `projects/<project_name>/`.

At this point, the project is ready to receive artifacts and produce its first
dataset version.

### Dataset Creation Workflow (High-Level)

1. Images are annotated in an external tool (e.g., CVAT)
2. Annotation exports are produced as ZIP files
3. ZIP files are placed into a project's `artifacts/incoming/` directory
4. Run `create-dataset <project_name>` and follow the prompts for artifacts and split ratio
5. A new `dataset_vN` directory is created
6. Consumed artifacts are archived under `artifacts/archive/dataset_vN/`
7. A manifest is written alongside the dataset

Result: a frozen, training-ready dataset version.

---

### Training / Experiment Workflow (High-Level)

1. Select a project and dataset version
2. Create a new experiment directory
3. Define experiment configuration
4. Run training via a script
5. Capture logs, metrics, and checkpoints
6. Optionally export trained models

Result: a reproducible experiment with traceable inputs and outputs.

---

### Iteration Workflow

- New annotations → new artifacts → new dataset version
- New ideas or configurations → new experiments
- No in-place mutation of existing datasets or experiments

---

This README intentionally defines **what things are and how they relate**, not how scripts are implemented.
