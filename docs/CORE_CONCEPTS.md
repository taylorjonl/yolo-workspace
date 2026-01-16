# Core Concepts

This document captures the core concepts defined in `README.md` for the vision
workspace structure and workflow.

## Project

A project represents a single vision problem with a defined label schema. It
maps 1:1 with a labeling project (e.g., CVAT) and owns its datasets,
experiments, and models.

## Artifacts

Artifacts are immutable external inputs used to build datasets, typically ZIP
exports from labeling tools. They are never modified after being added and are
archived for traceability.

## Dataset Version

A dataset version is a materialized, training-ready snapshot derived from one
or more artifacts. It is stored under a versioned directory (e.g., `dataset_vN`)
and remains immutable once created.

## Manifest

Each dataset version includes a manifest that documents which artifacts were
used, the label schema, dataset construction assumptions, and other context.
The manifest is descriptive, not executable.

## Experiment

An experiment is a reproducible training run tied to a specific dataset
version and configuration. It produces logs, metrics, and checkpoints without
modifying datasets or artifacts.

## Models

Models progress through distinct stages:

- Base models: pretrained weights used as starting points
- Checkpoints: intermediate or final training outputs
- Exports: converted models for deployment (e.g., ONNX)
