# CEUS Vision-POI Fusion Pipeline

Generated: 2026-07-21 20:22:48

This repository organizes the full experimental workflow for building-level residential health inspection using visual detection, POI context, and multi-source fusion evaluation.

## Repository Scope

Included: source code, lightweight configuration references, and anonymized tabular input data required by the public workflow.

Excluded: raw images, training datasets, annotated images, raw POI JSON files, trained weights, processed visual evidence tables, correlation-analysis outputs, logs, Word/PDF/PPT manuscript files, and figure images.

Privacy note: all location-linked tables in this public package are anonymized. Real community names, street names, building names, addresses, and coordinates have been redacted or replaced with placeholder tokens. The public release is intended for code review and workflow demonstration, not for recovering the original geographic dataset.

## Folder Structure

```text
scripts/
  01_data_preparation/              # dataset split, class counting, balanced dataset preparation
  02_visual_training/               # YOLO/RT-DETR/Vision training and evaluation scripts
  03_prediction/                    # prediction and visual evidence aggregation utilities
  04_poi_collection_and_correlation/ # POI crawling/cleaning and Pearson/Spearman correlation analysis
  05_fusion_evaluation/             # ground-truth-based ablation, GroupCV, feature exploration, summaries
configs/
  visual_training_args/             # lightweight args.yaml references from visual training runs
data/
  ground_truth/                     # ground-truth table used by the tabular experiments
  metadata/                         # class/category mappings and class lists
  tabular_inputs/                   # anonymized community/building coordinate tables
  poi_features/                     # anonymized POI feature tables by radius
```

## Main Pipeline

1. Data preparation: `scripts/01_data_preparation/`
2. Visual detector training: `scripts/02_visual_training/`
3. Prediction and evidence aggregation: `scripts/03_prediction/`
4. POI collection and correlation analysis: `scripts/04_poi_collection_and_correlation/`
5. Fusion evaluation: `scripts/05_fusion_evaluation/evaluate_ablation.py`
6. Strict community GroupCV: `scripts/05_fusion_evaluation/evaluate_ablation_groupcv.py`
7. Administrative-scale spatial validation: `scripts/05_fusion_evaluation/evaluate_spatial_cv_group.py`

## Example Commands

```bash
python scripts/05_fusion_evaluation/evaluate_ablation.py --model-name yolov8_balanced --poi-radius 1000 --fusion-policy guarded --cv-mode stratified
python scripts/05_fusion_evaluation/evaluate_ablation_groupcv.py --model-name yolov8_balanced --poi-radius 1000 --fusion-policy guarded
python scripts/05_fusion_evaluation/evaluate_spatial_cv_group.py --model-name yolov8_balanced --poi-radius 1000 --group-col 街道
```

## Data Note

The raw image dataset, model weights, raw POI JSON files, and processed visual evidence tables are not included. The tabular inputs shipped here are anonymized placeholder versions that preserve the public code paths without exposing original location information.

See `MANIFEST.csv` for file provenance and SHA-256 hashes.

