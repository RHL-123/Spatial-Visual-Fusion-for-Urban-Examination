# Spatial-Visual Fusion Framework (SV-Fusion)

### Official Code for: *Does the 15-Minute Community Life Circle Inform Residential Building Health? Spatial-Visual Fusion Framework for Urban Physical Examination* (iScience / Cell Press)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Visual%20Detectors-YOLO%2FRT--DETR-green)](#visual-front-end)
[![Privacy](https://img.shields.io/badge/Data-Anonymized-lightgrey)](#data-and-privacy)

This repository contains the public code package for a building-level residential inspection workflow. The pipeline combines visual defect evidence from object detection models with point-of-interest (POI) context and evaluates multiple fusion strategies under ordinary cross-validation and strict spatial group cross-validation.

This project provides the official research codebase and evaluation pipeline for our study published in *iScience* (Cell Press). The framework investigates whether 15-minute community life circle functional contexts provide informative priors for residential building health, establishing an end-to-end spatial-visual fusion architecture (MSCF) combined with multi-perspective aerial-street visual aggregation (MVVA).

## News

- 2026-07-21: Public code package released with anonymized tabular inputs.
- Dataset release: the full raw image dataset is not included in this repository due to privacy and size constraints. A separate dataset release is being prepared and will be announced independently after the release is finalized.

## Highlights

- **Visual front end.** Training and evaluation scripts are provided for YOLOv8, YOLOv9, YOLOv10, YOLO11, and RT-DETR style detectors.
- **POI-context modeling.** The repository includes scripts for POI collection, category cleaning, Pearson/Spearman correlation analysis, and radius-based POI feature construction.
- **Multi-source fusion.** The evaluation code supports detection-driven baseline, multi-view visual aggregation, POI-only context modeling, and multi-source contextual fusion.
- **Spatial reliability.** The code includes strict community GroupCV and administrative-scale spatial validation utilities to reduce overly optimistic spatial leakage.
- **Privacy-aware release.** Public tabular files are anonymized; original addresses, community names, building names, and coordinates are not exposed.

## Framework Overview

![Method framework](assets/figures/method_framework.png)

The framework contains four connected stages: visual perception and building-level feature aggregation, POI-based spatial association and prior identification, multi-source fusion and classification inference, and privacy-safe result display. The public repository keeps this workflow reproducible at the code level while excluding identifiable raw images and location-specific source data.

## Main Results

The public repository includes privacy-safe aggregate figures and source tables. These results are intended to show the experimental logic and headline trends without exposing raw images, building-level predictions, or identifiable locations.

### Ablation Under Strict Community GroupCV

![Ablation macro-F1 under strict community GroupCV](assets/figures/ablation_macro_f1.png)

Under strict community GroupCV, the multi-source contextual fusion setting (MSCF) achieves the highest aggregate Macro-F1 among the released ablation groups. POI-only context (UFCO) does not replace visual evidence, but POI context can provide auxiliary gains when fused with building-level visual evidence.

### Class-wise Fusion Gains

![Class-wise F1 gain heatmap](assets/figures/fusion_gain_heatmap.png)

The class-wise gain heatmap shows that POI contributions are category-dependent. This is the expected behavior for urban context features: they are contextual priors rather than direct visual evidence.

### Probability Ranking Curves

![Macro ROC and PR curves](assets/figures/macro_roc_pr_curves.png)

The ROC/PR curves summarize probability-ranking behavior across model families. PR curves are especially important for sparse building condition categories.

### POI Correlation Diagnostics

![Pearson POI correlation heatmap](assets/figures/poi_pearson_heatmap.png)

![Spearman POI correlation heatmap](assets/figures/poi_spearman_heatmap.png)

These correlation diagnostics motivate the use of POI context as an auxiliary spatial signal. They should not be interpreted as causal evidence.

Released aggregate result tables:

- `assets/results/ablation_macro_f1_summary.csv`

## Method Overview

The workflow is organized as five stages:

1. **Data preparation:** split and balance inspection images for visual detector training.
2. **Visual detection:** train detector variants and aggregate image-level evidence into building-level visual features.
3. **POI representation:** collect and clean POI data, then derive radius-based urban context features.
4. **Fusion evaluation:** compare visual-only, POI-only, and multi-source fusion models.
5. **Spatial validation:** test robustness under ordinary CV, community GroupCV, and administrative-scale group splits.

In the manuscript terminology, the four major model families correspond to:

- **DD:** detection-driven baseline.
- **MVVA:** multi-view visual aggregation.
- **UFCO:** urban functional context only.
- **MSCF:** multi-source contextual fusion.

## Repository Structure

```text
configs/
  visual_training_args/             # lightweight visual-training configuration references

assets/
  figures/                          # privacy-safe aggregate result figures
  results/                          # aggregate result source tables

data/
  ground_truth/                     # anonymized ground-truth table for public code paths
  metadata/                         # class/category mappings and class lists
  poi_features/                     # anonymized radius-based POI feature tables
  tabular_inputs/                   # anonymized coordinate and tabular support files

scripts/
  00_utilities/                     # dataset counting and table utilities
  01_data_preparation/              # dataset preparation and balancing
  02_visual_training/               # visual detector training/evaluation scripts
  03_prediction/                    # prediction and visual evidence aggregation utilities
  04_poi_collection_and_correlation/# POI crawling, cleaning, and correlation analysis
  05_fusion_evaluation/             # ablation, GroupCV, feature exploration, and summaries
