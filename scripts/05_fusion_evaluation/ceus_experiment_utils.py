from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluate_ablation import load_and_merge
from experiment_registry import CLASS_NAMES, NEW_EXP_DIR

CEUS_EXTRA_DIR = NEW_EXP_DIR / "ceus_extra"


def ensure_ceus_extra_dir() -> Path:
    CEUS_EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    return CEUS_EXTRA_DIR


def load_feature_table(model_name: str, poi_radius: int):
    cfg, poi_json, df, df_poi = load_and_merge(model_name, poi_radius)
    poi_features = [c for c in df_poi.columns if c != "社区"]
    return cfg, poi_json, df, poi_features


def class_feature_columns(df, poi_features: list[str], class_name: str) -> tuple[list[str], list[str], list[str]]:
    vision_features = [
        f"{class_name}_box_count",
        f"{class_name}_max_conf",
        f"{class_name}_mean_conf",
        f"{class_name}_infection_rate",
        f"{class_name}_median_conf",
        f"{class_name}_std_conf",
    ]
    valid_vision = [c for c in vision_features if c in df.columns]
    valid_poi = [c for c in poi_features if c in df.columns]
    return valid_vision, valid_poi, valid_vision + valid_poi


def class_xy(df, poi_features: list[str], class_name: str):
    _, _, feature_cols = class_feature_columns(df, poi_features, class_name)
    x = df[feature_cols].values
    y = df[f"{class_name}_gt"].values
    return x, y, feature_cols


def valid_classes(df) -> list[str]:
    classes = []
    for class_name in CLASS_NAMES:
        y = df[f"{class_name}_gt"].values
        if len(np.unique(y)) >= 2:
            classes.append(class_name)
    return classes

