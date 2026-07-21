"""
Build a GIS-ready Vision_Only complete classification table.

`predict_enhanced_features_*.xlsx` already contains building-level visual
0/1 columns for each defect class. This script copies those visual columns
onto the address/coordinate/region layout of 拆分结果-gis画图/完整分类表.xlsx.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "拆分结果-gis画图" / "完整分类表.xlsx"
DEFAULT_PREDICTION = PROJECT_ROOT / "new_experiment" / "predictions" / "predict_enhanced_features_yolov8_balanced.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "new_experiment" / "post_correction_tables" / "Visual_Baseline完整分类表_yolov8_balanced.xlsx"

VISION_LABEL_COLUMNS = [
    "illegal_renovation_expansion",
    "addition_balconies_windows",
    "public_spaces_illegal_occupation",
    "wall_damage",
    "elevator_addition",
    "pipeline_damage",
    "no_aging_place_modifications",
]


def build_complete_classification_table(
    template_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    label_columns: list[str],
) -> pd.DataFrame:
    """Return template rows with visual prediction label columns replaced."""
    out = template_df.copy()
    pred = prediction_df[["楼栋ID", *label_columns]].copy()

    out["_楼栋ID_key"] = out["楼栋ID"].astype(str)
    pred["_楼栋ID_key"] = pred["楼栋ID"].astype(str)
    pred = pred.drop_duplicates("_楼栋ID_key", keep="first").set_index("_楼栋ID_key")

    missing_ids = sorted(set(out["_楼栋ID_key"]) - set(pred.index))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"预测表缺少 {len(missing_ids)} 个楼栋ID，例如: {preview}")

    for col in label_columns:
        if col not in out.columns:
            raise ValueError(f"完整分类表缺少标签列: {col}")
        if col not in pred.columns:
            raise ValueError(f"预测表缺少视觉标签列: {col}")
        out[col] = out["_楼栋ID_key"].map(pred[col]).astype(int)

    return out.drop(columns=["_楼栋ID_key"])


def export_vision_only_table(
    prediction_path: Path = DEFAULT_PREDICTION,
    template_path: Path = DEFAULT_TEMPLATE,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    template_df = pd.read_excel(template_path)
    prediction_df = pd.read_excel(prediction_path)
    out_df = build_complete_classification_table(template_df, prediction_df, VISION_LABEL_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GIS-ready Vision_Only complete classification table.")
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_path = export_vision_only_table(args.prediction, args.template, args.output)
    print(f"Vision_Only完整分类表已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

