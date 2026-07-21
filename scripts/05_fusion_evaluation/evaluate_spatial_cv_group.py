from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from evaluate_ablation import GT_EXCEL, get_labels_for_class, load_and_merge
from experiment_registry import CLASS_NAMES, GT_EXCEL, add_model_name_argument
from ceus_experiment_utils import ensure_ceus_extra_dir


def _resolve_group_column(df: pd.DataFrame, group_col: str) -> str:
    """Return a usable grouping column, preferring GT-side location fields."""
    if group_col in df.columns:
        return group_col
    for candidate in (f"{group_col}_gt", f"{group_col}_pred"):
        if candidate in df.columns:
            df[group_col] = df[candidate].astype(str).str.strip()
            return group_col
    raise ValueError(f"group-col 不存在: {group_col}. 可用列包括: {list(df.columns[:40])} ...")


def _feature_columns(df: pd.DataFrame, poi_features: list[str], class_name: str) -> list[str]:
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
    if not valid_vision:
        raise KeyError(f"{class_name} 没有可用视觉特征列")
    return valid_vision + valid_poi


def run_spatial_cv_label(
    model_name: str,
    poi_radius: int,
    group_col: str,
    gt_path: Path,
) -> Path:
    cfg, _, df, df_poi = load_and_merge(model_name, poi_radius, gt_path)
    group_col = _resolve_group_column(df, group_col)
    poi_features = [c for c in df_poi.columns if c != "社区"]

    out_rows = []
    fold_rows = []
    for class_name in CLASS_NAMES:
        y_all, valid_mask, source_label_col = get_labels_for_class(df, class_name)
        class_df = df.loc[valid_mask].copy()
        y = y_all[valid_mask]
        if len(np.unique(y)) < 2:
            out_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Path": str(gt_path),
                    "Group_Column": group_col,
                    "Folds": 0,
                    "Class": class_name,
                    "Source_Label_Column": source_label_col,
                    "N": int(len(y)),
                    "Positive": int(y.sum()),
                    "Excluded": int((~valid_mask).sum()),
                    "Group_Count": int(class_df[group_col].astype(str).nunique()),
                    "Feature_Count": 0,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-Score": np.nan,
                    "Status": "SKIPPED_ONE_CLASS",
                }
            )
            continue

        feature_cols = _feature_columns(class_df, poi_features, class_name)
        groups = class_df[group_col].astype(str).str.strip().fillna("UNKNOWN").values
        unique_groups = sorted(set(groups))
        if len(unique_groups) < 2:
            raise ValueError(f"{group_col} 只有 {len(unique_groups)} 个唯一分组，不能做空间分组验证")
        n_splits = min(5, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        x = class_df[feature_cols].values
        y_pred = cross_val_predict(clf, x, y, cv=cv, groups=groups)

        for fold, (_, test_idx) in enumerate(cv.split(x, y, groups)):
            fold_y = y[test_idx]
            fold_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "Group_Column": group_col,
                    "Class": class_name,
                    "Fold": fold,
                    "Test_N": int(len(test_idx)),
                    "Test_Positive": int(fold_y.sum()),
                    "Test_Negative": int(len(fold_y) - fold_y.sum()),
                    "Test_Group_Count": int(len(set(groups[test_idx]))),
                }
            )

        out_rows.append(
            {
                "Model": cfg.name,
                "POI_Radius_m": poi_radius,
                "GT_Path": str(gt_path),
                "Group_Column": group_col,
                "Folds": n_splits,
                "Class": class_name,
                "Source_Label_Column": source_label_col,
                "N": int(len(y)),
                "Positive": int(y.sum()),
                "Excluded": int((~valid_mask).sum()),
                "Group_Count": int(len(unique_groups)),
                "Feature_Count": int(len(feature_cols)),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1-Score": f1_score(y, y_pred, zero_division=0),
                "Status": "OK",
            }
        )

    out_dir = ensure_ceus_extra_dir()
    out_path = out_dir / f"spatial_cv_{model_name}_{poi_radius}m_{group_col}.csv"
    fold_path = out_dir / f"spatial_cv_{model_name}_{poi_radius}m_{group_col}_fold_diagnostics.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig")
    print(f"标签空间分组验证结果已保存: {out_path}")
    print(f"折诊断已保存: {fold_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ground_truth GroupKFold spatial/generalization validation.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", type=int, required=True)
    parser.add_argument("--group-col", required=True, help="例如 区市、街道、社区")
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=GT_EXCEL,
        help="GT Excel路径。默认使用 data/ground_truth/ground_truth.xlsx",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_spatial_cv_label(args.model_name, args.poi_radius, args.group_col, args.gt_mode, args.gt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

