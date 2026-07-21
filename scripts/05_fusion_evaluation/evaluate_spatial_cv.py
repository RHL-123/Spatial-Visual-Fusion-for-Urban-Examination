from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from ceus_experiment_utils import class_xy, ensure_ceus_extra_dir, load_feature_table, valid_classes
from experiment_registry import add_model_name_argument


def run_spatial_cv(model_name: str, poi_radius: int, group_col: str) -> str:
    _, _, df, poi_features = load_feature_table(model_name, poi_radius)
    if group_col not in df.columns:
        raise ValueError(f"group-col 不存在: {group_col}. 可用列包括: {list(df.columns[:20])} ...")

    out_rows = []
    groups = df[group_col].astype(str).fillna("UNKNOWN").values
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError(f"{group_col} 只有 {len(unique_groups)} 个唯一分组，不能做空间分组验证")
    n_splits = min(5, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)

    for class_name in valid_classes(df):
        x, y, feature_cols = class_xy(df, poi_features, class_name)
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        y_pred = cross_val_predict(clf, x, y, cv=cv, groups=groups)
        out_rows.append(
            {
                "Model": model_name,
                "POI_Radius_m": poi_radius,
                "Group_Column": group_col,
                "Folds": n_splits,
                "Class": class_name,
                "Feature_Count": len(feature_cols),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1-Score": f1_score(y, y_pred, zero_division=0),
            }
        )

    out_dir = ensure_ceus_extra_dir()
    out_path = out_dir / f"spatial_cv_{model_name}_{poi_radius}m_{group_col}.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"空间分组验证结果已保存: {out_path}")
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GroupKFold spatial/generalization validation.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", type=int, required=True)
    parser.add_argument("--group-col", required=True, help="例如 区市、街道、社区")
    args = parser.parse_args()
    run_spatial_cv(args.model_name, args.poi_radius, args.group_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

