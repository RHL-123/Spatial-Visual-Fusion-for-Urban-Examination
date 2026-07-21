from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ceus_experiment_utils import class_xy, ensure_ceus_extra_dir, load_feature_table, valid_classes
from experiment_registry import add_model_name_argument


def model_zoo():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear"),
        ),
        "LinearSVM": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", max_iter=10000, dual="auto"),
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def run_fusion_models(model_name: str, poi_radius: int) -> str:
    _, _, df, poi_features = load_feature_table(model_name, poi_radius)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for class_name in valid_classes(df):
        x, y, feature_cols = class_xy(df, poi_features, class_name)
        for clf_name, clf in model_zoo().items():
            y_pred = cross_val_predict(clf, x, y, cv=cv)
            rows.append(
                {
                    "Model": model_name,
                    "POI_Radius_m": poi_radius,
                    "Class": class_name,
                    "Fusion_Classifier": clf_name,
                    "Feature_Count": len(feature_cols),
                    "Precision": precision_score(y, y_pred, zero_division=0),
                    "Recall": recall_score(y, y_pred, zero_division=0),
                    "F1-Score": f1_score(y, y_pred, zero_division=0),
                }
            )

    out_dir = ensure_ceus_extra_dir()
    out_path = out_dir / f"fusion_model_comparison_{model_name}_{poi_radius}m.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"融合分类器对比结果已保存: {out_path}")
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare alternative fusion classifiers.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", type=int, required=True)
    args = parser.parse_args()
    run_fusion_models(args.model_name, args.poi_radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

