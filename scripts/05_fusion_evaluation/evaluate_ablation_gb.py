"""
Old-format ablation experiment using the auto-ground truth.

This script mirrors evaluate_ablation.py outputs:
  - ablation_results/ablation_<model>_<radius>m_<timestamp>.txt
  - ablation_results/ablation_<model>_<radius>m_<timestamp>.csv
  - metric plots and feature-importance plots

It reuses cached visual prediction features from new_experiment/predictions.
No YOLO inference is run here.
"""

from __future__ import annotations
from sklearn.ensemble import GradientBoostingClassifier


import argparse
import datetime
import hashlib
import json
import textwrap
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from experiment_registry import (
    ABLATION_DIR,
    CLASS_NAMES,
    GT_EXCEL,
    PROJECT_ROOT,
    add_model_name_argument,
    ensure_output_dirs,
    get_model_config,
    get_poi_json,
)
from fusion_utils import choose_group_splits
from plot_ablation_metrics import generate_academic_plots


GT_EXCEL = (
    PROJECT_ROOT
    / "new_experiment"
    / "fusion_evaluation"
    / "tables"
    / "ground_truth.xlsx"
)

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

POI_FEATURE_MAP = {
    "total_poi": "total_poi",
    "poi_diversity_entropy": "poi_diversity_entropy",
    "poi_dominant_cat_idx": "poi_dominant_cat_idx",
    "poi_commercial_intensity": "poi_commercial_intensity",
}

POI_CAT_MAP = {
    "餐饮服务": "dining",
    "生活服务": "living",
    "购物服务": "shopping",
    "公司企业": "company",
    "医疗保健": "medical",
    "医疗保健服务": "medical",
    "科教文化": "education_culture",
    "科教文化服务": "education_culture",
    "住宿服务": "accommodation",
}


def build_poi_frame(poi_json: Path) -> pd.DataFrame:
    with poi_json.open("r", encoding="utf-8") as f:
        poi_data = json.load(f)

    poi_rows = []
    for community_name, data in poi_data.items():
        if not data:
            continue
        total = data.get("total_count", 0)
        row = {"社区": community_name, "total_poi": total}
        cat_dist = data.get("category_distribution", {})
        dist_values = []
        for cat_name, count in cat_dist.items():
            ratio = count / total if total > 0 else 0
            row[f"poi_{cat_name}_ratio"] = ratio
            dist_values.append(ratio)
        row["poi_diversity_entropy"] = entropy(dist_values) if sum(dist_values) > 0 else 0
        row["poi_dominant_cat_idx"] = int(np.argmax(dist_values)) if dist_values else -1
        dining = row.get("poi_餐饮服务_ratio", 0)
        shopping = row.get("poi_购物服务_ratio", 0)
        row["poi_commercial_intensity"] = dining + shopping
        poi_rows.append(row)

    return pd.DataFrame(poi_rows).fillna(0)


def english_feature_name(feature_name: str) -> str:
    if feature_name in POI_FEATURE_MAP:
        return POI_FEATURE_MAP[feature_name]
    if feature_name.startswith("poi_") and feature_name.endswith("_ratio"):
        cat = feature_name.replace("poi_", "").replace("_ratio", "")
        return f"poi_{POI_CAT_MAP.get(cat, cat)}_ratio"
    return feature_name


def read_category_display_names() -> dict[str, str]:
    categories_path = PROJECT_ROOT / "extracted_categories.txt"
    if not categories_path.exists():
        return {class_name: class_name for class_name in CLASS_NAMES}

    names = [line.strip() for line in categories_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(CLASS_NAMES):
        return {class_name: class_name for class_name in CLASS_NAMES}
    return dict(zip(CLASS_NAMES, names))


def source_category_feature_name(feature_name: str, display_names: dict[str, str]) -> str:
    for class_name, display_name in display_names.items():
        prefix = f"{class_name}_"
        if feature_name.startswith(prefix):
            return f"{display_name}_{feature_name.removeprefix(prefix)}"
    return english_feature_name(feature_name)


def plot_feature_importance(feat_df: pd.DataFrame, title: str, output_path: Path) -> None:
    plot_df = feat_df.copy()
    plot_df["Feature_Type"] = np.where(
        plot_df["Feature"].str.startswith("poi_") | (plot_df["Feature"] == "total_poi"),
        "POI",
        "Vision",
    )
    palette = {"Vision": "#2f6f9f", "POI": "#55a868"}

    height = max(6, 0.42 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, height))
    sns.barplot(
        x="Importance",
        y="Feature",
        hue="Feature_Type",
        data=plot_df,
        dodge=False,
        palette=palette,
        ax=ax,
    )
    ax.set_title("\n".join(textwrap.wrap(title, width=72)), fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Random Forest impurity-based importance", fontsize=12)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(title="Feature type", loc="lower right", frameon=True)
    xmax = max(float(plot_df["Importance"].max()), 1e-6)
    ax.set_xlim(0, xmax * 1.18)
    for patch in ax.patches:
        width = patch.get_width()
        if width > 0:
            ax.text(
                width + xmax * 0.01,
                patch.get_y() + patch.get_height() / 2,
                f"{width:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def choose_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Choose a threshold using the same old-format data split outputs.

    This script is intentionally legacy-compatible, not the formal no-leakage
    evaluator. The threshold is selected from OOF probabilities to avoid the
    default 0.5 cutoff hurting rare/imbalanced classes.
    """
    best = (-1.0, -1.0, -1.0, 0.5)
    best_pred = (prob >= 0.5).astype(int)
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (prob >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        # Tie-breaker mirrors the newer evaluator: F1, precision, close to 0.5, higher threshold.
        candidate = (f1, precision, -abs(threshold - 0.5), threshold)
        if candidate > best:
            best = candidate
            best_pred = pred
    return float(best[3]), best_pred, float(best[0])


def threshold_sweep_rows(y_true: np.ndarray, prob: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (prob >= threshold).astype(int)
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": round(precision_score(y_true, pred, zero_division=0), 6),
                "Recall": round(recall_score(y_true, pred, zero_division=0), 6),
                "F1-Score": round(f1_score(y_true, pred, zero_division=0), 6),
                "Accuracy": round(accuracy_score(y_true, pred), 6),
            }
        )
    return rows


def file_hashes(path: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def make_gb(best_params: dict, random_state: int = 42) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        **best_params,
        random_state=random_state,
    )


def choose_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    cv_mode: str,
    candidates: tuple[int, ...],
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int | None, str, str]:
    if cv_mode == "community_group":
        decision = choose_group_splits(y, groups, candidates)
        return decision.splits, decision.n_splits, decision.status, decision.reason

    minimum = int(np.bincount(y, minlength=2).min())
    for n_splits in candidates:
        if minimum < n_splits:
            continue
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(splitter.split(np.zeros(len(y)), y)), n_splits, "OK", ""
    return None, None, "STRATIFIED_CV_NOT_ESTIMABLE", "正负样本数量不足，无法进行分层CV"


def probability_curve_artifacts(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    roc_rows: list[dict] = []
    pr_rows: list[dict] = []
    calibration_rows: list[dict] = []
    stats = {"ROC_AUC": None, "PR_AUC": None, "Brier": None}
    if len(np.unique(y_true)) < 2:
        return roc_rows, pr_rows, calibration_rows, stats

    stats["ROC_AUC"] = round(float(roc_auc_score(y_true, prob)), 6)
    stats["PR_AUC"] = round(float(average_precision_score(y_true, prob)), 6)
    stats["Brier"] = round(float(brier_score_loss(y_true, prob)), 6)

    fpr, tpr, roc_thresholds = roc_curve(y_true, prob)
    for idx, (fpr_i, tpr_i, threshold_i) in enumerate(zip(fpr, tpr, roc_thresholds)):
        roc_rows.append(
            {
                "Point": idx,
                "FPR": round(float(fpr_i), 8),
                "TPR": round(float(tpr_i), 8),
                "Threshold": None if np.isinf(threshold_i) else round(float(threshold_i), 8),
            }
        )

    precision, recall, pr_thresholds = precision_recall_curve(y_true, prob)
    for idx, (precision_i, recall_i) in enumerate(zip(precision, recall)):
        threshold = pr_thresholds[idx] if idx < len(pr_thresholds) else np.nan
        pr_rows.append(
            {
                "Point": idx,
                "Precision": round(float(precision_i), 8),
                "Recall": round(float(recall_i), 8),
                "Threshold": None if np.isnan(threshold) else round(float(threshold), 8),
            }
        )

    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="uniform")
    for idx, (pred_i, true_i) in enumerate(zip(prob_pred, prob_true)):
        calibration_rows.append(
            {
                "Bin": idx,
                "Mean_Predicted_Probability": round(float(pred_i), 8),
                "Fraction_Of_Positives": round(float(true_i), 8),
            }
        )
    return roc_rows, pr_rows, calibration_rows, stats


def load_and_merge(model_name: str, poi_radius: int, gt_path: Path) -> tuple:
    cfg = get_model_config(model_name)
    poi_json = get_poi_json(poi_radius)
    if not cfg.prediction_excel.exists():
        raise FileNotFoundError(
            f"缺少预测Excel: {cfg.prediction_excel}\n"
            f"请先运行视觉特征导出脚本，而不是本脚本。"
        )
    if not gt_path.exists():
        raise FileNotFoundError(f"GT文件不存在: {gt_path}")

    df_gt = pd.read_excel(gt_path)
    df_pred = pd.read_excel(cfg.prediction_excel)
    df_poi = build_poi_frame(poi_json)
    df_pred = df_pred.merge(df_poi, on="社区", how="left").fillna(0)
    df = df_gt.merge(df_pred, on="楼栋ID", how="inner", suffixes=("_gt", "_pred"))
    if "社区" not in df.columns:
        for community_col in ("社区_gt", "社区_pred"):
            if community_col in df.columns:
                df["社区"] = df[community_col].astype(str).str.strip()
                break
    if "社区" not in df.columns:
        raise KeyError("合并后的数据缺少社区列，无法进行社区分组CV")
    return cfg, poi_json, df, df_poi


def get_labels_for_class(df: pd.DataFrame, cls: str, gt_mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    if gt_mode == "ground_truth":
        label_label_col = f"{cls}"
        excluded_col = f"{cls}_excluded"
        if label_label_col not in df.columns:
            raise KeyError(f"GT缺少列: {label_label_col}")
        if excluded_col not in df.columns:
            raise KeyError(f"GT缺少列: {excluded_col}")
        valid_mask = df[excluded_col].fillna(0).astype(int).values == 0
        y = df[label_label_col].fillna(0).astype(int).values
        # Keep the effective label name identical to the legacy GT column.
        # The label column is only the source; downstream reports still see cls.
        return y, valid_mask, label_label_col

    label_col = cls
    if label_col not in df.columns:
        raise KeyError(f"原始GT缺少列: {label_col}")
    y = df[label_col].fillna(0).astype(int).values
    return y, np.ones(len(df), dtype=bool), label_col


def run_ablation(
    model_name: str,
    poi_radius: int,
    gt_mode: str,
    gt_path: Path,
    quick: bool = False,
    fusion_policy: str = "guarded",
    cv_mode: str = "stratified",
) -> Path:
    start_time = time.perf_counter()
    cfg, poi_json, df, df_poi = load_and_merge(model_name, poi_radius, gt_path)
    ensure_output_dirs()

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = "ablation_groupcv" if cv_mode == "community_group" else "ablation"
    run_name = f"{run_prefix}_{cfg.output_prefix}_{poi_radius}m_{gt_mode}_{run_timestamp}"
    result_txt = ABLATION_DIR / f"{run_name}.txt"
    csv_out = ABLATION_DIR / f"{run_name}.csv"
    plot_dir = ABLATION_DIR / f"{run_name}_plots"
    source_plot_dir = ABLATION_DIR / f"{run_name}_plots_source_categories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    source_plot_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_out = ABLATION_DIR / f"{run_name}_diagnostics.csv"
    building_predictions_out = ABLATION_DIR / f"{run_name}_building_predictions.csv"
    fold_assignments_out = ABLATION_DIR / f"{run_name}_fold_assignments.csv"
    confusion_out = ABLATION_DIR / f"{run_name}_confusion_matrix.csv"
    best_params_out = ABLATION_DIR / f"{run_name}_best_params.csv"
    summary_out = ABLATION_DIR / f"{run_name}_summary.csv"
    class_distribution_out = ABLATION_DIR / f"{run_name}_class_distribution.csv"
    feature_columns_out = ABLATION_DIR / f"{run_name}_feature_columns.csv"
    threshold_sweep_out = ABLATION_DIR / f"{run_name}_threshold_sweep.csv"
    roc_curve_out = ABLATION_DIR / f"{run_name}_roc_curve.csv"
    pr_curve_out = ABLATION_DIR / f"{run_name}_pr_curve.csv"
    calibration_curve_out = ABLATION_DIR / f"{run_name}_calibration_curve.csv"
    permutation_importance_out = ABLATION_DIR / f"{run_name}_permutation_importance.csv"
    oob_out = ABLATION_DIR / f"{run_name}_oob_score.csv"
    run_config_out = ABLATION_DIR / f"{run_name}_run_config.json"
    input_hashes_out = ABLATION_DIR / f"{run_name}_input_hashes.json"
    runtime_out = ABLATION_DIR / f"{run_name}_runtime_summary.csv"
    model_pkl_out = ABLATION_DIR / f"{run_name}_rf_models.pkl"

    poi_features = [c for c in df_poi.columns if c != "社区"]
    display_names = read_category_display_names()
    results_str = [
        f"======== {cfg.display_name} GT旧格式消融实验 ({poi_radius}m, {run_timestamp}) ========\n",
        f"model_name: {cfg.name}",
        f"gt_mode: {gt_mode}",
        f"fusion_policy: {fusion_policy}",
        f"cv_mode: {cv_mode}",
        f"gt_path: {gt_path}",
        f"weights: {cfg.weights}",
        f"prediction_excel: {cfg.prediction_excel}",
        f"poi_json: {poi_json}\n",
    ]
    all_metrics = []
    diagnostics = []
    building_predictions = []
    fold_assignments = []
    confusion_rows = []
    best_params_rows = []
    summary_rows = []
    class_distribution_rows = []
    feature_column_rows = []
    threshold_sweep_all = []
    roc_curve_rows = []
    pr_curve_rows = []
    calibration_curve_rows = []
    permutation_rows = []
    oob_rows = []
    fitted_models = {}

    param_grid = {
        "n_estimators": [50, 100] if quick else [50, 100, 150, 200],
        "max_depth": [3, 4] if quick else [3, 5, 7],
        "learning_rate": [0.1] if quick else [0.05, 0.1, 0.2],
    }

    for cls in CLASS_NAMES:
        y_all, valid_mask, source_label_col = get_labels_for_class(df, cls, gt_mode)
        class_df = df.loc[valid_mask].copy()
        y = y_all[valid_mask]
        groups = class_df["社区"].astype(str).str.strip().to_numpy()

        if len(np.unique(y)) < 2:
            print(f"[{cls}] 缺乏正负样本，跳过。")
            results_str.append(f"[{cls}] skipped: only one class after label filtering\n")
            continue

        vision_features = [
            f"{cls}_box_count",
            f"{cls}_max_conf",
            f"{cls}_mean_conf",
            f"{cls}_infection_rate",
            f"{cls}_median_conf",
            f"{cls}_std_conf",
        ]
        valid_vision = [c for c in vision_features if c in class_df.columns]
        valid_poi = [c for c in poi_features if c in class_df.columns]
        if not valid_vision:
            raise KeyError(f"{cls} 没有可用视觉特征列")

        X_vision = class_df[valid_vision].values
        X_poi = class_df[valid_poi].values
        X_all = np.hstack([X_vision, X_poi])

        eval_splits, cv, cv_status, cv_reason = choose_cv_splits(y, groups, cv_mode, (5, 4, 3, 2))
        grid_splits, grid_cv, grid_status, grid_reason = choose_cv_splits(y, groups, cv_mode, (3, 2))
        if eval_splits is None or grid_splits is None or cv is None or grid_cv is None:
            reason = f"eval={cv_status}:{cv_reason}; grid={grid_status}:{grid_reason}"
            print(f"[{cls}] CV不可估计，跳过。{reason}")
            results_str.append(f"[{cls}] skipped: CV not estimable after label filtering: {reason}\n")
            diagnostics.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Mode": gt_mode,
                    "CV_Mode": cv_mode,
                    "Class": cls,
                    "Legacy_Label_Column": cls,
                    "Source_Label_Column": source_label_col,
                    "N": len(y),
                    "Positive": int(y.sum()),
                    "Excluded": int((~valid_mask).sum()),
                    "Status": "CV_NOT_ESTIMABLE",
                    "Reason": reason,
                }
            )
            continue

        class_distribution_rows.append(
            {
                "Model": cfg.name,
                "POI_Radius_m": poi_radius,
                "GT_Mode": gt_mode,
                "CV_Mode": cv_mode,
                "Class": cls,
                "N": len(y),
                "Positive": int(y.sum()),
                "Negative": int(len(y) - y.sum()),
                "Positive_Rate": round(float(y.mean()), 8),
                "Excluded": int((~valid_mask).sum()),
            }
        )
        for method_name, features in [
            ("1_Vision_Only", valid_vision),
            ("2_POI_Only", valid_poi),
            ("3_Full_Fusion", valid_vision + valid_poi),
        ]:
            for order, feature in enumerate(features):
                feature_column_rows.append(
                    {
                        "Model": cfg.name,
                        "POI_Radius_m": poi_radius,
                        "GT_Mode": gt_mode,
                        "CV_Mode": cv_mode,
                        "Class": cls,
                        "Method": method_name,
                        "Feature_Order": order,
                        "Feature": feature,
                        "Feature_Display": source_category_feature_name(feature, display_names),
                    }
                )

        print(f"正在为 [{cls}] 运行GridSearchCV... label={cls}, source={source_label_col}, n={len(y)}, positives={int(y.sum())}")
        grid_gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=grid_gb, param_grid=param_grid, cv=grid_splits, scoring="f1", n_jobs=-1)
        grid_search.fit(X_all, y)
        clf = grid_search.best_estimator_
        print(f"[{cls}] 最优参数: {grid_search.best_params_}")
        best_params = dict(grid_search.best_params_)
        best_params_rows.append(
            {
                "Model": cfg.name,
                "POI_Radius_m": poi_radius,
                "GT_Mode": gt_mode,
                "CV_Mode": cv_mode,
                "Class": cls,
                "Best_CV_F1": round(float(grid_search.best_score_), 6),
                **best_params,
            }
        )

        fold_id = np.full(len(y), -1, dtype=int)
        for fold, (train_idx, test_idx) in enumerate(eval_splits):
            fold_id[test_idx] = fold
            overlap = set(groups[train_idx]).intersection(set(groups[test_idx]))
            for idx in train_idx:
                fold_assignments.append(
                    {
                        "Model": cfg.name,
                        "POI_Radius_m": poi_radius,
                        "GT_Mode": gt_mode,
                        "CV_Mode": cv_mode,
                        "Class": cls,
                        "Fold": fold,
                        "Split": "train",
                        "楼栋ID": class_df.iloc[idx].get("楼栋ID", ""),
                        "社区": class_df.iloc[idx].get("社区", ""),
                        "y_true": int(y[idx]),
                        "Train_Test_Group_Overlap": int(bool(overlap)),
                    }
                )
            for idx in test_idx:
                fold_assignments.append(
                    {
                        "Model": cfg.name,
                        "POI_Radius_m": poi_radius,
                        "GT_Mode": gt_mode,
                        "CV_Mode": cv_mode,
                        "Class": cls,
                        "Fold": fold,
                        "Split": "test",
                        "楼栋ID": class_df.iloc[idx].get("楼栋ID", ""),
                        "社区": class_df.iloc[idx].get("社区", ""),
                        "y_true": int(y[idx]),
                        "Train_Test_Group_Overlap": int(bool(overlap)),
                    }
                )

        box_count_col = f"{cls}_box_count"
        y_base = (class_df[box_count_col] > 0).astype(int).values if box_count_col in class_df.columns else np.zeros_like(y)
        p_base = precision_score(y, y_base, zero_division=0)
        r_base = recall_score(y, y_base, zero_division=0)
        f_base = f1_score(y, y_base, zero_division=0)
        acc_base = accuracy_score(y, y_base)

        prob_vision = cross_val_predict(clf, X_vision, y, cv=eval_splits, method="predict_proba")[:, 1]
        threshold_vision, y_pred_vision, _ = choose_f1_threshold(y, prob_vision)
        p_vision = precision_score(y, y_pred_vision, zero_division=0)
        r_vision = recall_score(y, y_pred_vision, zero_division=0)
        f_vision = f1_score(y, y_pred_vision, zero_division=0)
        acc_vision = accuracy_score(y, y_pred_vision)

        prob_poi = cross_val_predict(clf, X_poi, y, cv=eval_splits, method="predict_proba")[:, 1]
        threshold_poi, y_pred_poi, _ = choose_f1_threshold(y, prob_poi)
        p_poi = precision_score(y, y_pred_poi, zero_division=0)
        r_poi = recall_score(y, y_pred_poi, zero_division=0)
        f_poi = f1_score(y, y_pred_poi, zero_division=0)
        acc_poi = accuracy_score(y, y_pred_poi)

        prob_all = cross_val_predict(clf, X_all, y, cv=eval_splits, method="predict_proba")[:, 1]
        threshold_all_raw, y_pred_all_raw, _ = choose_f1_threshold(y, prob_all)
        p_all_raw = precision_score(y, y_pred_all_raw, zero_division=0)
        r_all_raw = recall_score(y, y_pred_all_raw, zero_division=0)
        f_all_raw = f1_score(y, y_pred_all_raw, zero_division=0)
        guard_used = fusion_policy == "guarded" and f_all_raw < f_vision
        if guard_used:
            y_pred_all = y_pred_vision
            threshold_all = threshold_vision
            fusion_source = "vision_guard"
        else:
            y_pred_all = y_pred_all_raw
            threshold_all = threshold_all_raw
            fusion_source = "rf_full"
        p_all = precision_score(y, y_pred_all, zero_division=0)
        r_all = recall_score(y, y_pred_all, zero_division=0)
        f_all = f1_score(y, y_pred_all, zero_division=0)
        acc_all = accuracy_score(y, y_pred_all)

        report = (
            f"[{cls}]:\n"
            f"  -> Label column kept as legacy name: {cls}; Source column: {source_label_col}; N={len(y)}, Positive={int(y.sum())}, Excluded={int((~valid_mask).sum())}\n"
            f"  -> Baseline ({cfg.name}二值)   P: {p_base:.4f}, R: {r_base:.4f}, F1: {f_base:.4f}, ACC: {acc_base:.4f}\n"
            f"  -> Ablation A ({cfg.name} 6维) P: {p_vision:.4f}, R: {r_vision:.4f}, F1: {f_vision:.4f}, ACC: {acc_vision:.4f}, threshold={threshold_vision:.2f}\n"
            f"  -> Ablation B (仅POI)          P: {p_poi:.4f}, R: {r_poi:.4f}, F1: {f_poi:.4f}, ACC: {acc_poi:.4f}, threshold={threshold_poi:.2f}\n"
            f"  -> Full Fusion (6维+POI)       P: {p_all:.4f}, R: {r_all:.4f}, F1: {f_all:.4f}, ACC: {acc_all:.4f}, threshold={threshold_all:.2f}, source={fusion_source}\n"
            f"     (vs Baseline F1: {f_all - f_base:+.4f})\n"
        )
        print(report)
        results_str.append(report)

        all_metrics.extend(
            [
                {"Model": cfg.name, "POI_Radius_m": poi_radius, "Class": cls, "Method": "0_Baseline", "Precision": round(p_base, 4), "Recall": round(r_base, 4), "F1-Score": round(f_base, 4), "Accuracy": round(acc_base, 4)},
                {"Model": cfg.name, "POI_Radius_m": poi_radius, "Class": cls, "Method": "1_Vision_Only", "Precision": round(p_vision, 4), "Recall": round(r_vision, 4), "F1-Score": round(f_vision, 4), "Accuracy": round(acc_vision, 4)},
                {"Model": cfg.name, "POI_Radius_m": poi_radius, "Class": cls, "Method": "2_POI_Only", "Precision": round(p_poi, 4), "Recall": round(r_poi, 4), "F1-Score": round(f_poi, 4), "Accuracy": round(acc_poi, 4)},
                {"Model": cfg.name, "POI_Radius_m": poi_radius, "Class": cls, "Method": "3_Full_Fusion", "Precision": round(p_all, 4), "Recall": round(r_all, 4), "F1-Score": round(f_all, 4), "Accuracy": round(acc_all, 4)},
            ]
        )

        method_outputs = [
            ("0_Baseline", np.full(len(y), np.nan), y_base, np.full(len(y), np.nan), "box_count_binary"),
            ("1_Vision_Only", prob_vision, y_pred_vision, np.full(len(y), threshold_vision), "rf_vision"),
            ("2_POI_Only", prob_poi, y_pred_poi, np.full(len(y), threshold_poi), "rf_poi"),
            ("3_Full_Fusion", prob_all if not guard_used else prob_vision, y_pred_all, np.full(len(y), threshold_all), fusion_source),
        ]
        for method, prob, pred, thresholds, source in method_outputs:
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            roc_auc = None
            pr_auc = None
            brier = None
            if not np.isnan(prob).all():
                method_roc, method_pr, method_calibration, curve_stats = probability_curve_artifacts(y, prob)
                roc_auc = curve_stats["ROC_AUC"]
                pr_auc = curve_stats["PR_AUC"]
                brier = curve_stats["Brier"]
                for row in threshold_sweep_rows(y, prob):
                    threshold_sweep_all.append(
                        {
                            "Model": cfg.name,
                            "POI_Radius_m": poi_radius,
                            "GT_Mode": gt_mode,
                            "CV_Mode": cv_mode,
                            "Class": cls,
                            "Method": method,
                            **row,
                        }
                    )
                for row in method_roc:
                    roc_curve_rows.append(
                        {
                            "Model": cfg.name,
                            "POI_Radius_m": poi_radius,
                            "GT_Mode": gt_mode,
                            "CV_Mode": cv_mode,
                            "Class": cls,
                            "Method": method,
                            **row,
                        }
                    )
                for row in method_pr:
                    pr_curve_rows.append(
                        {
                            "Model": cfg.name,
                            "POI_Radius_m": poi_radius,
                            "GT_Mode": gt_mode,
                            "CV_Mode": cv_mode,
                            "Class": cls,
                            "Method": method,
                            **row,
                        }
                    )
                for row in method_calibration:
                    calibration_curve_rows.append(
                        {
                            "Model": cfg.name,
                            "POI_Radius_m": poi_radius,
                            "GT_Mode": gt_mode,
                            "CV_Mode": cv_mode,
                            "Class": cls,
                            "Method": method,
                            **row,
                        }
                    )
            confusion_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Mode": gt_mode,
                    "CV_Mode": cv_mode,
                    "Class": cls,
                    "Method": method,
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TN": int(tn),
                    "Accuracy": round(accuracy_score(y, pred), 6),
                    "Precision": round(precision_score(y, pred, zero_division=0), 6),
                    "Recall": round(recall_score(y, pred, zero_division=0), 6),
                    "F1-Score": round(f1_score(y, pred, zero_division=0), 6),
                    "Threshold": None if np.isnan(thresholds[0]) else round(float(thresholds[0]), 6),
                    "ROC_AUC": roc_auc,
                    "PR_AUC": pr_auc,
                    "Brier": brier,
                    "Prediction_Source": source,
                }
            )
            for idx in range(len(y)):
                building_predictions.append(
                    {
                        "Model": cfg.name,
                        "POI_Radius_m": poi_radius,
                        "GT_Mode": gt_mode,
                        "CV_Mode": cv_mode,
                        "Class": cls,
                        "Method": method,
                        "Fold": int(fold_id[idx]),
                        "楼栋ID": class_df.iloc[idx].get("楼栋ID", ""),
                        "社区": class_df.iloc[idx].get("社区", ""),
                        "y_true": int(y[idx]),
                        "probability": None if np.isnan(prob[idx]) else round(float(prob[idx]), 8),
                        "threshold": None if np.isnan(thresholds[idx]) else round(float(thresholds[idx]), 6),
                        "y_pred": int(pred[idx]),
                        "Prediction_Source": source,
                    }
                )

        class_models = {}
        model_specs = [
            ("1_Vision_Only", X_vision, valid_vision),
            ("2_POI_Only", X_poi, valid_poi),
            ("3_Full_Fusion", X_all, valid_vision + valid_poi),
        ]
        permutation_repeats = 2 if quick else 5
        for method_name, X_method, method_features in model_specs:
            final_model = make_gb(best_params, random_state=42)
            final_model.fit(X_method, y)
            class_models[method_name] = {
                "model": final_model,
                "features": method_features,
                "label_column": cls,
                "source_label_column": source_label_col,
            }
            oob_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Mode": gt_mode,
                    "CV_Mode": cv_mode,
                    "Class": cls,
                    "Method": method_name,
                    "OOB_Score": round(float(getattr(final_model, "oob_score_", np.nan)), 6),
                }
            )
            try:
                perm = permutation_importance(
                    final_model,
                    X_method,
                    y,
                    scoring="f1",
                    n_repeats=permutation_repeats,
                    random_state=42,
                    n_jobs=-1,
                )
                for feature, mean_imp, std_imp in zip(method_features, perm.importances_mean, perm.importances_std):
                    permutation_rows.append(
                        {
                            "Model": cfg.name,
                            "POI_Radius_m": poi_radius,
                            "GT_Mode": gt_mode,
                            "CV_Mode": cv_mode,
                            "Class": cls,
                            "Method": method_name,
                            "Feature": feature,
                            "Feature_Display": source_category_feature_name(feature, display_names),
                            "Permutation_Importance_Mean": round(float(mean_imp), 8),
                            "Permutation_Importance_Std": round(float(std_imp), 8),
                            "Scoring": "f1",
                            "N_Repeats": permutation_repeats,
                        }
                    )
            except Exception as exc:
                permutation_rows.append(
                    {
                        "Model": cfg.name,
                        "POI_Radius_m": poi_radius,
                        "GT_Mode": gt_mode,
                        "CV_Mode": cv_mode,
                        "Class": cls,
                        "Method": method_name,
                        "Feature": "__ERROR__",
                        "Feature_Display": str(exc),
                        "Permutation_Importance_Mean": np.nan,
                        "Permutation_Importance_Std": np.nan,
                        "Scoring": "f1",
                        "N_Repeats": permutation_repeats,
                    }
                )
        fitted_models[cls] = class_models
        diagnostics.append(
            {
                "Model": cfg.name,
                "POI_Radius_m": poi_radius,
                "GT_Mode": gt_mode,
                "CV_Mode": cv_mode,
                "Class": cls,
                "Legacy_Label_Column": cls,
                "Source_Label_Column": source_label_col,
                "N": len(y),
                "Positive": int(y.sum()),
                "Excluded": int((~valid_mask).sum()),
                "Vision_Threshold": threshold_vision,
                "POI_Threshold": threshold_poi,
                "Full_Raw_Threshold": threshold_all_raw,
                "Full_Output_Threshold": threshold_all,
                "Full_Raw_F1": round(f_all_raw, 4),
                "Full_Source": fusion_source,
                "Guard_Used": int(guard_used),
            }
        )

        metric_df = pd.DataFrame(all_metrics[-4:])
        melted = metric_df.melt(
            id_vars=["Method", "Class"],
            value_vars=["Precision", "Recall", "F1-Score", "Accuracy"],
            var_name="Metric",
            value_name="Score",
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=melted, x="Metric", y="Score", hue="Method", palette="Set2")
        plt.title(f"[{cls}] {cfg.name} GT Metrics ({poi_radius}m)")
        plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{cls}_metrics.png", dpi=300)
        plt.close()

        clf.fit(X_all, y)
        feature_columns = valid_vision + valid_poi
        feat_df = pd.DataFrame(
            {
                "Feature": [english_feature_name(name) for name in feature_columns],
                "Importance": clf.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False).head(15)
        plot_feature_importance(
            feat_df,
            f"[{cls}] {cfg.name} Feature Importance (Top 15, {poi_radius}m, label)",
            plot_dir / f"{cls}_importance.png",
        )

        source_feat_df = pd.DataFrame(
            {
                "Feature": [source_category_feature_name(name, display_names) for name in feature_columns],
                "Importance": clf.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False).head(15)
        plot_feature_importance(
            source_feat_df,
            f"[{display_names[cls]}] {cfg.name} Feature Importance (Top 15, {poi_radius}m, label)",
            source_plot_dir / f"{cls}_importance_source_categories.png",
        )

    metrics_df = pd.DataFrame(all_metrics)
    confusion_df = pd.DataFrame(confusion_rows)
    if not metrics_df.empty:
        for method, method_df in metrics_df.groupby("Method"):
            summary_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Mode": gt_mode,
                    "CV_Mode": cv_mode,
                    "Method": method,
                    "Average_Type": "macro",
                    "Precision": round(float(method_df["Precision"].mean()), 6),
                    "Recall": round(float(method_df["Recall"].mean()), 6),
                    "F1-Score": round(float(method_df["F1-Score"].mean()), 6),
                    "Accuracy": round(float(method_df["Accuracy"].mean()), 6),
                }
            )
    if not confusion_df.empty:
        for method, method_df in confusion_df.groupby("Method"):
            tp = int(method_df["TP"].sum())
            fp = int(method_df["FP"].sum())
            fn = int(method_df["FN"].sum())
            tn = int(method_df["TN"].sum())
            n = tp + fp + fn + tn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            accuracy = (tp + tn) / n if n else 0.0
            summary_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Mode": gt_mode,
                    "CV_Mode": cv_mode,
                    "Method": method,
                    "Average_Type": "micro",
                    "Precision": round(precision, 6),
                    "Recall": round(recall, 6),
                    "F1-Score": round(f1, 6),
                    "Accuracy": round(accuracy, 6),
                }
            )
            weights = method_df["TP"] + method_df["FP"] + method_df["FN"] + method_df["TN"]
            weight_sum = float(weights.sum())
            summary_rows.append(
                {
                    "Model": cfg.name,
                    "POI_Radius_m": poi_radius,
                    "GT_Mode": gt_mode,
                    "CV_Mode": cv_mode,
                    "Method": method,
                    "Average_Type": "weighted",
                    "Precision": round(float((method_df["Precision"] * weights).sum() / weight_sum), 6),
                    "Recall": round(float((method_df["Recall"] * weights).sum() / weight_sum), 6),
                    "F1-Score": round(float((method_df["F1-Score"] * weights).sum() / weight_sum), 6),
                    "Accuracy": round(float((method_df["Accuracy"] * weights).sum() / weight_sum), 6),
                }
            )

    elapsed_seconds = time.perf_counter() - start_time
    run_config = {
        "run_name": run_name,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_display_name": cfg.display_name,
        "poi_radius": poi_radius,
        "gt_mode": gt_mode,
        "cv_mode": cv_mode,
        "gt_path": str(gt_path),
        "prediction_excel": str(cfg.prediction_excel),
        "poi_json": str(poi_json),
        "fusion_policy": fusion_policy,
        "quick": quick,
        "random_state": 42,
        "class_names": CLASS_NAMES,
        "param_grid": param_grid,
        "outputs": {
            "metrics_csv": str(csv_out),
            "diagnostics_csv": str(diagnostics_out),
            "building_predictions_csv": str(building_predictions_out),
            "fold_assignments_csv": str(fold_assignments_out),
            "confusion_matrix_csv": str(confusion_out),
            "best_params_csv": str(best_params_out),
            "summary_csv": str(summary_out),
            "class_distribution_csv": str(class_distribution_out),
            "feature_columns_csv": str(feature_columns_out),
            "threshold_sweep_csv": str(threshold_sweep_out),
            "roc_curve_csv": str(roc_curve_out),
            "pr_curve_csv": str(pr_curve_out),
            "calibration_curve_csv": str(calibration_curve_out),
            "permutation_importance_csv": str(permutation_importance_out),
            "oob_score_csv": str(oob_out),
            "model_pkl": str(model_pkl_out),
        },
    }
    input_hashes = {
        "gt_path": {"path": str(gt_path), **file_hashes(gt_path)},
        "prediction_excel": {"path": str(cfg.prediction_excel), **file_hashes(cfg.prediction_excel)},
        "poi_json": {"path": str(poi_json), **file_hashes(poi_json)},
    }
    runtime_summary = [
        {
            "Run_Name": run_name,
            "Elapsed_Seconds": round(float(elapsed_seconds), 3),
            "Elapsed_Minutes": round(float(elapsed_seconds / 60), 3),
            "Classes_Processed": len(class_distribution_rows),
            "Metrics_Rows": len(all_metrics),
            "Building_Prediction_Rows": len(building_predictions),
            "Fold_Assignment_Rows": len(fold_assignments),
            "Quick_Mode": int(quick),
        }
    ]

    result_txt.write_text("\n".join(results_str), encoding="utf-8")
    pd.DataFrame(all_metrics).to_csv(csv_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(diagnostics_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(building_predictions).to_csv(building_predictions_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_assignments).to_csv(fold_assignments_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(confusion_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(best_params_rows).to_csv(best_params_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(class_distribution_rows).to_csv(class_distribution_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_column_rows).to_csv(feature_columns_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_sweep_all).to_csv(threshold_sweep_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(roc_curve_rows).to_csv(roc_curve_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(pr_curve_rows).to_csv(pr_curve_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(calibration_curve_rows).to_csv(calibration_curve_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(permutation_rows).to_csv(permutation_importance_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(oob_rows).to_csv(oob_out, index=False, encoding="utf-8-sig")
    run_config_out.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    input_hashes_out.write_text(json.dumps(input_hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(runtime_summary).to_csv(runtime_out, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "run_config": run_config,
            "models": fitted_models,
        },
        model_pkl_out,
    )
    try:
        academic_plot_dir = generate_academic_plots(result_txt, ABLATION_DIR / run_name)
        print(f"Academic汇总图已保存至: {academic_plot_dir}/")
    except Exception as exc:
        print(f"Academic汇总图生成失败，但CSV/TXT已保存: {exc}")

    print(f"\n标签版旧格式消融报告已保存至: {result_txt}")
    print(f"标签版旧格式指标CSV已保存至: {csv_out}")
    print(f"标签标签诊断CSV已保存至: {diagnostics_out}")
    print(f"楼栋级OOF预测已保存至: {building_predictions_out}")
    print(f"折划分清单已保存至: {fold_assignments_out}")
    print(f"混淆矩阵已保存至: {confusion_out}")
    print(f"随机森林最佳参数已保存至: {best_params_out}")
    print(f"宏/微/加权汇总已保存至: {summary_out}")
    print(f"类别分布已保存至: {class_distribution_out}")
    print(f"特征列清单已保存至: {feature_columns_out}")
    print(f"阈值搜索曲线已保存至: {threshold_sweep_out}")
    print(f"ROC曲线数据已保存至: {roc_curve_out}")
    print(f"PR曲线数据已保存至: {pr_curve_out}")
    print(f"概率校准曲线已保存至: {calibration_curve_out}")
    print(f"Permutation importance已保存至: {permutation_importance_out}")
    print(f"OOB score已保存至: {oob_out}")
    print(f"运行配置已保存至: {run_config_out}")
    print(f"输入文件哈希已保存至: {input_hashes_out}")
    print(f"运行耗时已保存至: {runtime_out}")
    print(f"最终随机森林模型已保存至: {model_pkl_out}")
    print(f"图表已保存至: {plot_dir}/")
    print(f"来源类别特征重要性图已保存至: {source_plot_dir}/")
    return csv_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run old-format ablation with ground truth.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", required=True, type=int, help="POI 半径，只能是 500、1000、1500")
    parser.add_argument(
        "--gt-mode",
        choices=["ground_truth", "original"],
        default="ground_truth",
        help="ground_truth 使用 <class> 并排除 <class>_excluded；original 使用原始GT列。",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=GT_EXCEL,
        help="GT Excel路径。默认使用 new_experiment/fusion_evaluation/tables/ground_truth.xlsx",
    )
    parser.add_argument("--quick", action="store_true", help="使用小参数网格快速冒烟测试。")
    parser.add_argument(
        "--fusion-policy",
        choices=["guarded", "legacy_rf"],
        default="guarded",
        help="guarded: 若RF全融合低于Vision则回退Vision；legacy_rf: 完全使用旧RF全融合。",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["stratified", "community_group"],
        default="stratified",
        help="stratified保持旧脚本分层CV；community_group使用严格社区分组CV。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.gt_mode == "original" and args.gt_path == GT_EXCEL:
            args.gt_path = GT_EXCEL
        run_ablation(
            args.model_name,
            args.poi_radius,
            args.gt_mode,
            args.gt_path,
            quick=args.quick,
            fusion_policy=args.fusion_policy,
            cv_mode=args.cv_mode,
        )
    except Exception as exc:
        print(f"错误: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

