from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from evaluate_ablation_nested import labelED_GT, VISION_SUFFIXES, run_class
from experiment_registry import CLASS_NAMES, GT_EXCEL, PROJECT_ROOT, get_model_config, get_poi_json


POI_CATEGORIES = ["餐饮服务", "生活服务", "购物服务", "公司企业", "医疗保健", "科教文化", "住宿服务"]
CORR_COL_TO_CAT = {
    "Dining": "餐饮服务",
    "Daily Service": "生活服务",
    "Shopping": "购物服务",
    "Corporate": "公司企业",
    "Healthcare": "医疗保健",
    "Education/Culture": "科教文化",
    "Accommodation": "住宿服务",
}
PROBLEM_TO_CLASS = {
    "Pipeline Damage": "pipeline_damage",
    "Addition/Balcony": "addition_balconies_windows",
    "Public Space Occ.": "public_spaces_illegal_occupation",
    "Aging Modification": "no_aging_place_modifications",
    "Wall Damage": "wall_damage",
    "Illegal Renovation": "illegal_renovation_expansion",
    "Elevator Addition": "elevator_addition",
}


def safe_feature_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", text).strip("_")


def normalize_category(raw: str) -> str | None:
    if not raw:
        return None
    primary = str(raw).split("|", 1)[0].split(";", 1)[0].strip()
    if primary == "医疗保健服务":
        return "医疗保健"
    if primary == "科教文化服务":
        return "科教文化"
    if primary in POI_CATEGORIES:
        return primary
    return None


def parse_distance(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, list):
        return default
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def build_extended_poi_frame(path: Path, radius: int) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    area_km2 = math.pi * (radius / 1000.0) ** 2
    for community, payload in raw.items():
        valid = (
            isinstance(payload, dict)
            and bool(payload)
            and "category_distribution" in payload
            and isinstance(payload.get("category_distribution"), dict)
        )
        if not valid:
            payload = {}
        total = float(payload.get("total_count", 0) or 0)
        distribution = payload.get("category_distribution", {}) or {}
        row: dict[str, object] = {
            "社区": str(community).strip(),
            "poi_missing_indicator": int(not valid),
            "total_poi": total,
            "log_total_poi": math.log1p(total),
            "poi_dummy_zero": 0.0,
        }

        ratios = []
        for category in POI_CATEGORIES:
            count = float(distribution.get(category, distribution.get(f"{category}服务", 0)) or 0)
            ratio = count / total if total > 0 else 0.0
            key = safe_feature_name(category)
            row[f"poi_{key}_count"] = count
            row[f"poi_{key}_log_count"] = math.log1p(count)
            row[f"poi_{key}_density"] = count / area_km2 if area_km2 > 0 else 0.0
            row[f"poi_{key}_ratio"] = ratio
            ratios.append(ratio)

        distance_values = {category: [] for category in POI_CATEGORIES}
        for poi in payload.get("poi_list", []) or []:
            if not isinstance(poi, dict):
                continue
            category = normalize_category(str(poi.get("type", "")))
            if category not in distance_values:
                continue
            distance_values[category].append(parse_distance(poi.get("distance"), float(radius)))

        for category in POI_CATEGORIES:
            key = safe_feature_name(category)
            distances = np.asarray(distance_values[category], dtype=float)
            if distances.size:
                row[f"poi_{key}_near500_count"] = float((distances <= 500).sum())
                row[f"poi_{key}_ring500_1000_count"] = float(((distances > 500) & (distances <= 1000)).sum())
                row[f"poi_{key}_mean_distance"] = float(distances.mean())
                row[f"poi_{key}_inv_distance_sum"] = float((1.0 / (1.0 + distances / 100.0)).sum())
                row[f"poi_{key}_exp_distance_sum"] = float(np.exp(-distances / 500.0).sum())
            else:
                row[f"poi_{key}_near500_count"] = 0.0
                row[f"poi_{key}_ring500_1000_count"] = 0.0
                row[f"poi_{key}_mean_distance"] = float(radius)
                row[f"poi_{key}_inv_distance_sum"] = 0.0
                row[f"poi_{key}_exp_distance_sum"] = 0.0

        row["poi_diversity_entropy"] = float(entropy(ratios)) if sum(ratios) > 0 else 0.0
        row["poi_dominant_category"] = POI_CATEGORIES[int(np.argmax(ratios))] if sum(ratios) > 0 else "NONE"
        row["poi_commercial_intensity"] = (
            float(row["poi_餐饮服务_ratio"]) + float(row["poi_购物服务_ratio"])
        )
        row["poi_life_service_intensity"] = (
            float(row["poi_生活服务_ratio"]) + float(row["poi_医疗保健_ratio"]) + float(row["poi_科教文化_ratio"])
        )
        row["poi_corporate_intensity"] = float(row["poi_公司企业_ratio"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_extended_feature_frame(model_name: str, radius: int) -> pd.DataFrame:
    cfg = get_model_config(model_name)
    features = pd.read_excel(cfg.prediction_excel)
    original = pd.read_excel(GT_EXCEL)
    ground_truth = pd.read_excel(labelED_GT, sheet_name="GT")
    if original["楼栋ID"].astype(str).tolist() != ground_truth["楼栋ID"].astype(str).tolist():
        raise ValueError("GT与原始GT楼栋顺序不一致")

    frame = original[["楼栋ID", *CLASS_NAMES]].merge(features, on="楼栋ID", how="inner", suffixes=("", "_pred"))
    if len(frame) != len(original):
        raise ValueError(f"{model_name}特征只匹配 {len(frame)}/{len(original)} 栋")

    label_columns = ["楼栋ID"]
    for class_name in CLASS_NAMES:
        label_columns.extend([f"{class_name}", f"{class_name}_excluded", f"{class_name}_label_status"])
    frame = frame.merge(ground_truth[label_columns], on="楼栋ID", how="left")

    poi = build_extended_poi_frame(get_poi_json(radius), radius)
    frame["社区"] = frame["社区"].astype(str).str.strip()
    frame = frame.merge(poi, on="社区", how="left", indicator="_poi_merge")
    unmatched = frame["_poi_merge"].ne("both")
    frame.loc[unmatched, "poi_missing_indicator"] = 1
    numeric_cols = [c for c in poi.columns if c not in {"社区", "poi_dominant_category"}]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame["poi_dominant_category"] = frame["poi_dominant_category"].fillna("MISSING").astype(str)
    return frame.drop(columns=["_poi_merge"])


def load_correlation_evidence() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    base = PROJECT_ROOT / "04_Correction_and_Fusion"
    specs = [
        ("Pearson", base / "Pearson_Correlation_Analysis" / "poi_correlation_results.xlsx", "Correlation_r"),
        ("Spearman", base / "Spearman_Correlation_Analysis" / "spearman_correlation_results.xlsx", "Spearman_rho"),
    ]
    rows = []
    for method, path, r_sheet in specs:
        r_df = pd.read_excel(path, sheet_name=r_sheet).rename(columns={"Unnamed: 0": "Problem"})
        q_df = pd.read_excel(path, sheet_name="P_FDR_corrected").rename(columns={"Unnamed: 0": "Problem"})
        for _, r_row in r_df.iterrows():
            problem = r_row["Problem"]
            class_name = PROBLEM_TO_CLASS.get(problem, str(problem))
            for col in r_df.columns[1:]:
                category = CORR_COL_TO_CAT[col]
                q_value = float(q_df.loc[q_df["Problem"].eq(problem), col].iloc[0])
                r_value = float(r_row[col])
                rows.append(
                    {
                        "Method": method,
                        "Problem": problem,
                        "Class": class_name,
                        "POI_Category_EN": col,
                        "POI_Category": category,
                        "Correlation": r_value,
                        "Abs_Correlation": abs(r_value),
                        "FDR_q": q_value,
                        "Significant_FDR05": q_value < 0.05,
                    }
                )
    evidence = pd.DataFrame(rows)
    significant: dict[str, list[str]] = {}
    top3: dict[str, list[str]] = {}
    for class_name in CLASS_NAMES:
        class_rows = evidence[evidence["Class"].eq(class_name)]
        sig = (
            class_rows[class_rows["Significant_FDR05"]]
            .sort_values(["FDR_q", "Abs_Correlation"], ascending=[True, False])["POI_Category"]
            .drop_duplicates()
            .tolist()
        )
        significant[class_name] = sig
        top = (
            class_rows[class_rows["Method"].eq("Spearman")]
            .sort_values("Abs_Correlation", ascending=False)["POI_Category"]
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        top3[class_name] = top
    return evidence, significant, top3


def ratio_cols(categories: list[str]) -> list[str]:
    return [f"poi_{safe_feature_name(category)}_ratio" for category in categories]


def enhanced_cols(categories: list[str]) -> list[str]:
    cols: list[str] = []
    for category in categories:
        key = safe_feature_name(category)
        cols.extend(
            [
                f"poi_{key}_ratio",
                f"poi_{key}_log_count",
                f"poi_{key}_near500_count",
                f"poi_{key}_ring500_1000_count",
                f"poi_{key}_inv_distance_sum",
            ]
        )
    return cols


def profile_features(profile: str, class_name: str, significant: dict[str, list[str]], top3: dict[str, list[str]]) -> tuple[list[str], str | None, str]:
    current_numeric = [
        "poi_missing_indicator",
        "total_poi",
        *ratio_cols(POI_CATEGORIES),
        "poi_diversity_entropy",
        "poi_commercial_intensity",
    ]
    current_set = set(current_numeric)
    if profile == "classwise_stage1_best":
        base = {
            "pipeline_damage": "current",
            "addition_balconies_windows": "counts_ratios7",
            "public_spaces_illegal_occupation": "counts_ratios7",
            "no_aging_place_modifications": "enhanced7",
            "wall_damage": "current",
            "illegal_renovation_expansion": "counts_ratios7",
            "elevator_addition": "current",
        }[class_name]
        numeric, categorical, _ = profile_features(base, class_name, significant, top3)
        return numeric, categorical, f"按类别使用上一轮Full Fusion最佳POI组: {base}"
    if profile == "classwise_current_or_counts":
        base = {
            "pipeline_damage": "current",
            "addition_balconies_windows": "counts_ratios7",
            "public_spaces_illegal_occupation": "counts_ratios7",
            "no_aging_place_modifications": "current",
            "wall_damage": "current",
            "illegal_renovation_expansion": "counts_ratios7",
            "elevator_addition": "current",
        }[class_name]
        numeric, categorical, _ = profile_features(base, class_name, significant, top3)
        return numeric, categorical, f"按类别在current和counts_ratios7之间择优: {base}"
    if profile == "current":
        return current_numeric, "poi_dominant_category", "当前特征: 总量+7类比例+熵+商业强度+主导类别"
    if profile == "current_no_dominant":
        return current_numeric, None, "当前特征去掉主导类别独热编码"
    if profile == "current_no_missing":
        return [c for c in current_numeric if c != "poi_missing_indicator"], "poi_dominant_category", "当前特征去掉poi_missing_indicator"
    if profile == "current_no_total":
        return [c for c in current_numeric if c != "total_poi"], "poi_dominant_category", "当前特征去掉total_poi"
    if profile == "current_no_entropy":
        return [c for c in current_numeric if c != "poi_diversity_entropy"], "poi_dominant_category", "当前特征去掉poi_diversity_entropy"
    if profile == "current_no_commercial":
        return [c for c in current_numeric if c != "poi_commercial_intensity"], "poi_dominant_category", "当前特征去掉poi_commercial_intensity"
    if profile == "current_core_no_handcrafted":
        return ["total_poi", *ratio_cols(POI_CATEGORIES)], None, "只保留total_poi和7类比例，去掉缺失、熵、商业强度、主导类别"
    if profile == "current_plus_intensities":
        cols = [*current_numeric, "log_total_poi", "poi_life_service_intensity", "poi_corporate_intensity"]
        return list(dict.fromkeys(cols)), "poi_dominant_category", "当前特征+log总量+生活服务强度+公司企业强度"
    if profile == "current_plus_log_counts":
        cols = [*current_numeric, "log_total_poi"]
        for category in POI_CATEGORIES:
            cols.append(f"poi_{safe_feature_name(category)}_log_count")
        return list(dict.fromkeys(cols)), "poi_dominant_category", "当前特征+7类log计数"
    if profile == "current_plus_densities":
        cols = [*current_numeric]
        for category in POI_CATEGORIES:
            cols.append(f"poi_{safe_feature_name(category)}_density")
        return list(dict.fromkeys(cols)), "poi_dominant_category", "当前特征+7类面积密度"
    if profile == "current_plus_distance_decay":
        cols = [*current_numeric]
        for category in POI_CATEGORIES:
            key = safe_feature_name(category)
            cols.extend([f"poi_{key}_inv_distance_sum", f"poi_{key}_exp_distance_sum"])
        return list(dict.fromkeys(cols)), "poi_dominant_category", "当前特征+7类距离衰减POI强度"
    if profile == "current_plus_near_ring":
        cols = [*current_numeric]
        for category in POI_CATEGORIES:
            key = safe_feature_name(category)
            cols.extend([f"poi_{key}_near500_count", f"poi_{key}_ring500_1000_count"])
        return list(dict.fromkeys(cols)), "poi_dominant_category", "当前特征+500m近邻数+500-1000m环带数"
    if profile == "current_plus_log_counts_distance":
        cols = [*current_numeric, "log_total_poi"]
        for category in POI_CATEGORIES:
            key = safe_feature_name(category)
            cols.extend([f"poi_{key}_log_count", f"poi_{key}_near500_count", f"poi_{key}_inv_distance_sum"])
        return list(dict.fromkeys(cols)), "poi_dominant_category", "当前特征+log计数+近邻数+距离衰减"
    if profile == "ratios7":
        return ratio_cols(POI_CATEGORIES), None, "只用7大类POI比例"
    if profile == "counts_ratios7":
        cols = ["log_total_poi"]
        for category in POI_CATEGORIES:
            key = safe_feature_name(category)
            cols.extend([f"poi_{key}_log_count", f"poi_{key}_ratio"])
        return cols, None, "7大类log计数+比例"
    if profile == "corr_fdr_ratios":
        categories = significant[class_name]
        return (ratio_cols(categories) if categories else ["poi_dummy_zero"]), None, "仅使用FDR显著相关POI比例；无显著项则为零基准"
    if profile == "corr_top3_ratios":
        return ratio_cols(top3[class_name]), None, "每类使用Spearman绝对相关Top3的POI比例"
    if profile == "corr_fdr_enhanced":
        categories = significant[class_name]
        if not categories:
            return ["poi_dummy_zero"], None, "FDR显著POI的比例+log计数+距离特征；无显著项则为零基准"
        return ["log_total_poi", "poi_diversity_entropy", *enhanced_cols(categories)], None, "FDR显著POI的比例+log计数+距离特征"
    if profile == "enhanced7":
        cols = [
            "log_total_poi",
            "poi_diversity_entropy",
            "poi_commercial_intensity",
            "poi_life_service_intensity",
            "poi_corporate_intensity",
            *enhanced_cols(POI_CATEGORIES),
        ]
        return cols, None, "7大类增强特征: 比例+log计数+500m近邻+500-1000m环带+距离衰减"
    raise ValueError(f"unknown profile: {profile}")


def clean_method_name(method: str) -> str:
    return method.split("_", 1)[1].replace("_", " ")


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = predictions.copy()
    predictions["Method_Clean"] = predictions["Method"].map(clean_method_name)
    for (profile, class_name, method), group in predictions.groupby(["Feature_Profile", "Class", "Method_Clean"], sort=False):
        y = group["y_true"].astype(int).to_numpy()
        yp = group["y_pred"].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        rows.append(
            {
                "Feature_Profile": profile,
                "Class": class_name,
                "Method": method,
                "N": len(group),
                "Positive": int(y.sum()),
                "Precision": precision_score(y, yp, zero_division=0),
                "Recall": recall_score(y, yp, zero_division=0),
                "F1": f1_score(y, yp, zero_division=0),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )
    class_metrics = pd.DataFrame(rows)
    macro = (
        class_metrics.groupby(["Feature_Profile", "Method"], as_index=False)
        .agg(Macro_F1=("F1", "mean"), Macro_Precision=("Precision", "mean"), Macro_Recall=("Recall", "mean"))
    )
    return class_metrics, macro


def interpretive_rf_importance(frame: pd.DataFrame, profile: str, significant: dict[str, list[str]], top3: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for class_name in CLASS_NAMES:
        original_y = frame[class_name].astype(int).to_numpy()
        y = frame[f"{class_name}"].astype(int).to_numpy()
        active = frame[f"{class_name}_excluded"].astype(int).eq(0).to_numpy()
        if np.unique(y[active]).size < 2:
            continue
        poi_numeric, poi_cat, _ = profile_features(profile, class_name, significant, top3)
        if poi_cat:
            # Keep this interpretive table numeric-only to avoid expanding one-hot columns.
            poi_numeric = [c for c in poi_numeric if c != poi_cat]
        vision_numeric = [f"{class_name}_{suffix}" for suffix in VISION_SUFFIXES]
        feature_cols = [*vision_numeric, *poi_numeric]
        x = frame.loc[active, feature_cols].copy()
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        model.fit(x, y[active])
        for feature, importance in zip(feature_cols, model.feature_importances_):
            rows.append(
                {
                    "Feature_Profile": profile,
                    "Class": class_name,
                    "Feature": feature,
                    "Source": "POI" if feature in poi_numeric else "Vision",
                    "Interpretive_RF_Importance": float(importance),
                    "Original_Positive": int(original_y.sum()),
                    "Ground truth_Positive_Active": int(y[active].sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["Class", "Interpretive_RF_Importance"], ascending=[True, False])


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore 7-category POI feature groups for ground_truth fusion.")
    parser.add_argument("--model-name", default="yolov9_1280_balanced")
    parser.add_argument("--poi-radius", type=int, default=1000)
    parser.add_argument("--gt-version", default="ground_truth", choices=["original", "ground_truth", "add_only", "gt1_no_evidence_exclude_only", "add_plus_gt1_no_evidence_exclude"])
    parser.add_argument("--profiles", nargs="+", default=["current", "ratios7", "counts_ratios7", "corr_fdr_ratios", "corr_top3_ratios", "corr_fdr_enhanced", "enhanced7"])
    parser.add_argument("--classes", nargs="+", choices=CLASS_NAMES, default=CLASS_NAMES)
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--search-jobs", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"poi_feature_exploration_{args.model_name}_{args.poi_radius}m_{args.gt_version}_{timestamp}"
    out_dir = PROJECT_ROOT / "new_experiment" / "fusion_evaluation" / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    evidence, significant, top3 = load_correlation_evidence()
    frame = load_extended_feature_frame(args.model_name, args.poi_radius)

    feature_rows = []
    all_folds, all_predictions, all_diagnostics = [], [], []
    for profile in args.profiles:
        for class_name in args.classes:
            poi_numeric, poi_categorical, description = profile_features(profile, class_name, significant, top3)
            missing = [col for col in poi_numeric + ([poi_categorical] if poi_categorical else []) if col not in frame.columns]
            if missing:
                raise ValueError(f"{profile}/{class_name} missing columns: {missing}")
            feature_rows.append(
                {
                    "Feature_Profile": profile,
                    "Class": class_name,
                    "Description": description,
                    "POI_Numeric_Features": ", ".join(poi_numeric),
                    "POI_Categorical_Feature": poi_categorical or "",
                    "Significant_POI_Categories": ", ".join(significant[class_name]),
                    "Spearman_Top3_POI_Categories": ", ".join(top3[class_name]),
                    "POI_Feature_Count": len(poi_numeric) + int(bool(poi_categorical)),
                }
            )
            print(f"[{profile}] {class_name}: {len(poi_numeric) + int(bool(poi_categorical))} POI features", flush=True)
            folds, predictions, diagnostics, _ = run_class(
                frame,
                class_name,
                args.gt_version,
                poi_numeric,
                poi_categorical,
                args.quick,
                0,
                args.model_name,
                args.poi_radius,
                args.search_jobs,
            )
            for row in folds:
                row["Feature_Profile"] = profile
            for row in predictions:
                row["Feature_Profile"] = profile
            for row in diagnostics:
                row["Feature_Profile"] = profile
            all_folds.extend(folds)
            all_predictions.extend(predictions)
            all_diagnostics.extend(diagnostics)

    folds_df = pd.DataFrame(all_folds)
    pred_df = pd.DataFrame(all_predictions)
    diag_df = pd.DataFrame(all_diagnostics)
    feature_plan = pd.DataFrame(feature_rows)
    class_metrics, macro = summarize_predictions(pred_df)

    macro_wide = macro.pivot(index="Feature_Profile", columns="Method", values="Macro_F1")
    for col in ["Baseline", "Vision Only", "POI Only", "Full Fusion"]:
        if col not in macro_wide:
            macro_wide[col] = np.nan
    macro_wide = macro_wide[["Baseline", "Vision Only", "POI Only", "Full Fusion"]]
    macro_wide["Full_minus_Vision"] = macro_wide["Full Fusion"] - macro_wide["Vision Only"]
    macro_wide["Full_minus_Current_Full"] = macro_wide["Full Fusion"] - float(macro_wide.loc["current", "Full Fusion"])
    macro_wide["POIOnly_minus_Current_POIOnly"] = macro_wide["POI Only"] - float(macro_wide.loc["current", "POI Only"])
    macro_wide["Best_Method"] = macro_wide[["Baseline", "Vision Only", "POI Only", "Full Fusion"]].idxmax(axis=1)

    full_scores = macro_wide.sort_values(["Full Fusion", "Full_minus_Vision", "POI Only"], ascending=False)
    recommended_profile = str(full_scores.index[0])
    importance = interpretive_rf_importance(frame, recommended_profile, significant, top3)
    poi_importance = importance[importance["Source"].eq("POI")].copy()

    summary_note = pd.DataFrame(
        [
            {
                "Recommended_Profile": recommended_profile,
                "Reason": "按快速社区CV的Full Fusion Macro-F1优先，其次看Full-Vision增量和POI Only表现；正式论文结果仍需用非quick嵌套CV复跑。",
                "Model": args.model_name,
                "POI_Radius_m": args.poi_radius,
                "GT_Version": args.gt_version,
            }
        ]
    )

    pred_df.to_csv(out_dir / "poi_feature_group_oof_predictions.csv.gz", index=False, encoding="utf-8-sig")
    folds_df.to_csv(out_dir / "poi_feature_group_fold_metrics.csv", index=False, encoding="utf-8-sig")
    diag_df.to_csv(out_dir / "poi_feature_group_diagnostics.csv", index=False, encoding="utf-8-sig")

    workbook = out_dir / "POI特征组探索结果.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary_note.to_excel(writer, sheet_name="推荐结论", index=False)
        macro_wide.round(6).to_excel(writer, sheet_name="宏平均F1_特征组")
        macro.round(6).to_excel(writer, sheet_name="宏平均长表", index=False)
        class_metrics.round(6).to_excel(writer, sheet_name="逐类指标", index=False)
        feature_plan.to_excel(writer, sheet_name="特征组定义", index=False)
        evidence.round(6).to_excel(writer, sheet_name="Pearson_Spearman相关性", index=False)
        poi_importance.round(8).to_excel(writer, sheet_name="推荐组POI重要性", index=False)
        importance.round(8).to_excel(writer, sheet_name="推荐组全部重要性", index=False)
        diag_df.to_excel(writer, sheet_name="诊断日志", index=False)

    config = {
        "run_name": run_name,
        "model_name": args.model_name,
        "poi_radius": args.poi_radius,
        "gt_version": args.gt_version,
        "profiles": args.profiles,
        "classes": args.classes,
        "quick": args.quick,
        "search_jobs": args.search_jobs,
        "recommended_profile": recommended_profile,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出目录: {out_dir}", flush=True)
    print(f"推荐特征组: {recommended_profile}", flush=True)
    print(macro_wide.round(6).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

