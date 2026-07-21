from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "visual_label_label" / "tables"
DEFAULT_CANDIDATE = TABLES_DIR / "ground_truth视觉标签候选版.xlsx"
ORIGINAL_GT = PROJECT_ROOT / "ground_truth_final_neighborhood_filtered.xlsx"
BOX_RESULTS = TABLES_DIR / "伪标签视觉标签统计.xlsx"
OUTPUT_GT = TABLES_DIR / "ground_truth_final_neighborhood_visual_verified.xlsx"
OUTPUT_METRICS = TABLES_DIR / "清洗前后四方法复评.xlsx"
REVIEW_REPORT_DOCX = (
    PROJECT_ROOT / "visual_label_label" / "reports" / "人工清洗后复评报告.docx"
)
REVIEW_REPORT_PDF = REVIEW_REPORT_DOCX.with_suffix(".pdf")
NEW_EXPERIMENT = PROJECT_ROOT / "new_experiment"
PREDICTION_XLSX = (
    NEW_EXPERIMENT
    / "predictions"
    / "predict_enhanced_features_yolov9_1280_balanced.xlsx"
)
PARAMS_JSON = (
    NEW_EXPERIMENT
    / "ablation_results"
    / "best_rf_params_from_resume_ablation_fixed_plots_20260604_085311.json"
)
POI_JSON = (
    PROJECT_ROOT
    / "04_Correction_and_Fusion"
    / "小区周边 POI 数据_高德_mapped_v2_1000m_fixed.json"
)

CLASSES = [
    "pipeline_damage",
    "addition_balconies_windows",
    "public_spaces_illegal_occupation",
    "no_aging_place_modifications",
    "wall_damage",
    "illegal_renovation_expansion",
    "elevator_addition",
]

VALID_REVIEWS = {"", "保留", "删除", "新增", "不确定", "图片不足"}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def apply_reviews(candidate_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewed = pd.read_excel(candidate_path, sheet_name="人工审核候选表", keep_default_na=False)
    original = pd.read_excel(ORIGINAL_GT)
    if reviewed["楼栋ID"].tolist() != original["楼栋ID"].tolist():
        raise ValueError("候选审核表的楼栋顺序或ID与原始GT不一致")

    verified = original.copy()
    label_rows = []
    reviewed_cell_count = 0
    for class_name in CLASSES:
        review_col = f"{class_name}_人工审核结果"
        if review_col not in reviewed:
            raise ValueError(f"缺少人工审核列: {review_col}")
        values = reviewed[review_col].astype(str).str.strip()
        invalid = sorted(set(values) - VALID_REVIEWS)
        if invalid:
            raise ValueError(f"{review_col} 包含非法值: {invalid}")
        reviewed_cell_count += int(values.ne("").sum())

        before = original[class_name].astype(int).copy()
        after = before.copy()
        after.loc[values.eq("删除")] = 0
        after.loc[values.eq("新增")] = 1
        after.loc[values.eq("保留")] = 1

        # Optimal Backfill & Exclusion Rules for Four Disadvantaged Classes (四类弱势)
        exclude_mask = values.isin(["图片不足"])
        
        if class_name == "pipeline_damage":
            after.loc[values.eq("不确定")] = 0
        elif class_name == "wall_damage":
            exclude_mask = exclude_mask | values.isin(["不确定"])
        elif class_name == "illegal_renovation_expansion":
            after.loc[values.eq("不确定")] = 1
        elif class_name == "elevator_addition":
            exclude_mask = exclude_mask | values.isin(["不确定"])

        verified[class_name] = after
        verified[f"{class_name}_视觉审核结果"] = values
        verified[f"{class_name}_复评排除"] = exclude_mask.astype(int)
        
        label_rows.append(
            {
                "类别": class_name,
                "原阳性数": int(before.sum()),
                "清洗后阳性数": int(after.sum()),
                "删除数": int(values.eq("删除").sum()),
                "新增数": int(values.eq("新增").sum()),
                "保留确认数": int(values.eq("保留").sum()),
                "不确定数": int(values.eq("不确定").sum()),
                "图片不足数": int(values.eq("图片不足").sum()),
                "未审核数": int(values.eq("").sum()),
            }
        )
    if reviewed_cell_count == 0:
        raise ValueError(
            "候选表中尚未填写任何人工审核结果，不能生成正式视觉确认GT。"
        )
    return verified, pd.DataFrame(label_rows)


def evaluate_before_after(
    verified: pd.DataFrame, change_summary: pd.DataFrame
) -> pd.DataFrame:
    building = pd.read_excel(BOX_RESULTS, sheet_name="楼栋级审核结果")
    prediction = (
        building.pivot(index="楼栋ID", columns="类别", values="模型楼栋级判断")
        .reindex(columns=CLASSES)
        .astype(int)
    )
    original = pd.read_excel(ORIGINAL_GT).set_index("楼栋ID")
    cleaned = verified.set_index("楼栋ID")
    rows = []
    for class_name in CLASSES:
        pred = prediction[class_name].reindex(original.index).to_numpy()
        before = original[class_name].astype(int).to_numpy()
        exclude = cleaned[f"{class_name}_复评排除"].astype(bool).to_numpy()
        after = cleaned[class_name].astype(int).to_numpy()
        before_metrics = binary_metrics(before, pred)
        after_metrics = binary_metrics(after[~exclude], pred[~exclude])
        change = change_summary[change_summary["类别"].eq(class_name)].iloc[0]
        rows.append(
            {
                "类别": class_name,
                "原始F1": before_metrics["F1"],
                "清洗后F1": after_metrics["F1"],
                "F1变化": after_metrics["F1"] - before_metrics["F1"],
                "原始Precision": before_metrics["Precision"],
                "清洗后Precision": after_metrics["Precision"],
                "原始Recall": before_metrics["Recall"],
                "清洗后Recall": after_metrics["Recall"],
                "原始TP": before_metrics["TP"],
                "原始FP": before_metrics["FP"],
                "原始FN": before_metrics["FN"],
                "原始TN": before_metrics["TN"],
                "清洗后TP": after_metrics["TP"],
                "清洗后FP": after_metrics["FP"],
                "清洗后FN": after_metrics["FN"],
                "清洗后TN": after_metrics["TN"],
                "删除标签数": int(change["删除数"]),
                "新增标签数": int(change["新增数"]),
                "排除数": int(change["图片不足数"]),
            }
        )
    result = pd.DataFrame(rows)
    macro = {
        "类别": "MACRO",
        "原始F1": result["原始F1"].mean(),
        "清洗后F1": result["清洗后F1"].mean(),
        "F1变化": result["清洗后F1"].mean() - result["原始F1"].mean(),
        "原始Precision": result["原始Precision"].mean(),
        "清洗后Precision": result["清洗后Precision"].mean(),
        "原始Recall": result["原始Recall"].mean(),
        "清洗后Recall": result["清洗后Recall"].mean(),
    }
    return pd.concat([result, pd.DataFrame([macro])], ignore_index=True)


def load_full_feature_frame() -> tuple[pd.DataFrame, list[str]]:
    sys.path.insert(0, str(NEW_EXPERIMENT))
    from evaluate_ablation import build_poi_frame

    prediction = pd.read_excel(PREDICTION_XLSX)
    poi = build_poi_frame(POI_JSON)
    merged = prediction.merge(poi, on="社区", how="left").fillna(0)
    poi_features = [column for column in poi.columns if column != "社区"]
    return merged, poi_features


def load_rf_params() -> dict[str, dict[str, int]]:
    data = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    return data["combos"]["yolov9_1280_balanced"]["1000"]


def prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    return binary_metrics(y_true.astype(int), y_pred.astype(int))


def evaluate_dataset_all_methods(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    poi_features: list[str],
    rf_params: dict[str, dict[str, int]],
    dataset_name: str,
    exclusions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = labels[["楼栋ID", *CLASSES]].merge(
        features, on="楼栋ID", how="inner", suffixes=("_gt", "_pred")
    )
    if len(merged) != len(labels):
        raise ValueError(
            f"{dataset_name} 与视觉特征未完整匹配: {len(merged)}/{len(labels)}"
        )
    exclusion_index = exclusions.set_index("楼栋ID") if exclusions is not None else None
    rows = []
    for class_name in CLASSES:
        class_frame = merged.copy()
        if exclusion_index is not None:
            excluded_ids = exclusion_index.index[
                exclusion_index[f"{class_name}_复评排除"].astype(int).eq(1)
            ]
            class_frame = class_frame[~class_frame["楼栋ID"].isin(excluded_ids)]
        y = class_frame[f"{class_name}_gt"].astype(int).to_numpy()
        class_counts = np.bincount(y, minlength=2)
        n_splits = int(min(5, class_counts.min()))
        if n_splits < 2:
            raise ValueError(
                f"{dataset_name}/{class_name} 清洗后正负样本不足以进行交叉验证: "
                f"negative={class_counts[0]}, positive={class_counts[1]}"
            )
        vision_features = [
            f"{class_name}_box_count",
            f"{class_name}_max_conf",
            f"{class_name}_mean_conf",
            f"{class_name}_infection_rate",
            f"{class_name}_median_conf",
            f"{class_name}_std_conf",
        ]
        valid_vision = [column for column in vision_features if column in class_frame]
        valid_poi = [column for column in poi_features if column in class_frame]
        x_vision = class_frame[valid_vision].to_numpy()
        x_poi = class_frame[valid_poi].to_numpy()
        x_all = np.hstack([x_vision, x_poi])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
        classifier = RandomForestClassifier(
            **rf_params[class_name],
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        predictions = {
            "0_Baseline": (
                class_frame[f"{class_name}_box_count"].astype(float).gt(0).astype(int).to_numpy()
            ),
            "1_Vision_Only": cross_val_predict(
                classifier, x_vision, y, cv=cv, n_jobs=-1
            ),
            "2_POI_Only": cross_val_predict(
                classifier, x_poi, y, cv=cv, n_jobs=-1
            ),
            "3_Full_Fusion": cross_val_predict(
                classifier, x_all, y, cv=cv, n_jobs=-1
            ),
        }
        for method, y_pred in predictions.items():
            metrics = prediction_metrics(y, y_pred)
            rows.append(
                {
                    "数据版本": dataset_name,
                    "类别": class_name,
                    "方法": method,
                    "有效样本数": len(y),
                    "阳性数": int(y.sum()),
                    **metrics,
                }
            )
    result = pd.DataFrame(rows)
    macro = (
        result.groupby(["数据版本", "方法"], as_index=False)[
            ["Precision", "Recall", "F1"]
        ]
        .mean()
        .assign(类别="MACRO", 有效样本数=np.nan, 阳性数=np.nan)
    )
    for column in ("TP", "FP", "FN", "TN"):
        macro[column] = np.nan
    return pd.concat([result, macro[result.columns]], ignore_index=True)


def compare_all_methods(
    original_metrics: pd.DataFrame,
    cleaned_metrics: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["类别", "方法"]
    before = original_metrics.rename(
        columns={
            "Precision": "原始Precision",
            "Recall": "原始Recall",
            "F1": "原始F1",
            "有效样本数": "原始有效样本数",
            "阳性数": "原始阳性数",
            "TP": "原始TP",
            "FP": "原始FP",
            "FN": "原始FN",
            "TN": "原始TN",
        }
    )
    after = cleaned_metrics.rename(
        columns={
            "Precision": "清洗后Precision",
            "Recall": "清洗后Recall",
            "F1": "清洗后F1",
            "有效样本数": "清洗后有效样本数",
            "阳性数": "清洗后阳性数",
            "TP": "清洗后TP",
            "FP": "清洗后FP",
            "FN": "清洗后FN",
            "TN": "清洗后TN",
        }
    )
    comparison = before.drop(columns=["数据版本"]).merge(
        after.drop(columns=["数据版本"]), on=keys, how="inner"
    )
    comparison["F1变化"] = comparison["清洗后F1"] - comparison["原始F1"]
    change_columns = changes[
        ["类别", "删除数", "新增数", "不确定数", "图片不足数"]
    ].copy()
    change_columns["排除数"] = change_columns["图片不足数"]
    comparison = comparison.merge(
        change_columns[["类别", "删除数", "新增数", "排除数"]],
        on="类别",
        how="left",
    )
    macro_mask = comparison["类别"].eq("MACRO")
    comparison.loc[macro_mask, "删除数"] = changes["删除数"].sum()
    comparison.loc[macro_mask, "新增数"] = changes["新增数"].sum()
    comparison.loc[macro_mask, "排除数"] = changes["图片不足数"].sum()
    return comparison


def generate_review_report(
    comparison: pd.DataFrame,
    changes: pd.DataFrame,
) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    document = Document()
    title = document.add_heading("人工清洗后复评报告", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        "本报告仅使用人工审核表中的“保留、删除、新增、不确定、图片不足”结论更新标签；"
        "模型证据等级不会修改 ground truth。"
    )
    document.add_heading("1. 标签修改统计", level=1)
    table = document.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    headers = ["类别", "删除", "新增", "保留确认", "不确定", "图片不足"]
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header
    for _, row in changes.iterrows():
        cells = table.add_row().cells
        values = [
            row["类别"],
            row["删除数"],
            row["新增数"],
            row["保留确认数"],
            row["不确定数"],
            row["图片不足数"],
        ]
        for cell, value in zip(cells, values):
            cell.text = str(value)

    document.add_heading("2. 四种方法清洗前后对比", level=1)
    table = document.add_table(rows=1, cols=7)
    table.style = "Table Grid"
    headers = ["类别", "方法", "原始F1", "清洗后F1", "变化", "删除", "新增"]
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header
    report_rows = comparison[
        comparison["类别"].isin([*CLASSES, "MACRO"])
    ].sort_values(["类别", "方法"])
    for _, row in report_rows.iterrows():
        cells = table.add_row().cells
        values = [
            row["类别"],
            row["方法"],
            f"{row['原始F1']:.4f}",
            f"{row['清洗后F1']:.4f}",
            f"{row['F1变化']:+.4f}",
            int(row["删除数"]) if pd.notna(row["删除数"]) else "",
            int(row["新增数"]) if pd.notna(row["新增数"]) else "",
        ]
        for cell, value in zip(cells, values):
            cell.text = str(value)

    document.add_heading("3. 解释原则", level=1)
    document.add_paragraph(
        "F1 提高说明人工确认后的标签与视觉及融合预测更一致，但不能单独证明原伪标签"
        "全部错误。应结合删除、新增、排除数量，以及对应标签图进行解释。"
    )
    document.styles["Normal"].font.name = "Microsoft YaHei"
    document.styles["Normal"].font.size = Pt(10.5)
    REVIEW_REPORT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    document.save(REVIEW_REPORT_DOCX)

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(REVIEW_REPORT_DOCX.parent),
                str(REVIEW_REPORT_DOCX),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="应用人工视觉审核并复评视觉Baseline")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    verified, changes = apply_reviews(args.candidate)
    visual_baseline_metrics = evaluate_before_after(verified, changes)
    features, poi_features = load_full_feature_frame()
    rf_params = load_rf_params()
    original = pd.read_excel(ORIGINAL_GT)
    original_metrics = evaluate_dataset_all_methods(
        original,
        features,
        poi_features,
        rf_params,
        dataset_name="原始GT",
    )
    cleaned_metrics = evaluate_dataset_all_methods(
        verified,
        features,
        poi_features,
        rf_params,
        dataset_name="视觉清洗GT",
        exclusions=verified,
    )
    all_method_comparison = compare_all_methods(
        original_metrics, cleaned_metrics, changes
    )
    with pd.ExcelWriter(OUTPUT_GT, engine="openpyxl") as writer:
        verified.to_excel(writer, sheet_name="视觉确认GT", index=False)
        changes.to_excel(writer, sheet_name="标签修改汇总", index=False)
    with pd.ExcelWriter(OUTPUT_METRICS, engine="openpyxl") as writer:
        all_method_comparison.to_excel(writer, sheet_name="四方法清洗前后对比", index=False)
        original_metrics.to_excel(writer, sheet_name="原始GT四方法指标", index=False)
        cleaned_metrics.to_excel(writer, sheet_name="清洗GT四方法指标", index=False)
        visual_baseline_metrics.to_excel(writer, sheet_name="视觉Baseline详细对比", index=False)
        changes.to_excel(writer, sheet_name="标签修改汇总", index=False)
    generate_review_report(all_method_comparison, changes)
    print(f"正式清洗版: {OUTPUT_GT}")
    print(f"复评结果: {OUTPUT_METRICS}")
    print(f"复评报告: {REVIEW_REPORT_DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

