from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from fusion_utils import FUSION_RESULTS
from evaluate_ablation_nested import labelED_GT
from experiment_registry import CLASS_NAMES


BASE = Path(r"D:\人工智能数据\complete")
NEW_EXP = BASE / "new_experiment"
POI_ROOT = BASE / "04_Correction_and_Fusion"
FORMAL_RUN = FUSION_RESULTS / "formal_yolov9_1280_1000m_current_no_missing_final"
OUT_DIR = FUSION_RESULTS / "poi_improvement_diagnosis"
OUT_XLSX = OUT_DIR / "POI提升空间诊断.xlsx"


def load_old_ablation_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((NEW_EXP / "ablation_results").glob("ablation_*_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"Model", "POI_Radius_m", "Method", "Class"}
        if not required.issubset(frame.columns):
            continue
        f1_col = "F1-Score" if "F1-Score" in frame.columns else "F1"
        methods = {
            "Baseline": frame["Method"].astype(str).str.contains("Baseline", na=False),
            "Vision Only": frame["Method"].astype(str).str.contains("Vision", na=False),
            "POI Only": frame["Method"].astype(str).str.contains("POI", na=False),
            "Full Fusion": frame["Method"].astype(str).str.contains("Full", na=False),
        }
        row = {
            "文件": path.name,
            "Model": frame["Model"].iloc[0],
            "POI_Radius_m": int(frame["POI_Radius_m"].iloc[0]),
        }
        for name, mask in methods.items():
            row[name] = frame.loc[mask, f1_col].mean()
        row["Full_minus_Vision"] = row["Full Fusion"] - row["Vision Only"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Full Fusion", ascending=False)


def load_formal_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = FORMAL_RUN / "无泄漏消融与POI贡献汇总.xlsx"
    metrics = pd.read_excel(workbook, sheet_name="总体指标")
    ranking = pd.read_excel(workbook, sheet_name="Full排名")
    bootstrap = pd.read_excel(workbook, sheet_name="社区Bootstrap")
    return metrics, ranking, bootstrap


def oracle_poi_potential() -> pd.DataFrame:
    predictions = pd.read_csv(FORMAL_RUN / "building_oof_predictions.csv.gz")
    predictions = predictions.loc[predictions["GT_Version"].eq("ground_truth")].copy()
    rows = []
    for class_name, group in predictions.groupby("Class"):
        pivot = group.pivot_table(
            index=["楼栋ID", "社区", "y_true"],
            columns="Method",
            values=["y_pred", "probability"],
            aggfunc="first",
        )
        pivot.columns = [f"{a}|{b}" for a, b in pivot.columns]
        pivot = pivot.reset_index()
        y = pivot["y_true"].astype(int).to_numpy()
        vision = pivot["y_pred|1_Vision_Only"].astype(int).to_numpy()
        poi = pivot["y_pred|2_POI_Only"].astype(int).to_numpy()
        full = pivot["y_pred|3_Full_Fusion"].astype(int).to_numpy()
        vision_fn = (y == 1) & (vision == 0)
        vision_fp = (y == 0) & (vision == 1)
        oracle = np.where((vision == y) | (poi == y), y, vision)
        union = ((vision == 1) | (poi == 1)).astype(int)
        intersection = ((vision == 1) & (poi == 1)).astype(int)
        rows.append(
            {
                "Class": class_name,
                "N": len(y),
                "Positive": int(y.sum()),
                "Vision_F1": f1_score(y, vision, zero_division=0),
                "POI_F1": f1_score(y, poi, zero_division=0),
                "Full_F1": f1_score(y, full, zero_division=0),
                "Vision_FN": int(vision_fn.sum()),
                "Vision_FP": int(vision_fp.sum()),
                "POI_catches_Vision_FN": int((vision_fn & (poi == 1)).sum()),
                "POI_FP_when_Vision_TN": int(((y == 0) & (vision == 0) & (poi == 1)).sum()),
                "Union_F1": f1_score(y, union, zero_division=0),
                "Intersection_F1": f1_score(y, intersection, zero_division=0),
                "Oracle_choose_V_or_POI_F1": f1_score(y, oracle, zero_division=0),
            }
        )
    result = pd.DataFrame(rows)
    result["Oracle_gain_over_Vision"] = result["Oracle_choose_V_or_POI_F1"] - result["Vision_F1"]
    result["Full_gain_over_Vision"] = result["Full_F1"] - result["Vision_F1"]
    return result.sort_values("Oracle_gain_over_Vision", ascending=False)


def detailed_poi_features(radius: int = 1000) -> pd.DataFrame:
    path = POI_ROOT / "小区周边POI数据_高德_mapped_v2_fixed.json"
    if radius == 500:
        path = POI_ROOT / "小区周边 POI 数据_高德_mapped_v2_500m_fixed.json"
    elif radius == 1500:
        path = POI_ROOT / "小区周边POI数据_高德_mapped_v2_1500m.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for community, payload in raw.items():
        total = float(payload.get("total_count", 0) or 0)
        row = {"社区": str(community).strip(), "total_poi": total}
        for poi_type, count in (payload.get("type_distribution") or {}).items():
            safe = str(poi_type).replace("\n", " ").strip()
            value = float(count or 0)
            row[f"type_count::{safe}"] = value
            row[f"type_ratio::{safe}"] = value / total if total > 0 else 0.0
        distance_bins: dict[str, dict[str, float]] = {}
        for poi in payload.get("poi_list", []) or []:
            poi_type = str(poi.get("type", "")).strip()
            if not poi_type:
                continue
            try:
                distance = float(poi.get("distance", 1000) or 1000)
            except Exception:
                distance = 1000.0
            bucket = distance_bins.setdefault(poi_type, {"near300": 0.0, "near500": 0.0, "inv": 0.0})
            bucket["near300"] += float(distance <= 300)
            bucket["near500"] += float(distance <= 500)
            bucket["inv"] += 1.0 / (1.0 + distance / 100.0)
        for poi_type, values in distance_bins.items():
            safe = poi_type.replace("\n", " ").strip()
            for name, value in values.items():
                row[f"type_{name}::{safe}"] = value
        rows.append(row)
    frame = pd.DataFrame(rows).fillna(0.0)
    return frame


def community_label_frame() -> pd.DataFrame:
    ground_truth = pd.read_excel(labelED_GT, sheet_name="GT")
    rows = []
    for class_name in CLASS_NAMES:
        available = ground_truth.loc[ground_truth[f"{class_name}_excluded"].astype(int).eq(0)].copy()
        grouped = (
            available.groupby("社区", as_index=False)
            .agg(
                Positive_Rate=(f"{class_name}", "mean"),
                Any_Positive=(f"{class_name}", "max"),
                N=("楼栋ID", "size"),
                Positive=(f"{class_name}", "sum"),
            )
        )
        grouped["Class"] = class_name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def detailed_type_correlations(top_k: int = 30) -> pd.DataFrame:
    poi = detailed_poi_features(1000)
    labels = community_label_frame()
    feature_columns = [c for c in poi.columns if c != "社区"]
    rows = []
    for class_name, label_group in labels.groupby("Class"):
        merged = label_group.merge(poi, on="社区", how="inner")
        y = merged["Positive_Rate"].astype(float).to_numpy()
        for column in feature_columns:
            x = merged[column].astype(float).to_numpy()
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            rho, p_value = spearmanr(x, y, nan_policy="omit")
            if not np.isfinite(rho):
                continue
            poi_type = column.split("::", 1)[1] if "::" in column else column
            rows.append(
                {
                    "Class": class_name,
                    "Feature": column,
                    "POI_Type": poi_type,
                    "Spearman_Rho": float(rho),
                    "Abs_Rho": abs(float(rho)),
                    "P_Value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "Communities": int(len(merged)),
                    "Nonzero_Communities": int((merged[column] > 0).sum()),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (
        result.sort_values(["Class", "Abs_Rho"], ascending=[True, False])
        .groupby("Class", as_index=False)
        .head(top_k)
        .sort_values(["Class", "Abs_Rho"], ascending=[True, False])
    )


def recommendations(
    old_ablation: pd.DataFrame,
    formal_metrics: pd.DataFrame,
    formal_ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    ground_truth_macro = formal_metrics.loc[
        formal_metrics["GT_Version"].eq("ground_truth") & formal_metrics["Class"].eq("MACRO")
    ][["Method", "F1"]]
    macro_map = dict(zip(ground_truth_macro["Method"], ground_truth_macro["F1"]))
    boot_macro = bootstrap.loc[bootstrap["GT_Version"].eq("ground_truth") & bootstrap["Class"].eq("MACRO")]
    ci = ""
    if not boot_macro.empty:
        row = boot_macro.iloc[0]
        ci = f"{row['CI95_Low']:.4f} ~ {row['CI95_High']:.4f}"
    rows = [
        {
            "优先级": 1,
            "方向": "细粒度POI类型筛选",
            "理由": "1000m JSON有863个高德细分类，正式模型只用了7大类；当前最可能遗漏了与维修、商业、医疗、建材、物业等相关的细类型信号。",
            "建议实验": "在外层训练折内做L1/单变量筛选，只保留每类top 10-30个细分类和距离衰减特征，再跑无泄漏CV。",
            "风险": "社区只有92个，高维特征必须训练折内筛选，不能全量先选后评估。",
        },
        {
            "优先级": 2,
            "方向": "错误门控融合",
            "理由": f"POI与Vision的乐观互补上限Macro-F1约{oracle['Oracle_choose_V_or_POI_F1'].mean():.4f}，高于当前Full {macro_map.get('3_Full_Fusion', np.nan):.4f}。",
            "建议实验": "训练一个只在Vision低置信/近阈值/类别困难样本上启用POI的gating模型，减少POI在Vision真阴性上的误报。",
            "风险": "门控也必须使用OOF概率训练，不能用测试折调门。",
        },
        {
            "优先级": 3,
            "方向": "类别级特征策略",
            "理由": "no_aging_place_modifications显著受益，但public_spaces和elevator被POI拖低；统一特征组不适合全部类别。",
            "建议实验": "每类固定候选profile，由内层CV选择；若Full候选在训练阶段不优，回退Vision Only，但回退规则必须只看训练折。",
            "风险": "如果按最终测试表现挑类别，会构成选择偏差。",
        },
        {
            "优先级": 4,
            "方向": "半径敏感性正式复核",
            "理由": "旧实验里1000m通常优于500/1500m，但GT下还没完整跑500/1500m的current_no_missing正式版。",
            "建议实验": "跑yolov9_1280 + 500/1000/1500mGT，保留同一折清单，比较Full-Vision增量。",
            "风险": "半径不能事后只报最好，需作为敏感性实验呈现。",
        },
        {
            "优先级": 5,
            "方向": "论文表述策略",
            "理由": f"当前GT Full-Vision Macro提升约{macro_map.get('3_Full_Fusion', np.nan)-macro_map.get('1_Vision_Only', np.nan):.4f}，Bootstrap CI为{ci}。",
            "建议实验": "主张POI提供类别相关的上下文补充，而不是全面显著提升。",
            "风险": "若Bootstrap置信区间跨越0，则不宜表述为严格统计显著提升。",
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_ablation = load_old_ablation_summary()
    formal_metrics, formal_ranking, bootstrap = load_formal_metrics()
    oracle = oracle_poi_potential()
    detailed_corr = detailed_type_correlations()
    recs = recommendations(old_ablation, formal_metrics, formal_ranking, bootstrap, oracle)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        recs.to_excel(writer, sheet_name="提升建议", index=False)
        old_ablation.to_excel(writer, sheet_name="旧消融汇总", index=False)
        formal_metrics.to_excel(writer, sheet_name="正式无泄漏指标", index=False)
        formal_ranking.to_excel(writer, sheet_name="Full排名", index=False)
        bootstrap.to_excel(writer, sheet_name="Bootstrap", index=False)
        oracle.to_excel(writer, sheet_name="POI互补上限", index=False)
        detailed_corr.to_excel(writer, sheet_name="细分类POI相关性Top30", index=False)

    print(f"诊断表已保存: {OUT_XLSX}")
    print(recs[["优先级", "方向", "理由"]].to_string(index=False))
    print("\nPOI互补上限:")
    print(
        oracle[
            [
                "Class",
                "Vision_F1",
                "POI_F1",
                "Full_F1",
                "Oracle_choose_V_or_POI_F1",
                "Oracle_gain_over_Vision",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

