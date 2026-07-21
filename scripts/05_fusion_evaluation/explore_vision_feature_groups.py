from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import evaluate_ablation_nested as nested
from fusion_utils import FUSION_RESULTS, ensure_fusion_dirs
from experiment_registry import CLASS_NAMES


VISION_PROFILES: dict[str, list[str]] = {
    "all6": ["box_count", "max_conf", "mean_conf", "infection_rate", "median_conf", "std_conf"],
    "core4": ["box_count", "max_conf", "mean_conf", "infection_rate"],
    "no_box_count": ["max_conf", "mean_conf", "infection_rate", "median_conf", "std_conf"],
    "no_infection_rate": ["box_count", "max_conf", "mean_conf", "median_conf", "std_conf"],
    "no_std": ["box_count", "max_conf", "mean_conf", "infection_rate", "median_conf"],
    "no_median": ["box_count", "max_conf", "mean_conf", "infection_rate", "std_conf"],
    "no_mean": ["box_count", "max_conf", "infection_rate", "median_conf", "std_conf"],
    "no_max": ["box_count", "mean_conf", "infection_rate", "median_conf", "std_conf"],
    "count_rate_conf": ["box_count", "max_conf", "infection_rate"],
    "count_rate": ["box_count", "infection_rate"],
    "conf_stats": ["max_conf", "mean_conf", "median_conf", "std_conf"],
    "conf_core": ["max_conf", "mean_conf", "median_conf"],
    "count_only": ["box_count"],
    "max_only": ["max_conf"],
    "infection_only": ["infection_rate"],
}


def clean_method_name(method: str) -> str:
    return method.split("_", 1)[1].replace("_", " ")


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (profile, class_name, method), group in predictions.groupby(["Vision_Profile", "Class", "Method"]):
        y = group["y_true"].astype(int)
        pred = group["y_pred"].astype(int)
        rows.append(
            {
                "Vision_Profile": profile,
                "Class": class_name,
                "Method": clean_method_name(method),
                "N": len(group),
                "Positive": int(y.sum()),
                "Precision": precision_score(y, pred, zero_division=0),
                "Recall": recall_score(y, pred, zero_division=0),
                "F1": f1_score(y, pred, zero_division=0),
            }
        )
    detail = pd.DataFrame(rows)
    macro = (
        detail.groupby(["Vision_Profile", "Method"], as_index=False)
        .agg(
            Macro_Precision=("Precision", "mean"),
            Macro_Recall=("Recall", "mean"),
            Macro_F1=("F1", "mean"),
        )
    )
    wide = macro.pivot(index="Vision_Profile", columns="Method", values="Macro_F1").reset_index()
    if "Full Fusion" in wide and "Vision Only" in wide:
        wide["Full_minus_Vision"] = wide["Full Fusion"] - wide["Vision Only"]
    if "Vision Only" in wide:
        wide = wide.sort_values(["Vision Only", "Full Fusion"], ascending=False)
    return detail, wide


def run_exploration(args: argparse.Namespace) -> Path:
    ensure_fusion_dirs()
    run_name = args.run_name or f"vision_feature_ablation_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output = FUSION_RESULTS / run_name
    output.mkdir(parents=True, exist_ok=False)

    profiles = args.profiles or list(VISION_PROFILES)
    unknown = sorted(set(profiles) - set(VISION_PROFILES))
    if unknown:
        raise ValueError(f"未知视觉特征组: {unknown}")

    frame, poi_numeric, poi_categorical = nested.load_feature_frame(args.model_name, args.poi_radius)
    poi_numeric, poi_categorical = nested.select_poi_feature_profile(args.poi_feature_profile, poi_numeric, poi_categorical)

    original_suffixes = list(nested.VISION_SUFFIXES)
    all_folds, all_predictions, all_diagnostics = [], [], []
    try:
        for profile in profiles:
            suffixes = VISION_PROFILES[profile]
            nested.VISION_SUFFIXES = suffixes
            print(f"=== Vision profile: {profile} ({', '.join(suffixes)}) ===", flush=True)
            for class_name in args.classes:
                print(f"[ground_truth] {class_name}", flush=True)
                folds, predictions, diagnostics, _ = nested.run_class(
                    frame=frame,
                    class_name=class_name,
                    gt_version="ground_truth",
                    poi_numeric=poi_numeric,
                    poi_categorical=poi_categorical,
                    quick=args.quick,
                    permutations=0,
                    model_name=args.model_name,
                    radius=args.poi_radius,
                    search_jobs=args.search_jobs,
                    poi_profile=args.poi_feature_profile,
                    fusion_policy=args.fusion_policy,
                )
                for row in folds:
                    row["Vision_Profile"] = profile
                    row["Vision_Suffixes"] = ",".join(suffixes)
                for row in predictions:
                    row["Vision_Profile"] = profile
                    row["Vision_Suffixes"] = ",".join(suffixes)
                for row in diagnostics:
                    row["Vision_Profile"] = profile
                    row["Vision_Suffixes"] = ",".join(suffixes)
                all_folds.extend(folds)
                all_predictions.extend(predictions)
                all_diagnostics.extend(diagnostics)
    finally:
        nested.VISION_SUFFIXES = original_suffixes

    folds_frame = pd.DataFrame(all_folds)
    predictions_frame = pd.DataFrame(all_predictions)
    diagnostics_frame = pd.DataFrame(all_diagnostics)
    detail, ranking = summarize_predictions(predictions_frame)

    folds_frame.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(output / "building_oof_predictions.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    diagnostics_frame.to_csv(output / "diagnostics.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output / "视觉特征组消融结果.xlsx", engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="特征组排名", index=False)
        detail.to_excel(writer, sheet_name="逐类指标", index=False)
        pd.DataFrame(
            [
                {
                    "Vision_Profile": profile,
                    "Vision_Suffixes": ",".join(VISION_PROFILES[profile]),
                    "Feature_Count": len(VISION_PROFILES[profile]),
                }
                for profile in profiles
            ]
        ).to_excel(writer, sheet_name="特征组定义", index=False)
        diagnostics_frame.to_excel(writer, sheet_name="诊断日志", index=False)

    (output / "run_config.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "model_name": args.model_name,
                "poi_radius": args.poi_radius,
                "poi_feature_profile": args.poi_feature_profile,
                "fusion_policy": args.fusion_policy,
                "quick": args.quick,
                "profiles": profiles,
                "classes": args.classes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果目录: {output}")
    print(ranking.to_string(index=False))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore YOLO visual aggregate feature groups.")
    parser.add_argument("--model-name", default="yolov9_1280_balanced")
    parser.add_argument("--poi-radius", type=int, default=1000)
    parser.add_argument("--poi-feature-profile", default="current_no_missing")
    parser.add_argument("--fusion-policy", default="guarded")
    parser.add_argument("--classes", nargs="+", choices=CLASS_NAMES, default=CLASS_NAMES)
    parser.add_argument("--profiles", nargs="+", choices=sorted(VISION_PROFILES), default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--search-jobs", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_exploration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

