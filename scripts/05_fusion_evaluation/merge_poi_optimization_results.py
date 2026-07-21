from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    base = Path(r"D:\人工智能数据\complete\new_experiment\fusion_evaluation\results")
    runs = [
        base / "poi_feature_optimization_batch1_yolov9_1280_1000m",
        base / "poi_feature_optimization_batch2_removal_yolov9_1280_1000m",
        base / "poi_feature_exploration_yolov9_1280_1000m",
    ]
    macro_parts = []
    class_parts = []
    feature_parts = []
    for run in runs:
        workbook = run / "POI特征组探索结果.xlsx"
        if not workbook.exists():
            continue
        macro = pd.read_excel(workbook, sheet_name="宏平均F1_特征组")
        macro["Source_Run"] = run.name
        macro_parts.append(macro)

        class_metrics = pd.read_excel(workbook, sheet_name="逐类指标")
        class_metrics["Source_Run"] = run.name
        class_parts.append(class_metrics)

        features = pd.read_excel(workbook, sheet_name="特征组定义")
        features["Source_Run"] = run.name
        feature_parts.append(features)

    macro_all = pd.concat(macro_parts, ignore_index=True)
    macro_all = (
        macro_all.sort_values(["Feature_Profile", "Full Fusion"], ascending=[True, False])
        .drop_duplicates("Feature_Profile")
        .sort_values(["Full Fusion", "Full_minus_Vision", "POI Only"], ascending=False)
    )
    class_all = pd.concat(class_parts, ignore_index=True)
    feature_all = pd.concat(feature_parts, ignore_index=True).drop_duplicates(["Feature_Profile", "Class"])

    out_dir = base / "poi_feature_optimization_final_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "POI特征优化最终汇总.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        macro_all.round(6).to_excel(writer, sheet_name="全部特征组排名", index=False)
        class_all.round(6).to_excel(writer, sheet_name="逐类指标", index=False)
        feature_all.to_excel(writer, sheet_name="特征组定义", index=False)

    md = out_dir / "POI特征优化最终建议.md"
    md.write_text(
        """# POI特征优化最终建议

第二轮围绕 current 特征做了新增、删减和按类别择优实验。

## 最优结论

快速社区CV下，current_no_missing 的 Full Fusion Macro-F1 最高：

current:            Full Fusion = 0.745562
current_no_missing: Full Fusion = 0.747894  (+0.002332)
classwise_stage1_best: Full Fusion = 0.747890 (+0.002328)
current_core_no_handcrafted: Full Fusion = 0.747804 (+0.002241)

因为 poi_missing_indicator 在当前数据中全为0，没有实际区分信息，推荐最终先采用 current_no_missing：

total_poi
poi_餐饮服务_ratio
poi_生活服务_ratio
poi_购物服务_ratio
poi_公司企业_ratio
poi_医疗保健_ratio
poi_科教文化_ratio
poi_住宿服务_ratio
poi_diversity_entropy
poi_commercial_intensity
poi_dominant_category

## 备选

如果想在论文中减少“手工特征”的质疑，可以使用 current_core_no_handcrafted：

total_poi
poi_餐饮服务_ratio
poi_生活服务_ratio
poi_购物服务_ratio
poi_公司企业_ratio
poi_医疗保健_ratio
poi_科教文化_ratio
poi_住宿服务_ratio

它的 Full Fusion 只比 current_no_missing 低约 0.00009，但 POI Only 更高，解释更干净。

## 注意

这些是 quick 社区CV筛选结果。若作为最终论文主结果，应把推荐特征组放入正式非quick嵌套CV重新跑一次。
""",
        encoding="utf-8",
    )

    print("BEST MACRO")
    print(
        macro_all[
            [
                "Feature_Profile",
                "Baseline",
                "Vision Only",
                "POI Only",
                "Full Fusion",
                "Full_minus_Vision",
                "Full_minus_Current_Full",
                "Best_Method",
                "Source_Run",
            ]
        ]
        .head(12)
        .round(6)
        .to_string(index=False)
    )
    print(f"WROTE {out}")
    print(f"WROTE {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

