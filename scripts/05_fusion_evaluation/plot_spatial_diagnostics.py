from __future__ import annotations

import argparse

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ceus_experiment_utils import ensure_ceus_extra_dir, load_feature_table
from experiment_registry import CLASS_NAMES, add_model_name_argument


GROUP_LABEL_EN = {
    "区市": "district",
    "街道": "street",
    "社区": "community",
}


def plot_group_prevalence(model_name: str, poi_radius: int, group_col: str) -> str:
    _, _, df, _ = load_feature_table(model_name, poi_radius)
    if group_col not in df.columns:
        raise ValueError(f"group-col 不存在: {group_col}")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = []
    for group, part in df.groupby(group_col):
        row = {"Group": group, "Building_Count": len(part)}
        for class_name in CLASS_NAMES:
            gt_col = f"{class_name}_gt"
            pred_col = f"{class_name}_box_count"
            if gt_col in part.columns:
                row[f"{class_name}_gt_rate"] = float(part[gt_col].mean())
            if pred_col in part.columns:
                row[f"{class_name}_pred_rate"] = float((part[pred_col] > 0).mean())
        rows.append(row)

    out_dir = ensure_ceus_extra_dir()
    csv_path = out_dir / f"spatial_diagnostics_{model_name}_{poi_radius}m_{group_col}.csv"
    plot_path = out_dir / f"spatial_diagnostics_{model_name}_{poi_radius}m_{group_col}.png"
    label_map_path = out_dir / f"spatial_diagnostics_{model_name}_{poi_radius}m_{group_col}_label_map.csv"
    result = pd.DataFrame(rows).sort_values("Building_Count", ascending=False)
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top = result.head(20)
    top = top.copy()
    prefix = "D" if group_col == "区市" else "G"
    top["Plot_Label"] = [f"{prefix}{i:02d}" for i in range(1, len(top) + 1)]
    top[["Plot_Label", "Group", "Building_Count"]].to_csv(label_map_path, index=False, encoding="utf-8-sig")
    print(f"Figure labels are anonymized for CEUS-style English figures. Label map saved: {label_map_path}")

    top = top.sort_values("Building_Count", ascending=True)
    height = max(5, min(12, 0.42 * len(top) + 1.5))
    plt.figure(figsize=(10, height))
    plt.barh(top["Plot_Label"], top["Building_Count"], color="#2f7f9f")
    for idx, value in enumerate(top["Building_Count"]):
        plt.text(value, idx, f" {int(value)}", va="center", fontsize=8)
    plt.xlabel("Building count")
    group_label = GROUP_LABEL_EN.get(group_col, "group")
    plt.ylabel(group_label.title())
    plt.title(f"Building Sample Coverage by {group_label.title()} ({model_name}, {poi_radius} m POI)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"空间/分组诊断CSV已保存: {csv_path}")
    print(f"空间/分组诊断图已保存: {plot_path}")
    return str(csv_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create group-level spatial diagnostics without requiring GIS geometry.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", type=int, required=True)
    parser.add_argument("--group-col", default="区市")
    args = parser.parse_args()
    plot_group_prevalence(args.model_name, args.poi_radius, args.group_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

