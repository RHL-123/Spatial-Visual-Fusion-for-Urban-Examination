from __future__ import annotations

import argparse

import pandas as pd

from ceus_experiment_utils import ensure_ceus_extra_dir, load_feature_table
from experiment_registry import CLASS_NAMES, add_model_name_argument


def build_case_study_pack(model_name: str, poi_radius: int, top_k: int) -> str:
    cfg, _, df, _ = load_feature_table(model_name, poi_radius)
    rows = []
    for class_name in CLASS_NAMES:
        score_col = f"{class_name}_box_count"
        gt_col = f"{class_name}_gt"
        conf_col = f"{class_name}_max_conf"
        if score_col not in df.columns or gt_col not in df.columns:
            continue
        part = df[["楼栋ID", "区市", "街道", "社区", gt_col, score_col, conf_col]].copy()
        part["Class"] = class_name
        part["Case_Type"] = "high_conflict"
        part["Conflict_Score"] = (part[score_col] > 0).astype(int) != part[gt_col].astype(int)
        part["Sort_Score"] = part["Conflict_Score"].astype(int) * 1000 + part[score_col].fillna(0) + part[conf_col].fillna(0)
        rows.append(part.sort_values("Sort_Score", ascending=False).head(top_k))

    if not rows:
        raise ValueError("没有可构建案例包的数据列")
    out = pd.concat(rows, ignore_index=True)
    out_dir = ensure_ceus_extra_dir()
    out_path = out_dir / f"case_study_candidates_{model_name}_{poi_radius}m.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"案例候选包已保存: {out_path}")
    print(f"可结合 detailed_boxes 文件追溯图片: {cfg.detailed_boxes_csv}")
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate cases for qualitative CEUS-style diagnostics.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    build_case_study_pack(args.model_name, args.poi_radius, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

