"""
Summarize all unified ablation CSV outputs into one matrix-friendly CSV.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from experiment_registry import ABLATION_DIR, ensure_output_dirs


def summarize() -> Path:
    ensure_output_dirs()
    csv_files = sorted(ABLATION_DIR.glob("ablation_*_*m_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到统一后矫正CSV: {ABLATION_DIR / 'ablation_*_*m_*.csv'}")

    frames = []
    pattern = re.compile(r"^ablation_(?P<model>.+)_(?P<radius>\d+)m_\d{8}_\d{6}\.csv$")
    for path in csv_files:
        df = pd.read_csv(path)
        match = pattern.match(path.name)
        if match:
            df["Source_Model_From_File"] = match.group("model")
            df["Source_POI_Radius_From_File"] = int(match.group("radius"))
        df["Source_File"] = path.relative_to(ABLATION_DIR.parent).as_posix()
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out_path = ABLATION_DIR / "ablation_matrix_summary.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    pivot_path = ABLATION_DIR / "ablation_matrix_f1_full_fusion.csv"
    full = merged[merged["Method"] == "3_Full_Fusion"].copy()
    if not full.empty:
        pivot = full.pivot_table(
            index=["Model", "POI_Radius_m"],
            columns="Class",
            values="F1-Score",
            aggfunc="max",
        ).reset_index()
        pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")
    print(f"汇总CSV已保存至: {out_path}")
    if pivot_path.exists():
        print(f"Full Fusion F1透视表已保存至: {pivot_path}")
    return out_path


if __name__ == "__main__":
    try:
        summarize()
    except Exception as exc:
        print(f"错误: {exc}")
        raise SystemExit(1)

