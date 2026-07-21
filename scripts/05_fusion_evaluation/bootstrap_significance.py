from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ceus_experiment_utils import ensure_ceus_extra_dir


def bootstrap_summary(summary_csv: str, n_boot: int, seed: int) -> str:
    df = pd.read_csv(summary_csv)
    required = {"Model", "POI_Radius_m", "Class", "Method", "F1-Score"}
    if not required.issubset(df.columns):
        raise ValueError(f"summary_csv 缺少列: {sorted(required - set(df.columns))}")

    base = df[df["Method"] == "0_Baseline"]
    full = df[df["Method"] == "3_Full_Fusion"]
    merged = full.merge(
        base,
        on=["Model", "POI_Radius_m", "Class"],
        suffixes=("_Full", "_Baseline"),
    )
    if merged.empty:
        raise ValueError("没有可比较的 Baseline 与 Full Fusion 成对结果")

    rng = np.random.default_rng(seed)
    rows = []
    for (model, radius), group in merged.groupby(["Model", "POI_Radius_m"]):
        deltas = group["F1-Score_Full"].values - group["F1-Score_Baseline"].values
        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            boot_means.append(float(np.mean(sample)))
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        p_nonpositive = float(np.mean(np.array(boot_means) <= 0))
        rows.append(
            {
                "Model": model,
                "POI_Radius_m": int(radius),
                "Class_Count": len(deltas),
                "Mean_Delta_F1": float(np.mean(deltas)),
                "CI95_Low": float(lo),
                "CI95_High": float(hi),
                "Bootstrap_P_Delta_LE_0": p_nonpositive,
                "Note": "Class-level bootstrap over defect categories; stronger sample-level bootstrap requires saved per-building predictions.",
            }
        )

    out_dir = ensure_ceus_extra_dir()
    out_path = out_dir / "bootstrap_significance_from_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"bootstrap 显著性结果已保存: {out_path}")
    return str(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Full Fusion vs Baseline F1 from ablation summary.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    bootstrap_summary(args.summary_csv, args.n_boot, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

