"""
Old-format ground_truth ablation experiment with strict community-grouped CV.

This is a thin entry script around evaluate_ablation.py. It keeps the
same metrics, plots, and diagnostic outputs, but fixes cv_mode to
community_group so train/test communities never overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_ablation import GT_EXCEL, run_ablation
from experiment_registry import GT_EXCEL, add_model_name_argument


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation with strict community-grouped CV.")
    add_model_name_argument(parser)
    parser.add_argument("--poi-radius", required=True, type=int, help="POI 半径，只能是 500、1000、1500")
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=GT_EXCEL,
        help="GT Excel路径。默认使用 data/ground_truth/ground_truth.xlsx",
    )
    parser.add_argument("--quick", action="store_true", help="使用小参数网格快速冒烟测试。")
    parser.add_argument(
        "--fusion-policy",
        choices=["guarded", "legacy_rf"],
        default="guarded",
        help="guarded: 若RF全融合低于Vision则回退Vision；legacy_rf: 完全使用RF全融合。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_ablation(
            args.model_name,
            args.poi_radius,
            args.gt_path,
            quick=args.quick,
            fusion_policy=args.fusion_policy,
            cv_mode="community_group",
        )
    except Exception as exc:
        print(f"错误: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

