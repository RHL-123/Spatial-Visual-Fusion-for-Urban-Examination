"""
Extract GridSearchCV best RF parameters from ablation nohup logs.

The historical ablation script printed best_params_ to stdout but did not save
them in the CSV/TXT metric artifacts. This utility reconstructs a reusable JSON
cache from the log blocks.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "new_experiment" / "logs" / "resume_ablation_fixed_plots_20260604_085311.log"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "new_experiment"
    / "ablation_results"
    / "best_rf_params_from_resume_ablation_fixed_plots_20260604_085311.json"
)

BEST_PARAM_RE = re.compile(r"^\[(?P<class>[A-Za-z0-9_]+)\]\s*最优参数:\s*(?P<params>\{.*\})\s*$")
REPORT_RE = re.compile(
    r"^ablation_(?P<model>.+)_(?P<radius>500|1000|1500)m_(?P<timestamp>\d{8}_\d{6})\.txt$"
)


def extract_best_params(log_path: Path) -> dict:
    pending_params: dict[str, dict] = {}
    runs: list[dict] = []
    combos: dict[str, dict[str, dict[str, dict]]] = {}

    for line_no, line in enumerate(log_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        best_match = BEST_PARAM_RE.search(line.strip())
        if best_match:
            cls = best_match.group("class")
            params = ast.literal_eval(best_match.group("params"))
            pending_params[cls] = params
            continue

        if "消融报告已保存至:" not in line:
            continue
        report_path = line.split("消融报告已保存至:", 1)[1].strip()
        report_match = REPORT_RE.match(Path(report_path).name)
        if not report_match:
            continue

        model = report_match.group("model")
        radius = report_match.group("radius")
        timestamp = report_match.group("timestamp")
        run_name = f"ablation_{model}_{radius}m_{timestamp}"
        runs.append(
            {
                "run_name": run_name,
                "model": model,
                "poi_radius": int(radius),
                "timestamp": timestamp,
                "line_no": line_no,
                "class_params": pending_params,
                "class_count": len(pending_params),
            }
        )
        combos.setdefault(model, {})[radius] = pending_params
        pending_params = {}

    expected_models = [
        "rtdetr_balanced",
        "yolo11_balanced",
        "yolov10_balanced",
        "yolov8_balanced",
        "yolov9_1280_balanced",
        "yolov9_640_balanced",
    ]
    expected_radii = ["500", "1000", "1500"]
    missing_combos = [
        {"model": model, "poi_radius": int(radius)}
        for model in expected_models
        for radius in expected_radii
        if model not in combos or radius not in combos[model]
    ]
    incomplete_runs = [
        {"run_name": run["run_name"], "class_count": run["class_count"]}
        for run in runs
        if run["class_count"] != 7
    ]

    return {
        "source_log": str(log_path),
        "combo_count": sum(len(radii) for radii in combos.values()),
        "class_param_count": sum(run["class_count"] for run in runs),
        "expected_combo_count": len(expected_models) * len(expected_radii),
        "expected_class_param_count": len(expected_models) * len(expected_radii) * 7,
        "missing_combos": missing_combos,
        "incomplete_runs": incomplete_runs,
        "combos": combos,
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract RF best params from ablation logs.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = extract_best_params(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已提取参数文件: {args.output}")
    print(f"组合数: {data['combo_count']} / {data['expected_combo_count']}")
    print(f"类别参数数: {data['class_param_count']} / {data['expected_class_param_count']}")
    if data["missing_combos"]:
        print("缺失组合:")
        for item in data["missing_combos"]:
            print(f"  - {item['model']} {item['poi_radius']}m")
    if data["incomplete_runs"]:
        print("不完整运行:")
        for item in data["incomplete_runs"]:
            print(f"  - {item['run_name']}: {item['class_count']} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

