from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_EXP_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "processed_visual_predictions"
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablation_results"
LEGACY_DIR = PROJECT_ROOT / "outputs" / "legacy"
REGISTRY_PATH = Path(__file__).resolve().with_name("model_registry.yaml")

GT_EXCEL = PROJECT_ROOT / "data" / "ground_truth" / "ground_truth.xlsx"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw_images_not_included"

POI_RADIUS_FILES = {
    500: PROJECT_ROOT / "data" / "poi_features" / "poi_features_500m.csv",
    1000: PROJECT_ROOT / "data" / "poi_features" / "poi_features_1000m.csv",
    1500: PROJECT_ROOT / "data" / "poi_features" / "poi_features_1500m.csv",
}

CLASS_NAMES = [
    "pipeline_damage",
    "addition_balconies_windows",
    "public_spaces_illegal_occupation",
    "no_aging_place_modifications",
    "wall_damage",
    "illegal_renovation_expansion",
    "elevator_addition",
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    role: str
    arch: str
    imgsz: int
    batch: int
    output_prefix: str
    expected_data_keyword: str
    weights: Path
    args: Path

    @property
    def prediction_excel(self) -> Path:
        return PREDICTIONS_DIR / f"predict_enhanced_features_{self.output_prefix}.xlsx"

    @property
    def detailed_boxes_csv(self) -> Path:
        return PREDICTIONS_DIR / f"predict_detailed_boxes_{self.output_prefix}.csv"


def ensure_output_dirs() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _load_registry_data() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"模型注册表不存在: {REGISTRY_PATH}")

    try:
        import yaml  # type: ignore

        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("model_registry.yaml 顶层必须是字典")
        return data
    except ModuleNotFoundError:
        return _load_registry_data_without_yaml()


def _load_registry_data_without_yaml() -> dict[str, Any]:
    data: dict[str, Any] = {"models": {}}
    current_model: str | None = None
    for raw_line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        if raw_line == "models:":
            continue
        model_match = re.match(r"^  ([A-Za-z0-9_]+):\s*$", raw_line)
        if model_match:
            current_model = model_match.group(1)
            data["models"][current_model] = {}
            continue
        field_match = re.match(r"^    ([A-Za-z0-9_]+):\s*(.+?)\s*$", raw_line)
        if field_match and current_model:
            key, value = field_match.groups()
            data["models"][current_model][key] = _coerce_scalar(value)
    return data


def load_models(validate: bool = False) -> dict[str, ModelConfig]:
    raw = _load_registry_data()
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        raise ValueError("model_registry.yaml 必须包含 models 字典")

    models: dict[str, ModelConfig] = {}
    for name, item in models_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"模型 {name} 的配置必须是字典")
        cfg = ModelConfig(
            name=name,
            display_name=str(item["display_name"]),
            role=str(item["role"]),
            arch=str(item["arch"]),
            imgsz=int(item["imgsz"]),
            batch=int(item["batch"]),
            output_prefix=str(item["output_prefix"]),
            expected_data_keyword=str(item["expected_data_keyword"]),
            weights=PROJECT_ROOT / str(item["weights"]),
            args=PROJECT_ROOT / str(item["args"]),
        )
        if validate:
            validate_model_config(cfg)
        models[name] = cfg
    return models


def available_model_names() -> list[str]:
    return sorted(load_models(validate=False).keys())


def get_model_config(model_name: str, validate: bool = False) -> ModelConfig:
    models = load_models(validate=validate)
    if model_name not in models:
        choices = ", ".join(sorted(models))
        suggestion = ""
        if model_name in {"yolov8", "yolov9", "yolov10", "yolo11", "rtdetr"}:
            suggestion = f" 请使用明确实验ID，例如 {model_name}_balanced。"
        raise ValueError(f"未知 model-name: {model_name}.{suggestion} 可用值: {choices}")
    return models[model_name]


def validate_model_config(cfg: ModelConfig) -> None:
    missing = [str(path) for path in (cfg.weights, cfg.args) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{cfg.name} 缺少必要文件: {missing}")
    args_text = cfg.args.read_text(encoding="utf-8", errors="ignore")
    if cfg.expected_data_keyword not in args_text:
        raise ValueError(
            f"{cfg.name} 的 args.yaml 未包含 {cfg.expected_data_keyword}，"
            f"请勿作为正式 balanced 实验使用: {cfg.args}"
        )


def get_poi_table(radius: int) -> Path:
    if radius not in POI_RADIUS_FILES:
        allowed = ", ".join(str(k) for k in sorted(POI_RADIUS_FILES))
        raise ValueError(f"--poi-radius 只能是 {allowed}，当前为 {radius}")
    path = POI_RADIUS_FILES[radius]
    if not path.exists():
        raise FileNotFoundError(f"POI feature table 不存在: {path}")
    return path


def add_model_name_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name",
        required=True,
        help="明确实验ID，例如 yolov8_balanced、yolo11_balanced；不要使用 yolo11 这类裸模型名。",
    )


