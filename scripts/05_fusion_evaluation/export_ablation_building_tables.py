"""
Export building-level complete classification tables for ablation methods.

This script mirrors new_experiment/evaluate_ablation.py, but writes the
per-building predictions for 0_Baseline, 1_Vision_Only, 2_POI_Only, and
3_Full_Fusion instead of only class-level metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "拆分结果-gis画图" / "完整分类表.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "new_experiment" / "post_correction_tables" / "ablation_building_tables"
DEFAULT_PARAM_CACHE = (
    PROJECT_ROOT
    / "new_experiment"
    / "ablation_results"
    / "best_rf_params_from_resume_ablation_fixed_plots_20260604_085311.json"
)

METHOD_OUTPUT_NAMES = {
    "0_Baseline": "0_Baseline",
    "1_Vision_Only_RF": "1_Vision_Only_RF",
    "2_POI_Only_RF": "2_POI_Only_RF",
    "3_Full_Fusion_RF": "3_Full_Fusion_RF",
}

RF_PARAM_GRID = {
    "n_estimators": [100, 150, 200, 250, 300, 350, 400],
    "max_depth": [4, 6, 8, 12, 16, 20],
    "min_samples_split": [2, 5, 8, 10, 12],
    "min_samples_leaf": [1, 2, 4, 6, 8],
}


def _ensure_import_path() -> None:
    new_exp_dir = PROJECT_ROOT / "new_experiment"
    path = str(new_exp_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_experiment_modules():
    _ensure_import_path()
    from evaluate_ablation import load_and_merge
    from experiment_registry import CLASS_NAMES, POI_RADIUS_FILES, available_model_names

    return load_and_merge, CLASS_NAMES, POI_RADIUS_FILES, available_model_names


def output_path_for(output_dir: Path, model_name: str, poi_radius: int, method: str) -> Path:
    if method not in METHOD_OUTPUT_NAMES:
        allowed = ", ".join(METHOD_OUTPUT_NAMES)
        raise ValueError(f"未知方法: {method}. 可用方法: {allowed}")
    combo_dir = output_dir / f"{model_name}_{poi_radius}m"
    return combo_dir / f"{METHOD_OUTPUT_NAMES[method]}完整分类表_{model_name}_{poi_radius}m.xlsx"


def build_complete_classification_table(
    template_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    label_columns: list[str],
) -> pd.DataFrame:
    """Return template rows with prediction label columns mapped by 楼栋ID."""
    out = template_df.copy()
    pred = prediction_df[["楼栋ID", *label_columns]].copy()

    out["_楼栋ID_key"] = out["楼栋ID"].astype(str)
    pred["_楼栋ID_key"] = pred["楼栋ID"].astype(str)
    pred = pred.drop_duplicates("_楼栋ID_key", keep="first").set_index("_楼栋ID_key")

    missing_ids = sorted(set(out["_楼栋ID_key"]) - set(pred.index))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(f"预测结果缺少 {len(missing_ids)} 个楼栋ID，例如: {preview}")

    for col in label_columns:
        if col not in out.columns:
            raise ValueError(f"完整分类表缺少标签列: {col}")
        if col not in pred.columns:
            raise ValueError(f"预测结果缺少标签列: {col}")
        out[col] = out["_楼栋ID_key"].map(pred[col]).astype(int)

    return out.drop(columns=["_楼栋ID_key"])


def validate_output_table(
    out_df: pd.DataFrame,
    template_df: pd.DataFrame,
    label_columns: list[str],
) -> None:
    if out_df.shape != template_df.shape:
        raise ValueError(f"输出表形状不一致: output={out_df.shape}, template={template_df.shape}")
    if list(out_df.columns) != list(template_df.columns):
        raise ValueError("输出表列名或列顺序与完整分类表模板不一致")
    if out_df["楼栋ID"].astype(str).tolist() != template_df["楼栋ID"].astype(str).tolist():
        raise ValueError("输出表楼栋ID顺序与完整分类表模板不一致")
    for col in label_columns:
        values = set(out_df[col].dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"输出表 {col} 包含非0/1值: {sorted(values)}")


def load_param_cache(param_cache: Path | None) -> dict:
    if param_cache is None:
        return {}
    if not param_cache.exists():
        raise FileNotFoundError(f"参数缓存不存在: {param_cache}")
    return json.loads(param_cache.read_text(encoding="utf-8"))


def get_cached_params(cache: dict, model_name: str, poi_radius: int, cls: str) -> dict | None:
    return (
        cache.get("combos", {})
        .get(model_name, {})
        .get(str(poi_radius), {})
        .get(cls)
    )


def should_skip_combo_for_missing_cache(cache: dict, model_name: str, poi_radius: int) -> bool:
    _load_and_merge, class_names, _poi_radius_files, _available_model_names = _load_experiment_modules()
    combo = cache.get("combos", {}).get(model_name, {}).get(str(poi_radius), {})
    return any(cls not in combo for cls in class_names)


def select_models(model_name: str | None) -> list[str]:
    _load_and_merge, _class_names, _poi_radius_files, available_model_names = _load_experiment_modules()
    models = available_model_names()
    if model_name is None:
        return models
    if model_name not in models:
        choices = ", ".join(models)
        raise ValueError(f"未知 model-name: {model_name}. 可用值: {choices}")
    return [model_name]


def select_radii(poi_radius: int | None) -> list[int]:
    _load_and_merge, _class_names, poi_radius_files, _available_model_names = _load_experiment_modules()
    radii = sorted(poi_radius_files)
    if poi_radius is None:
        return radii
    if poi_radius not in poi_radius_files:
        allowed = ", ".join(str(r) for r in radii)
        raise ValueError(f"--poi-radius 只能是 {allowed}，当前为 {poi_radius}")
    return [poi_radius]


def _class_feature_columns(cls: str) -> list[str]:
    return [
        f"{cls}_box_count",
        f"{cls}_max_conf",
        f"{cls}_mean_conf",
        f"{cls}_infection_rate",
        f"{cls}_median_conf",
        f"{cls}_std_conf",
    ]


def _rf_predictions_for_combo(
    model_name: str,
    poi_radius: int,
    param_cache_data: dict,
    grid_search_missing: bool,
) -> tuple[dict[str, pd.DataFrame], object]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_predict

    load_and_merge, class_names, _poi_radius_files, _available_model_names = _load_experiment_modules()
    cfg, _poi_json, df, df_poi = load_and_merge(model_name, poi_radius)
    poi_features = [c for c in df_poi.columns if c != "社区"]

    method_predictions = {
        method: pd.DataFrame({"楼栋ID": df["楼栋ID"]})
        for method in METHOD_OUTPUT_NAMES
    }

    for cls in class_names:
        y = df[f"{cls}_gt"].astype(int).values
        box_count_col = f"{cls}_box_count"
        if box_count_col in df.columns:
            method_predictions["0_Baseline"][cls] = (df[box_count_col] > 0).astype(int).values
        else:
            method_predictions["0_Baseline"][cls] = np.zeros(len(df), dtype=int)

        if len(np.unique(y)) < 2:
            for method in ("1_Vision_Only_RF", "2_POI_Only_RF", "3_Full_Fusion_RF"):
                method_predictions[method][cls] = np.zeros(len(df), dtype=int)
            print(f"[{model_name} {poi_radius}m {cls}] 缺乏正负样本，RF方法输出全0。")
            continue

        valid_vision = [c for c in _class_feature_columns(cls) if c in df.columns]
        valid_poi = [c for c in poi_features if c in df.columns]
        if not valid_vision:
            raise ValueError(f"{model_name} 缺少 {cls} 的视觉统计特征列")
        if not valid_poi:
            raise ValueError(f"{poi_radius}m POI 数据未产生可用POI特征")

        x_vision = df[valid_vision].values
        x_poi = df[valid_poi].values
        x_all = np.hstack([x_vision, x_poi])

        cached_params = get_cached_params(param_cache_data, model_name, poi_radius, cls)
        if cached_params is not None:
            print(f"[{model_name} {poi_radius}m {cls}] 使用缓存参数: {cached_params}")
            clf = RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                **cached_params,
            )
        elif grid_search_missing:
            grid_rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=grid_rf,
                param_grid=RF_PARAM_GRID,
                cv=3,
                scoring="f1",
                n_jobs=-1,
            )
            print(f"[{model_name} {poi_radius}m {cls}] 缓存缺失，运行GridSearchCV...")
            grid_search.fit(x_all, y)
            clf = grid_search.best_estimator_
            print(f"[{model_name} {poi_radius}m {cls}] best_params={grid_search.best_params_}")
        else:
            raise KeyError(
                f"缺少缓存参数: model={model_name}, poi_radius={poi_radius}, class={cls}. "
                "如需现场补算，请显式加 --grid-search-missing。"
            )

        method_predictions["1_Vision_Only_RF"][cls] = cross_val_predict(clf, x_vision, y, cv=5).astype(int)
        method_predictions["2_POI_Only_RF"][cls] = cross_val_predict(clf, x_poi, y, cv=5).astype(int)
        method_predictions["3_Full_Fusion_RF"][cls] = cross_val_predict(clf, x_all, y, cv=5).astype(int)

    return method_predictions, cfg


def export_combo(
    model_name: str,
    poi_radius: int,
    template_df: pd.DataFrame,
    label_columns: list[str],
    output_dir: Path,
    param_cache_data: dict,
    grid_search_missing: bool,
) -> list[Path]:
    method_predictions, cfg = _rf_predictions_for_combo(
        model_name,
        poi_radius,
        param_cache_data,
        grid_search_missing,
    )
    written: list[Path] = []
    for method, prediction_df in method_predictions.items():
        out_df = build_complete_classification_table(template_df, prediction_df, label_columns)
        validate_output_table(out_df, template_df, label_columns)
        out_path = output_path_for(output_dir, cfg.name, poi_radius, method)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_excel(out_path, index=False)
        written.append(out_path)
        print(f"已保存: {out_path}")
    return written


def export_ablation_building_tables(
    model_name: str | None = None,
    poi_radius: int | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    template_path: Path = DEFAULT_TEMPLATE,
    param_cache: Path | None = DEFAULT_PARAM_CACHE,
    grid_search_missing: bool = False,
) -> list[Path]:
    _load_and_merge, class_names, _poi_radius_files, _available_model_names = _load_experiment_modules()
    models = select_models(model_name)
    radii = select_radii(poi_radius)
    template_df = pd.read_excel(template_path)
    param_cache_data = load_param_cache(param_cache)

    skipped: list[tuple[str, int]] = []
    export_targets: list[tuple[str, int]] = []
    for model in models:
        for radius in radii:
            if param_cache_data and should_skip_combo_for_missing_cache(param_cache_data, model, radius) and not grid_search_missing:
                skipped.append((model, radius))
            else:
                export_targets.append((model, radius))

    expected_combos = len(export_targets)
    expected_tables = expected_combos * len(METHOD_OUTPUT_NAMES)
    requested_combos = len(models) * len(radii)
    requested_tables = requested_combos * len(METHOD_OUTPUT_NAMES)
    print(f"请求组合数: {requested_combos}")
    print(f"请求输出表数: {requested_tables}")
    if param_cache is not None:
        print(f"参数缓存: {param_cache}")
    if skipped:
        print("以下组合缺少完整缓存参数，默认跳过；如需补算请加 --grid-search-missing:")
        for model, radius in skipped:
            print(f"  - {model} {radius}m")
    print(f"预计组合数: {expected_combos}")
    print(f"预计输出表数: {expected_tables}")

    written: list[Path] = []
    for model, radius in export_targets:
        print(f"\n=== 导出 {model} / {radius}m ===")
        written.extend(
            export_combo(
                model,
                radius,
                template_df,
                class_names,
                output_dir,
                param_cache_data,
                grid_search_missing,
            )
        )

    print("\n导出完成")
    print(f"请求组合数: {requested_combos}")
    print(f"跳过组合数: {len(skipped)}")
    print(f"预计组合数: {expected_combos}")
    print(f"预计输出表数: {expected_tables}")
    print(f"实际成功输出表数: {len(written)}")
    for path in written:
        print(path)
    if len(written) != expected_tables:
        raise RuntimeError(f"实际输出表数不等于预期: actual={len(written)}, expected={expected_tables}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export building-level ablation complete classification tables.")
    parser.add_argument("--model-name", default=None, help="只导出指定模型，例如 yolov8_balanced；默认导出全部模型。")
    parser.add_argument("--poi-radius", default=None, type=int, help="只导出指定POI半径，例如 500；默认导出全部半径。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录。")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="完整分类表模板路径。")
    parser.add_argument(
        "--param-cache",
        type=Path,
        default=DEFAULT_PARAM_CACHE,
        help="RF最优参数JSON缓存；默认使用从历史log提取的缓存。",
    )
    parser.add_argument(
        "--no-param-cache",
        action="store_true",
        help="不使用参数缓存，所有组合重新GridSearch；非常慢，不建议全量使用。",
    )
    parser.add_argument(
        "--grid-search-missing",
        action="store_true",
        help="当某个组合/类别缺少缓存参数时，现场运行GridSearch补算。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_ablation_building_tables(
        model_name=args.model_name,
        poi_radius=args.poi_radius,
        output_dir=args.output_dir,
        template_path=args.template,
        param_cache=None if args.no_param_cache else args.param_cache,
        grid_search_missing=args.grid_search_missing or args.no_param_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

