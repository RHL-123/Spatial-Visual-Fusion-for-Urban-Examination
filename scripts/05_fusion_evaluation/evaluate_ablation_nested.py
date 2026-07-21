from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy, spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fusion_utils import (
    FUSION_RESULTS,
    FUSION_TABLES,
    MODEL_NAMES,
    atomic_json_write,
    choose_group_splits,
    ensure_fusion_dirs,
    select_f1_threshold,
    stable_seed,
    validate_precomputed_splits,
)
from experiment_registry import CLASS_NAMES, GT_EXCEL, PROJECT_ROOT, get_model_config, get_poi_json


labelED_GT = FUSION_TABLES / "ground_truth.xlsx"
CORRECTION_DIR = PROJECT_ROOT / "04_Correction_and_Fusion"
BUILDING_COORDS_CSV = CORRECTION_DIR / "各社区楼栋坐标明细.csv"
VISION_SUFFIXES = ["box_count", "max_conf", "mean_conf", "infection_rate", "median_conf", "std_conf"]
VISION_FEATURE_PROFILES = {
    "all6": ["box_count", "max_conf", "mean_conf", "infection_rate", "median_conf", "std_conf"],
    "no_max": ["box_count", "mean_conf", "infection_rate", "median_conf", "std_conf"],
    "no_median": ["box_count", "max_conf", "mean_conf", "infection_rate", "std_conf"],
    "no_mean": ["box_count", "max_conf", "infection_rate", "median_conf", "std_conf"],
    "count_rate_conf": ["box_count", "max_conf", "infection_rate"],
    "count_rate": ["box_count", "infection_rate"],
    "conf_stats": ["max_conf", "mean_conf", "median_conf", "std_conf"],
}
POI_CATEGORIES = ["餐饮服务", "生活服务", "购物服务", "公司企业", "医疗保健", "科教文化", "住宿服务"]
BASE_POI_NUMERIC = [
    "total_poi",
    *[f"poi_{category}_ratio" for category in POI_CATEGORIES],
    "poi_diversity_entropy",
    "poi_commercial_intensity",
]


def safe_feature_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", str(text)).strip("_")


def parse_distance(value: object, default: float) -> float:
    if value is None or isinstance(value, list):
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def normalize_primary_category(raw: str) -> str | None:
    primary = str(raw or "").split("|", 1)[0].split(";", 1)[0].strip()
    if primary == "医疗保健服务":
        return "医疗保健"
    if primary == "科教文化服务":
        return "科教文化"
    if primary in POI_CATEGORIES:
        return primary
    return None


def haversine_meters(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    radius = 6371000.0
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    dlon = lon2_rad[None, :] - lon1_rad[:, None]
    dlat = lat2_rad[None, :] - lat1_rad[:, None]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad)[:, None] * np.cos(lat2_rad)[None, :] * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class Candidate:
    name: str
    estimator: object
    scale_numeric: bool
    weighted_fit: bool = False


@dataclass
class TunedModel:
    candidate: Candidate
    threshold: float
    inner_f1: float
    inner_precision: float
    fitted: Pipeline | None = None


def build_poi_frame_with_missing(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for community, payload in raw.items():
        valid = (
            isinstance(payload, dict)
            and bool(payload)
            and "category_distribution" in payload
            and isinstance(payload.get("category_distribution"), dict)
        )
        row = {"社区": str(community).strip(), "poi_missing_indicator": int(not valid)}
        if not valid:
            payload = {}
        total = float(payload.get("total_count", 0) or 0)
        distribution = payload.get("category_distribution", {}) or {}
        row["total_poi"] = total
        ratios = []
        for category in POI_CATEGORIES:
            count = distribution.get(category, distribution.get(f"{category}服务", 0)) or 0
            ratio = float(count) / total if total > 0 else 0.0
            row[f"poi_{category}_ratio"] = ratio
            ratios.append(ratio)
        row["poi_diversity_entropy"] = float(entropy(ratios)) if sum(ratios) > 0 else 0.0
        row["poi_dominant_category"] = (
            POI_CATEGORIES[int(np.argmax(ratios))] if sum(ratios) > 0 else "NONE"
        )
        row["poi_commercial_intensity"] = row["poi_餐饮服务_ratio"] + row["poi_购物服务_ratio"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_fine_poi_frame(path: Path, radius: int) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    area_km2 = math.pi * (radius / 1000.0) ** 2
    for community, payload in raw.items():
        if not isinstance(payload, dict):
            payload = {}
        total = float(payload.get("total_count", 0) or 0)
        row: dict[str, object] = {"社区": str(community).strip()}

        for poi_type, count in (payload.get("type_distribution") or {}).items():
            key = safe_feature_name(poi_type)
            if not key:
                continue
            value = float(count or 0)
            row[f"fine_count_{key}"] = value
            row[f"fine_ratio_{key}"] = value / total if total > 0 else 0.0
            row[f"fine_density_{key}"] = value / area_km2 if area_km2 > 0 else 0.0

        distances: dict[str, list[float]] = {}
        for poi in payload.get("poi_list", []) or []:
            if not isinstance(poi, dict):
                continue
            key = safe_feature_name(poi.get("type", ""))
            if not key:
                continue
            distances.setdefault(key, []).append(parse_distance(poi.get("distance"), float(radius)))

        for key, values in distances.items():
            array = np.asarray(values, dtype=float)
            row[f"fine_near300_{key}"] = float((array <= 300).sum())
            row[f"fine_near500_{key}"] = float((array <= 500).sum())
            row[f"fine_inv_distance_{key}"] = float((1.0 / (1.0 + array / 100.0)).sum())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def parse_poi_location(value: object) -> tuple[float, float] | None:
    if value is None or isinstance(value, list):
        return None
    parts = str(value).split(",")
    if len(parts) < 2:
        return None
    try:
        lon = float(parts[0])
        lat = float(parts[1])
    except ValueError:
        return None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    return lon, lat


def empty_building_decay_row(building_id: object) -> dict[str, float | str]:
    row: dict[str, float | str] = {"楼栋ID": str(building_id)}
    for category in POI_CATEGORIES:
        key = safe_feature_name(category)
        for sigma in (100, 300, 500):
            row[f"building_poi_{key}_decay_s{sigma}"] = 0.0
        row[f"building_poi_{key}_near100_count"] = 0.0
        row[f"building_poi_{key}_near300_count"] = 0.0
        row[f"building_poi_{key}_near500_count"] = 0.0
        row[f"building_poi_{key}_ring300_500_count"] = 0.0
        row[f"building_poi_{key}_ring500_1000_count"] = 0.0
        row[f"building_poi_{key}_min_distance"] = 1000.0
    row["building_poi_total_decay_s300"] = 0.0
    row["building_poi_commercial_decay_s300"] = 0.0
    row["building_poi_life_decay_s300"] = 0.0
    row["building_poi_public_service_decay_s300"] = 0.0
    row["building_poi_total_near300_count"] = 0.0
    row["building_poi_total_near500_count"] = 0.0
    row["building_poi_min_distance"] = 1000.0
    return row


def build_building_distance_decay_poi_frame(path: Path, radius: int) -> pd.DataFrame:
    if not BUILDING_COORDS_CSV.exists():
        raise FileNotFoundError(f"缺少楼栋坐标文件: {BUILDING_COORDS_CSV}")
    coords = pd.read_csv(BUILDING_COORDS_CSV)
    coords["楼栋ID"] = coords["楼栋ID"].astype(str)
    coords["社区"] = coords["社区"].astype(str).str.strip()
    coords["经度"] = pd.to_numeric(coords["经度"], errors="coerce")
    coords["纬度"] = pd.to_numeric(coords["纬度"], errors="coerce")
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, float | str]] = []
    for community, buildings in coords.groupby("社区", sort=False):
        building_rows = buildings.dropna(subset=["经度", "纬度"]).copy()
        missing_coord_ids = set(buildings.loc[buildings[["经度", "纬度"]].isna().any(axis=1), "楼栋ID"].astype(str))
        payload = raw.get(str(community).strip(), {})
        poi_by_category: dict[str, list[tuple[float, float]]] = {category: [] for category in POI_CATEGORIES}
        if isinstance(payload, dict):
            for poi in payload.get("poi_list", []) or []:
                if not isinstance(poi, dict):
                    continue
                category = normalize_primary_category(str(poi.get("type", "")))
                if category is None:
                    continue
                location = parse_poi_location(poi.get("location"))
                if location is None:
                    location = parse_poi_location(poi.get("entr_location"))
                if location is None:
                    continue
                poi_by_category[category].append(location)

        for building_id in missing_coord_ids:
            rows.append(empty_building_decay_row(building_id))
        if building_rows.empty:
            continue
        lon = building_rows["经度"].to_numpy(dtype=float)
        lat = building_rows["纬度"].to_numpy(dtype=float)
        per_building: list[dict[str, float | str]] = [
            empty_building_decay_row(building_id)
            for building_id in building_rows["楼栋ID"].astype(str).tolist()
        ]
        all_min_distances: list[np.ndarray] = []
        for category, locations in poi_by_category.items():
            key = safe_feature_name(category)
            if not locations:
                continue
            poi_lon = np.asarray([location[0] for location in locations], dtype=float)
            poi_lat = np.asarray([location[1] for location in locations], dtype=float)
            distances = haversine_meters(lon, lat, poi_lon, poi_lat)
            distances = np.where(distances <= radius, distances, np.nan)
            valid = np.isfinite(distances)
            if not valid.any():
                continue
            distance_for_min = np.where(valid, distances, float(radius))
            category_min = distance_for_min.min(axis=1)
            all_min_distances.append(category_min)
            decay100 = np.nansum(np.where(valid, np.exp(-distances / 100.0), 0.0), axis=1)
            decay300 = np.nansum(np.where(valid, np.exp(-distances / 300.0), 0.0), axis=1)
            decay500 = np.nansum(np.where(valid, np.exp(-distances / 500.0), 0.0), axis=1)
            near100 = np.nansum(np.where(valid & (distances <= 100), 1.0, 0.0), axis=1)
            near300 = np.nansum(np.where(valid & (distances <= 300), 1.0, 0.0), axis=1)
            near500 = np.nansum(np.where(valid & (distances <= 500), 1.0, 0.0), axis=1)
            ring300_500 = np.nansum(np.where(valid & (distances > 300) & (distances <= 500), 1.0, 0.0), axis=1)
            ring500_1000 = np.nansum(np.where(valid & (distances > 500) & (distances <= 1000), 1.0, 0.0), axis=1)
            min_distance = category_min
            for index, row in enumerate(per_building):
                row[f"building_poi_{key}_decay_s100"] = float(decay100[index])
                row[f"building_poi_{key}_decay_s300"] = float(decay300[index])
                row[f"building_poi_{key}_decay_s500"] = float(decay500[index])
                row[f"building_poi_{key}_near100_count"] = float(near100[index])
                row[f"building_poi_{key}_near300_count"] = float(near300[index])
                row[f"building_poi_{key}_near500_count"] = float(near500[index])
                row[f"building_poi_{key}_ring300_500_count"] = float(ring300_500[index])
                row[f"building_poi_{key}_ring500_1000_count"] = float(ring500_1000[index])
                row[f"building_poi_{key}_min_distance"] = float(min_distance[index])
                row["building_poi_total_decay_s300"] = float(row["building_poi_total_decay_s300"]) + float(decay300[index])
                row["building_poi_total_near300_count"] = float(row["building_poi_total_near300_count"]) + float(near300[index])
                row["building_poi_total_near500_count"] = float(row["building_poi_total_near500_count"]) + float(near500[index])
                if category in {"餐饮服务", "购物服务"}:
                    row["building_poi_commercial_decay_s300"] = float(row["building_poi_commercial_decay_s300"]) + float(decay300[index])
                if category in {"生活服务", "医疗保健", "科教文化"}:
                    row["building_poi_life_decay_s300"] = float(row["building_poi_life_decay_s300"]) + float(decay300[index])
                if category in {"医疗保健", "科教文化"}:
                    row["building_poi_public_service_decay_s300"] = float(row["building_poi_public_service_decay_s300"]) + float(decay300[index])
        if all_min_distances:
            stacked_min = np.vstack(all_min_distances)
            overall_min = stacked_min.min(axis=0)
            for index, row in enumerate(per_building):
                row["building_poi_min_distance"] = float(overall_min[index])
        rows.extend(per_building)
    result = pd.DataFrame(rows).drop_duplicates("楼栋ID", keep="first").fillna(0.0)
    return result


def load_feature_frame(
    model_name: str,
    radius: int,
    include_fine_poi: bool = False,
    include_building_decay_poi: bool = False,
    require: bool = False,
) -> tuple[pd.DataFrame, list[str], str]:
    cfg = get_model_config(model_name)
    features = pd.read_excel(cfg.prediction_excel)
    original = pd.read_excel(GT_EXCEL)
    frame = original[["楼栋ID", *CLASS_NAMES]].merge(
        features, on="楼栋ID", how="inner", suffixes=("", "_pred")
    )
    if len(frame) != len(original):
        raise ValueError(f"{model_name}特征只匹配 {len(frame)}/{len(original)} 栋")
    if require:
        if not labelED_GT.exists():
            raise FileNotFoundError(f"需要GT，但文件不存在: {labelED_GT}")
        ground_truth = pd.read_excel(labelED_GT, sheet_name="GT")
        if original["楼栋ID"].astype(str).tolist() != ground_truth["楼栋ID"].astype(str).tolist():
            raise ValueError("GT与原始GT楼栋顺序不一致")
        label_columns = ["楼栋ID"]
        for class_name in CLASS_NAMES:
            label_columns.extend([f"{class_name}", f"{class_name}_excluded", f"{class_name}_label_status"])
        frame = frame.merge(ground_truth[label_columns], on="楼栋ID", how="left")
    else:
        for class_name in CLASS_NAMES:
            frame[f"{class_name}"] = frame[class_name].astype(int)
            frame[f"{class_name}_excluded"] = 0
            frame[f"{class_name}_label_status"] = ""

    poi = build_poi_frame_with_missing(get_poi_json(radius))
    if include_fine_poi:
        fine = build_fine_poi_frame(get_poi_json(radius), radius)
        poi = poi.merge(fine, on="社区", how="left")
    frame["社区"] = frame["社区"].astype(str).str.strip()
    frame = frame.merge(poi, on="社区", how="left", indicator="_poi_merge")
    poi_numeric = [column for column in poi.columns if column not in {"社区", "poi_dominant_category"}]
    if include_building_decay_poi:
        building_decay = build_building_distance_decay_poi_frame(get_poi_json(radius), radius)
        frame["楼栋ID"] = frame["楼栋ID"].astype(str)
        frame = frame.merge(building_decay, on="楼栋ID", how="left")
        decay_numeric = [column for column in building_decay.columns if column != "楼栋ID"]
        poi_numeric.extend(decay_numeric)
    unmatched = frame["_poi_merge"].ne("both")
    frame.loc[unmatched, "poi_missing_indicator"] = 1
    frame[poi_numeric] = frame[poi_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame["poi_missing_indicator"] = frame["poi_missing_indicator"].fillna(1).astype(int)
    frame["poi_dominant_category"] = frame["poi_dominant_category"].fillna("MISSING").astype(str)
    frame = frame.drop(columns=["_poi_merge"])
    return frame, poi_numeric, "poi_dominant_category"


def candidates(quick: bool) -> list[Candidate]:
    if quick:
        return [
            Candidate(
                "RF_quick",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=6,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
                False,
            ),
            Candidate(
                "LR_quick",
                LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42),
                True,
            ),
        ]
    result = []
    for trees in (200, 400):
        for depth in (6, 12, None):
            for leaf in (1, 4, 8):
                result.append(
                    Candidate(
                        f"RF_n{trees}_d{depth}_l{leaf}",
                        RandomForestClassifier(
                            n_estimators=trees,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=1,
                        ),
                        False,
                    )
                )
    for trees in (100, 200):
        for rate in (0.03, 0.10):
            for depth in (2, 3):
                result.append(
                    Candidate(
                        f"GB_n{trees}_r{rate}_d{depth}",
                        GradientBoostingClassifier(
                            n_estimators=trees,
                            learning_rate=rate,
                            max_depth=depth,
                            random_state=42,
                        ),
                        False,
                        True,
                    )
                )
    for c_value in (0.01, 0.1, 1.0, 10.0):
        result.append(
            Candidate(
                f"LR_C{c_value}",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
                True,
            )
        )
    return result


def make_pipeline(candidate: Candidate, numeric: list[str], categorical: list[str]) -> Pipeline:
    numeric_steps = [("impute", SimpleImputer(strategy="constant", fill_value=0.0))]
    if candidate.scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", preprocess), ("classifier", clone(candidate.estimator))])


def fit_pipeline(model: Pipeline, candidate: Candidate, x: pd.DataFrame, y: np.ndarray) -> Pipeline:
    kwargs = {}
    if candidate.weighted_fit:
        positives = max(1, int((y == 1).sum()))
        negatives = max(1, int((y == 0).sum()))
        weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
        kwargs["classifier__sample_weight"] = weights
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(x, y, **kwargs)
    return model


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("分类器未产生二分类概率")
    result = np.clip(probabilities[:, 1], 1e-6, 1 - 1e-6)
    if not np.isfinite(result).all():
        raise ValueError("分类概率包含非有限值")
    return result


def oof_for_candidate(
    candidate: Candidate,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    numeric: list[str],
    categorical: list[str],
) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for train_idx, valid_idx in splits:
        model = make_pipeline(candidate, numeric, categorical)
        fit_pipeline(model, candidate, x.iloc[train_idx], y[train_idx])
        oof[valid_idx] = predict_probability(model, x.iloc[valid_idx])
    if np.isnan(oof).any():
        raise ValueError("OOF概率未覆盖全部样本")
    return oof


def evaluate_candidate(
    candidate: Candidate,
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    numeric: list[str],
    categorical: list[str],
) -> tuple[dict, TunedModel | None]:
    try:
        oof = oof_for_candidate(candidate, x, y, splits, numeric, categorical)
        threshold, score, precision = select_f1_threshold(y, oof)
        diagnostic = {
            "candidate": candidate.name,
            "status": "OK",
            "threshold": threshold,
            "f1": score,
            "precision": precision,
            "error": "",
        }
        return diagnostic, TunedModel(candidate, threshold, score, precision)
    except (ValueError, FloatingPointError, ConvergenceWarning) as exc:
        return {
            "candidate": candidate.name,
            "status": "FAILED",
            "threshold": np.nan,
            "f1": np.nan,
            "precision": np.nan,
            "error": str(exc),
        }, None


def tune_model(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    quick: bool,
    allow_global_fallback: bool = False,
    search_jobs: int = 1,
) -> tuple[TunedModel, list[dict], list[tuple[np.ndarray, np.ndarray]]]:
    split_decision = choose_group_splits(y, groups, (3, 2))
    if split_decision.splits is None:
        if not allow_global_fallback:
            raise ValueError(f"内层空间CV不可估计: {split_decision.reason}")
        minimum = int(np.bincount(y, minlength=2).min())
        n_splits = min(3, minimum)
        if n_splits < 2:
            raise ValueError("内层全局CV也不可估计")
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        inner_splits = list(splitter.split(np.zeros(len(y)), y))
    else:
        inner_splits = split_decision.splits
    specs = candidates(quick)
    if quick or search_jobs == 1:
        evaluated = [
            evaluate_candidate(candidate, x, y, inner_splits, numeric, categorical)
            for candidate in specs
        ]
    else:
        evaluated = Parallel(n_jobs=search_jobs, backend="loky")(
            delayed(evaluate_candidate)(candidate, x, y, inner_splits, numeric, categorical)
            for candidate in specs
        )
    diagnostics = []
    best: TunedModel | None = None
    for diagnostic, current in evaluated:
        diagnostics.append(diagnostic)
        if current is not None and (
            best is None
            or (current.inner_f1, current.inner_precision, -abs(current.threshold - 0.5), current.threshold)
            > (best.inner_f1, best.inner_precision, -abs(best.threshold - 0.5), best.threshold)
        ):
            best = current
    if best is None:
        raise ValueError("全部候选分类器均失败")
    best.fitted = make_pipeline(best.candidate, numeric, categorical)
    fit_pipeline(best.fitted, best.candidate, x, y)
    return best, diagnostics, inner_splits


class LeakageSafeLateFusion:
    def __init__(self, meta_c: float, threshold: float):
        self.meta_c = meta_c
        self.threshold = threshold
        self.vision_model: Pipeline | None = None
        self.poi_model: Pipeline | None = None
        self.meta_model: LogisticRegression | None = None

    @staticmethod
    def _meta(c_value: float) -> LogisticRegression:
        return LogisticRegression(
            penalty="l2",
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        )

    @classmethod
    def tune_and_fit(
        cls,
        x_vision: pd.DataFrame,
        x_poi: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        vision_tuned: TunedModel,
        poi_tuned: TunedModel,
        vision_numeric: list[str],
        poi_numeric: list[str],
        poi_categorical: list[str],
        inner_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple["LeakageSafeLateFusion", float, float, list[dict]]:
        vision_oof = oof_for_candidate(vision_tuned.candidate, x_vision, y, inner_splits, vision_numeric, [])
        poi_oof = oof_for_candidate(poi_tuned.candidate, x_poi, y, inner_splits, poi_numeric, poi_categorical)
        meta_x = np.clip(np.column_stack([vision_oof, poi_oof]), 1e-6, 1 - 1e-6)
        diagnostics = []
        best = None
        for c_value in (0.1, 1.0):
            try:
                meta_oof = np.full(len(y), np.nan)
                for train_idx, valid_idx in inner_splits:
                    model = cls._meta(c_value)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                        model.fit(meta_x[train_idx], y[train_idx])
                    meta_oof[valid_idx] = model.predict_proba(meta_x[valid_idx])[:, 1]
                if np.isnan(meta_oof).any() or not np.isfinite(meta_oof).all():
                    raise ValueError("元分类器OOF概率异常")
                threshold, score, precision = select_f1_threshold(y, meta_oof)
                diagnostics.append({"candidate": f"LateFusion_LR_C{c_value}", "status": "OK", "threshold": threshold, "f1": score, "precision": precision, "error": ""})
                rank = (score, precision, -abs(threshold - 0.5), threshold)
                if best is None or rank > best[0]:
                    best = (rank, c_value, threshold, score, precision)
            except (ValueError, FloatingPointError, ConvergenceWarning) as exc:
                diagnostics.append({"candidate": f"LateFusion_LR_C{c_value}", "status": "FAILED", "threshold": np.nan, "f1": np.nan, "precision": np.nan, "error": str(exc)})
        if best is None:
            raise ValueError("晚期融合元分类器全部失败")
        _, c_value, threshold, score, precision = best
        instance = cls(c_value, threshold)
        instance.vision_model = make_pipeline(vision_tuned.candidate, vision_numeric, [])
        fit_pipeline(instance.vision_model, vision_tuned.candidate, x_vision, y)
        instance.poi_model = make_pipeline(poi_tuned.candidate, poi_numeric, poi_categorical)
        fit_pipeline(instance.poi_model, poi_tuned.candidate, x_poi, y)
        instance.meta_model = cls._meta(c_value)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            instance.meta_model.fit(meta_x, y)
        coefficients = np.r_[instance.meta_model.coef_.ravel(), instance.meta_model.intercept_.ravel()]
        if not np.isfinite(coefficients).all():
            raise ValueError("元分类器系数包含非有限值")
        return instance, score, precision, diagnostics

    def predict_proba(self, x_vision: pd.DataFrame, x_poi: pd.DataFrame) -> np.ndarray:
        if self.vision_model is None or self.poi_model is None or self.meta_model is None:
            raise RuntimeError("晚期融合器尚未拟合")
        meta_x = np.column_stack([
            predict_probability(self.vision_model, x_vision),
            predict_probability(self.poi_model, x_poi),
        ])
        probabilities = self.meta_model.predict_proba(np.clip(meta_x, 1e-6, 1 - 1e-6))[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError("晚期融合概率包含非有限值")
        return np.clip(probabilities, 1e-6, 1 - 1e-6)


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(y))
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        if not mask.any():
            continue
        error += float(mask.sum()) / total * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return float(error)


def metrics(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    roc_auc = roc_auc_score(y, probability) if np.unique(y).size == 2 else np.nan
    pr_auc = average_precision_score(y, probability) if np.unique(y).size == 2 else np.nan
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
        "Accuracy": accuracy_score(y, prediction),
        "Balanced_Accuracy": balanced_accuracy_score(y, prediction),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Brier": brier_score_loss(y, probability),
        "ECE_10bin": expected_calibration_error(y, probability, bins=10),
    }


def shared_outer_splits(
    original_y: np.ndarray,
    version_y: np.ndarray,
    groups: np.ndarray,
    active_mask: np.ndarray,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int | None, str, str]:
    active_global = np.flatnonzero(active_mask)
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(active_global)}
    reasons = []
    for n_splits in (5, 4, 3, 2):
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            base = list(splitter.split(np.zeros(len(original_y)), original_y, groups))
            filtered = []
            for train_global, test_global in base:
                train = np.array([global_to_local[idx] for idx in train_global if idx in global_to_local], dtype=int)
                test = np.array([global_to_local[idx] for idx in test_global if idx in global_to_local], dtype=int)
                filtered.append((train, test))
            valid, reason = validate_precomputed_splits(version_y[active_mask], groups[active_mask], filtered)
            if not valid:
                raise ValueError(reason)
            return filtered, n_splits, "SPATIAL_GROUP_CV", ""
        except (ValueError, IndexError) as exc:
            reasons.append(f"{n_splits}折失败: {exc}")
    return None, None, "SPATIAL_CV_NOT_ESTIMABLE", " | ".join(reasons)


def global_fallback_splits(y: np.ndarray, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    minimum = int(np.bincount(y, minlength=2).min())
    n_splits = min(5, minimum)
    if n_splits < 2:
        raise ValueError("正负样本不足，连全局分层CV也无法运行")
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)), y))


def permute_poi_by_community(x_poi: pd.DataFrame, groups: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    result = x_poi.copy()
    unique = np.array(sorted(set(groups.astype(str))))
    shuffled = rng.permutation(unique)
    source = {target: donor for target, donor in zip(unique, shuffled)}
    first_rows = {group: x_poi.iloc[np.flatnonzero(groups.astype(str) == group)[0]].copy() for group in unique}
    for target in unique:
        mask = groups.astype(str) == target
        result.loc[mask, :] = first_rows[source[target]].to_numpy()
    return result


def fine_topk_from_training(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    candidates: list[str],
    top_k: int,
    min_communities: int = 8,
) -> list[str]:
    if not candidates or top_k <= 0:
        return []
    train = x_train[candidates].copy()
    train["_y"] = y_train
    train["_group"] = groups_train.astype(str)
    community = train.groupby("_group", as_index=False).mean(numeric_only=True)
    y = community["_y"].astype(float).to_numpy()
    if np.nanstd(y) == 0:
        return []
    ranked: list[tuple[float, str]] = []
    for column in candidates:
        x = community[column].astype(float).to_numpy()
        if np.nanstd(x) == 0:
            continue
        nonzero = int((community[column] > 0).sum())
        if nonzero < min_communities:
            continue
        rho, _ = spearmanr(x, y, nan_policy="omit")
        if np.isfinite(rho):
            ranked.append((abs(float(rho)), column))
    ranked.sort(reverse=True)
    return [column for _, column in ranked[:top_k]]


def selected_poi_columns_for_fold(
    profile: str,
    x_poi_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    poi_numeric: list[str],
) -> tuple[list[str], list[str]]:
    fine_columns = [column for column in poi_numeric if column.startswith("fine_")]
    base_columns = [column for column in poi_numeric if not column.startswith("fine_")]
    if profile == "fine_poi_top10":
        return [*base_columns, *fine_topk_from_training(x_poi_train, y_train, groups_train, fine_columns, 10)], fine_columns
    if profile == "fine_poi_top20":
        return [*base_columns, *fine_topk_from_training(x_poi_train, y_train, groups_train, fine_columns, 20)], fine_columns
    if profile == "fine_poi_top30":
        return [*base_columns, *fine_topk_from_training(x_poi_train, y_train, groups_train, fine_columns, 30)], fine_columns
    return poi_numeric, []


def run_class(
    frame: pd.DataFrame,
    class_name: str,
    gt_version: str,
    poi_numeric: list[str],
    poi_categorical: str | None,
    quick: bool,
    permutations: int,
    model_name: str,
    radius: int,
    search_jobs: int,
    poi_profile: str = "current",
    fusion_policy: str = "standard",
    outer_seed: int = 42,
    allow_global_fallback: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    original_y = frame[class_name].astype(int).to_numpy()
    label_status = frame[f"{class_name}_label_status"].fillna("").astype(str)
    auto_add_mask = label_status.isin(["AUTO_ADD", "AUTO_ADD_CANDIDATE"]).to_numpy()
    gt1_no_evidence_mask = label_status.eq("EXCLUDE_GT1_NO_EVIDENCE").to_numpy()
    if gt_version == "original":
        version_y = original_y.copy()
        active_mask = np.ones(len(frame), dtype=bool)
    elif gt_version == "ground_truth":
        version_y = frame[f"{class_name}"].astype(int).to_numpy()
        active_mask = frame[f"{class_name}_excluded"].astype(int).eq(0).to_numpy()
    elif gt_version == "add_only":
        version_y = original_y.copy()
        version_y[auto_add_mask] = 1
        active_mask = np.ones(len(frame), dtype=bool)
    elif gt_version == "gt1_no_evidence_exclude_only":
        version_y = original_y.copy()
        active_mask = ~gt1_no_evidence_mask
    elif gt_version == "add_plus_gt1_no_evidence_exclude":
        version_y = original_y.copy()
        version_y[auto_add_mask] = 1
        active_mask = ~gt1_no_evidence_mask
    else:
        raise ValueError(f"Unsupported gt_version: {gt_version}")
    groups_all = frame["社区"].fillna("UNKNOWN").astype(str).to_numpy()
    splits, n_splits, cv_type, reason = shared_outer_splits(
        original_y, version_y, groups_all, active_mask, outer_seed
    )
    active = frame.loc[active_mask].reset_index().rename(columns={"index": "_global_index"})
    y = version_y[active_mask]
    groups = groups_all[active_mask]
    if splits is None:
        if not allow_global_fallback:
            diagnostic = {
                "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                "Class": class_name, "Outer_Seed": outer_seed, "Fold": -1,
                "Stage": "outer_split", "Status": "SPATIAL_CV_NOT_ESTIMABLE",
                "Detail": reason, "Positive": int(y.sum()), "Negative": int((y == 0).sum()),
                "Positive_Communities": int(pd.Series(groups[y == 1]).nunique()),
            }
            return [], [], [diagnostic], []
        splits = global_fallback_splits(y, outer_seed)
        n_splits = len(splits)
        cv_type = "GLOBAL_STRATIFIED_CV"

    vision_numeric = [f"{class_name}_{suffix}" for suffix in VISION_SUFFIXES]
    poi_categorical_cols = [poi_categorical] if poi_categorical else []
    poi_cols = [*poi_numeric, *poi_categorical_cols]
    x_vision = active[vision_numeric].copy()
    x_poi = active[poi_cols].copy()
    x_full = pd.concat([x_vision, x_poi], axis=1)
    full_numeric = [*vision_numeric, *poi_numeric]
    fold_rows, prediction_rows, diagnostic_rows, permutation_rows = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(splits):
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "outer", "Status": "FAILED", "Detail": "训练或测试折缺少正负样本"})
            continue
        method_models = {}
        method_probabilities = {}
        method_thresholds = {}
        try:
            allow_global_inner = cv_type == "GLOBAL_STRATIFIED_CV"
            vision, diag, inner_splits = tune_model(
                x_vision.iloc[train_idx], y_train, groups_train, vision_numeric, [], quick,
                allow_global_inner, search_jobs
            )
            method_models["1_Vision_Only"] = vision
            method_probabilities["1_Vision_Only"] = predict_probability(vision.fitted, x_vision.iloc[test_idx])
            method_thresholds["1_Vision_Only"] = vision.threshold
            for item in diag:
                diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "vision_inner", "Status": item["status"], "Detail": json.dumps(item, ensure_ascii=False)})

            fold_poi_numeric, dropped_fine = selected_poi_columns_for_fold(
                poi_profile, x_poi.iloc[train_idx], y_train, groups_train, poi_numeric
            )
            fold_poi_cols = [*fold_poi_numeric, *poi_categorical_cols]
            fold_x_poi = x_poi[fold_poi_cols].copy()
            fold_x_full = pd.concat([x_vision, fold_x_poi], axis=1)
            fold_full_numeric = [*vision_numeric, *fold_poi_numeric]
            if dropped_fine:
                diagnostic_rows.append({
                    "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                    "Class": class_name, "Fold": fold, "Stage": "poi_feature_selection",
                    "Status": "OK",
                    "Detail": json.dumps(
                        {
                            "profile": poi_profile,
                            "selected_fine_features": [c for c in fold_poi_numeric if c.startswith("fine_")],
                            "candidate_fine_features": len(dropped_fine),
                        },
                        ensure_ascii=False,
                    ),
                })

            poi, diag, _ = tune_model(
                fold_x_poi.iloc[train_idx], y_train, groups_train, fold_poi_numeric, poi_categorical_cols, quick,
                allow_global_inner, search_jobs
            )
            method_models["2_POI_Only"] = poi
            method_probabilities["2_POI_Only"] = predict_probability(poi.fitted, fold_x_poi.iloc[test_idx])
            method_thresholds["2_POI_Only"] = poi.threshold
            for item in diag:
                diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "poi_inner", "Status": item["status"], "Detail": json.dumps(item, ensure_ascii=False)})

            baseline_train_probability = active.iloc[train_idx][f"{class_name}_box_count"].gt(0).astype(float).to_numpy()
            baseline_train_prediction = baseline_train_probability.astype(int)
            baseline_inner_score = f1_score(y_train, baseline_train_prediction, zero_division=0)
            baseline_inner_precision = precision_score(y_train, baseline_train_prediction, zero_division=0)
            baseline_probability = active.iloc[test_idx][f"{class_name}_box_count"].gt(0).astype(float).to_numpy()
            method_probabilities["0_Baseline"] = baseline_probability
            method_thresholds["0_Baseline"] = 0.5

            early, diag, _ = tune_model(
                fold_x_full.iloc[train_idx], y_train, groups_train, fold_full_numeric, poi_categorical_cols, quick,
                allow_global_inner, search_jobs
            )
            early_probability = predict_probability(early.fitted, fold_x_full.iloc[test_idx])
            for item in diag:
                diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "early_fusion_inner", "Status": item["status"], "Detail": json.dumps(item, ensure_ascii=False)})

            full_structure = "EARLY"
            full_probability = early_probability
            full_threshold = early.threshold
            full_model: object = early
            full_inner_score = early.inner_f1
            full_inner_precision = early.inner_precision
            try:
                late, late_score, late_precision, late_diag = LeakageSafeLateFusion.tune_and_fit(
                    x_vision.iloc[train_idx], fold_x_poi.iloc[train_idx], y_train, groups_train,
                    vision, poi, vision_numeric, fold_poi_numeric, poi_categorical_cols, inner_splits,
                )
                for item in late_diag:
                    diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "late_fusion_inner", "Status": item["status"], "Detail": json.dumps(item, ensure_ascii=False)})
                if (late_score, late_precision) > (early.inner_f1, early.inner_precision):
                    full_structure = "LATE"
                    full_probability = late.predict_proba(x_vision.iloc[test_idx], fold_x_poi.iloc[test_idx])
                    full_threshold = late.threshold
                    full_model = late
                    full_inner_score = late_score
                    full_inner_precision = late_precision
            except (ValueError, FloatingPointError, ConvergenceWarning) as exc:
                diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "late_fusion_inner", "Status": "FALLBACK_EARLY", "Detail": str(exc)})
            if fusion_policy in {"guarded", "baseline_guarded"} and (
                full_inner_score,
                full_inner_precision,
                -abs(full_threshold - 0.5),
                full_threshold,
            ) <= (
                vision.inner_f1,
                vision.inner_precision,
                -abs(vision.threshold - 0.5),
                vision.threshold,
            ):
                full_structure = "GUARDED_VISION"
                full_probability = method_probabilities["1_Vision_Only"]
                full_threshold = vision.threshold
                full_model = vision
                diagnostic_rows.append({
                    "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                    "Class": class_name, "Fold": fold, "Stage": "guarded_fusion_inner",
                    "Status": "USE_VISION",
                    "Detail": json.dumps(
                        {
                            "fusion_inner_f1": full_inner_score,
                            "fusion_inner_precision": full_inner_precision,
                            "vision_inner_f1": vision.inner_f1,
                            "vision_inner_precision": vision.inner_precision,
                        },
                        ensure_ascii=False,
                    ),
                })
            if fusion_policy == "baseline_guarded" and (
                baseline_inner_score,
                baseline_inner_precision,
                0.0,
                0.5,
            ) > (
                full_inner_score,
                full_inner_precision,
                -abs(full_threshold - 0.5),
                full_threshold,
            ):
                full_structure = "BASELINE_GUARDED_BASELINE"
                full_probability = baseline_probability
                full_threshold = 0.5
                full_model = None
                diagnostic_rows.append({
                    "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                    "Class": class_name, "Fold": fold, "Stage": "baseline_guarded_inner",
                    "Status": "USE_BASELINE",
                    "Detail": json.dumps(
                        {
                            "baseline_inner_f1": baseline_inner_score,
                            "baseline_inner_precision": baseline_inner_precision,
                            "fusion_inner_f1": full_inner_score,
                            "fusion_inner_precision": full_inner_precision,
                        },
                        ensure_ascii=False,
                    ),
                })
            if fusion_policy in {"selector", "selector_no_poi"}:
                selector_candidates = [
                    {
                        "name": "BASELINE",
                        "structure": "SELECTOR_BASELINE",
                        "score": baseline_inner_score,
                        "precision": baseline_inner_precision,
                        "threshold": 0.5,
                        "probability": baseline_probability,
                        "model": None,
                        "complexity": 0,
                    },
                    {
                        "name": "VISION",
                        "structure": "SELECTOR_VISION",
                        "score": vision.inner_f1,
                        "precision": vision.inner_precision,
                        "threshold": vision.threshold,
                        "probability": method_probabilities["1_Vision_Only"],
                        "model": vision,
                        "complexity": 1,
                    },
                    {
                        "name": full_structure,
                        "structure": f"SELECTOR_{full_structure}",
                        "score": full_inner_score,
                        "precision": full_inner_precision,
                        "threshold": full_threshold,
                        "probability": full_probability,
                        "model": full_model,
                        "complexity": 2,
                    },
                ]
                if fusion_policy == "selector":
                    selector_candidates.append(
                        {
                            "name": "POI",
                            "structure": "SELECTOR_POI",
                            "score": poi.inner_f1,
                            "precision": poi.inner_precision,
                            "threshold": poi.threshold,
                            "probability": method_probabilities["2_POI_Only"],
                            "model": poi,
                            "complexity": 1,
                        }
                    )
                selected = max(
                    selector_candidates,
                    key=lambda item: (
                        item["score"],
                        item["precision"],
                        -abs(item["threshold"] - 0.5),
                        item["threshold"],
                        -item["complexity"],
                    ),
                )
                previous_structure = full_structure
                full_structure = str(selected["structure"])
                full_probability = np.asarray(selected["probability"], dtype=float)
                full_threshold = float(selected["threshold"])
                full_model = selected["model"]
                diagnostic_rows.append({
                    "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                    "Class": class_name, "Fold": fold, "Stage": "method_selector_inner",
                    "Status": f"SELECT_{selected['name']}",
                    "Detail": json.dumps(
                        {
                            "selected": selected["name"],
                            "previous_fusion_structure": previous_structure,
                            "baseline_inner_f1": baseline_inner_score,
                            "baseline_inner_precision": baseline_inner_precision,
                            "vision_inner_f1": vision.inner_f1,
                            "vision_inner_precision": vision.inner_precision,
                            "poi_inner_f1": poi.inner_f1,
                            "poi_inner_precision": poi.inner_precision,
                            "fusion_inner_f1": full_inner_score,
                            "fusion_inner_precision": full_inner_precision,
                        },
                        ensure_ascii=False,
                    ),
                })
            method_models["3_Full_Fusion"] = full_model
            method_probabilities["3_Full_Fusion"] = full_probability
            method_thresholds["3_Full_Fusion"] = full_threshold

            for method, probability in method_probabilities.items():
                prediction = (probability >= method_thresholds[method]).astype(int)
                row_metrics = metrics(y_test, prediction, probability)
                fold_rows.append({
                    "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                    "Class": class_name, "Method": method, "Outer_Seed": outer_seed, "Fold": fold,
                    "CV_Type": cv_type, "Outer_Folds": n_splits, "Threshold": method_thresholds[method],
                    "Fusion_Structure": full_structure if method == "3_Full_Fusion" else "",
                    "Train_Size": len(train_idx), "Test_Size": len(test_idx), **row_metrics,
                })
                for position, local_idx in enumerate(test_idx):
                    prediction_rows.append({
                        "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                        "Class": class_name, "Method": method, "Outer_Seed": outer_seed,
                        "Fold": fold, "CV_Type": cv_type,
                        "楼栋ID": active.iloc[local_idx]["楼栋ID"], "社区": active.iloc[local_idx]["社区"],
                        "y_true": int(y_test[position]), "probability": float(probability[position]),
                        "threshold": float(method_thresholds[method]), "y_pred": int(prediction[position]),
                    })

            if permutations > 0:
                observed = f1_score(y_test, (full_probability >= full_threshold).astype(int), zero_division=0)
                rng = np.random.default_rng(stable_seed(model_name, radius, gt_version, class_name, outer_seed, fold))
                for repeat in range(permutations):
                    permuted_poi = permute_poi_by_community(fold_x_poi.iloc[test_idx].reset_index(drop=True), groups_test, rng)
                    if full_structure == "LATE":
                        perm_probability = full_model.predict_proba(x_vision.iloc[test_idx], permuted_poi)
                    elif full_structure in {"GUARDED_VISION", "SELECTOR_BASELINE", "SELECTOR_VISION"}:
                        perm_probability = full_probability
                    elif full_structure == "SELECTOR_POI":
                        perm_probability = predict_probability(full_model.fitted, permuted_poi)
                    else:
                        perm_full = pd.concat([x_vision.iloc[test_idx].reset_index(drop=True), permuted_poi], axis=1)
                        perm_probability = predict_probability(full_model.fitted, perm_full)
                    perm_f1 = f1_score(y_test, (perm_probability >= full_threshold).astype(int), zero_division=0)
                    permutation_rows.append({
                        "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                        "Class": class_name, "Outer_Seed": outer_seed, "Fold": fold, "Repeat": repeat,
                        "Fusion_Structure": full_structure, "Observed_F1": observed,
                        "Permuted_F1": perm_f1, "Delta_F1": observed - perm_f1,
                    })
        except Exception as exc:
            diagnostic_rows.append({"Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version, "Class": class_name, "Fold": fold, "Stage": "class_fold", "Status": "FAILED", "Detail": repr(exc)})

    diagnostic_rows.append({
        "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
        "Class": class_name, "Outer_Seed": outer_seed, "Fold": -1,
        "Stage": "outer_split", "Status": cv_type,
        "Detail": reason, "Positive": int(y.sum()), "Negative": int((y == 0).sum()),
        "Positive_Communities": int(pd.Series(groups[y == 1]).nunique()),
    })
    return fold_rows, prediction_rows, diagnostic_rows, permutation_rows


def combinations(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.formal_suite:
        return [("yolov9_1280_balanced", radius) for radius in (500, 1000, 1500)] + [
            (model, 1000) for model in MODEL_NAMES if model != "yolov9_1280_balanced"
        ]
    return [(args.model_name, args.poi_radius)]


def select_poi_feature_profile(
    profile: str,
    poi_numeric: list[str],
    poi_categorical: str | None,
) -> tuple[list[str], str | None]:
    if profile == "current":
        return poi_numeric, poi_categorical
    if profile == "current_no_missing":
        return [column for column in poi_numeric if column != "poi_missing_indicator"], poi_categorical
    if profile == "current_core_no_handcrafted":
        core = ["total_poi", *[f"poi_{category}_ratio" for category in POI_CATEGORIES]]
        missing = [column for column in core if column not in poi_numeric]
        if missing:
            raise ValueError(f"current_core_no_handcrafted缺少POI列: {missing}")
        return core, None
    if profile in {"fine_poi_top10", "fine_poi_top20", "fine_poi_top30"}:
        fine_columns = [column for column in poi_numeric if column.startswith("fine_")]
        if not fine_columns:
            raise ValueError(f"{profile}需要细粒度POI特征，但当前POI表没有fine_列")
        base = [column for column in BASE_POI_NUMERIC if column in poi_numeric]
        return [*base, *fine_columns], poi_categorical
    if profile == "building_distance_decay_poi":
        decay_columns = [column for column in poi_numeric if column.startswith("building_poi_")]
        if not decay_columns:
            raise ValueError("building_distance_decay_poi需要楼栋级距离衰减POI列")
        base = [column for column in BASE_POI_NUMERIC if column in poi_numeric]
        return [*base, *decay_columns], poi_categorical
    if profile == "building_distance_decay_only":
        decay_columns = [column for column in poi_numeric if column.startswith("building_poi_")]
        if not decay_columns:
            raise ValueError("building_distance_decay_only需要楼栋级距离衰减POI列")
        return decay_columns, None
    raise ValueError(f"未知POI特征组: {profile}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe nested spatial ablation evaluation.")
    parser.add_argument("--model-name", default="yolov9_1280_balanced")
    parser.add_argument("--poi-radius", type=int, default=1000)
    parser.add_argument(
        "--vision-feature-profile",
        choices=sorted(VISION_FEATURE_PROFILES),
        default="all6",
        help="YOLO building-level visual aggregate feature set.",
    )
    parser.add_argument(
        "--poi-feature-profile",
        choices=[
            "current",
            "current_no_missing",
            "current_core_no_handcrafted",
            "fine_poi_top10",
            "fine_poi_top20",
            "fine_poi_top30",
            "building_distance_decay_poi",
            "building_distance_decay_only",
        ],
        default="current",
        help="POI feature set. fine_poi_topK selects detailed AMap POI types inside each training fold.",
    )
    parser.add_argument(
        "--fusion-policy",
        choices=["standard", "guarded", "selector", "selector_no_poi", "baseline_guarded"],
        default="standard",
        help="guarded falls back to Vision; selector variants choose methods using training-fold evidence.",
    )
    parser.add_argument(
        "--gt-versions",
        nargs="+",
        choices=[
            "original",
            "add_only",
            "gt1_no_evidence_exclude_only",
            "add_plus_gt1_no_evidence_exclude",
            "ground_truth",
        ],
        default=["original"],
    )
    parser.add_argument("--classes", nargs="+", choices=CLASS_NAMES, default=CLASS_NAMES)
    parser.add_argument("--poi-permutations", type=int, default=100)
    parser.add_argument(
        "--outer-seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Repeated outer StratifiedGroupKFold seeds. Formal reruns should use at least five fixed seeds.",
    )
    parser.add_argument(
        "--allow-global-fallback",
        action="store_true",
        help="Allow non-spatial CV when grouped CV is impossible. Do not use for formal manuscript results.",
    )
    parser.add_argument("--quick", action="store_true", help="Reduced candidate grid for smoke testing only.")
    parser.add_argument("--formal-suite", action="store_true")
    parser.add_argument(
        "--search-jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of candidate configurations evaluated in parallel.",
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if args.search_jobs < 1:
        parser.error("--search-jobs必须至少为1")
    global VISION_SUFFIXES
    VISION_SUFFIXES = VISION_FEATURE_PROFILES[args.vision_feature_profile]
    ensure_fusion_dirs()
    require = any(version != "original" for version in args.gt_versions)
    if require and not labelED_GT.exists():
        raise FileNotFoundError(f"所选GT版本需要先运行build_auto_gt.py: {labelED_GT}")
    run_name = args.run_name or f"nested_{dt.datetime.now():%Y%m%d_%H%M%S}"
    output = FUSION_RESULTS / run_name
    output.mkdir(parents=True, exist_ok=False)
    all_folds, all_predictions, all_diagnostics, all_permutations = [], [], [], []
    for model_name, radius in combinations(args):
        print(f"=== {model_name} / {radius}m ===", flush=True)
        include_fine_poi = args.poi_feature_profile.startswith("fine_poi_")
        include_building_decay_poi = args.poi_feature_profile.startswith("building_distance_decay")
        frame, poi_numeric, poi_categorical = load_feature_frame(
            model_name,
            radius,
            include_fine_poi=include_fine_poi,
            include_building_decay_poi=include_building_decay_poi,
            require=require,
        )
        poi_numeric, poi_categorical = select_poi_feature_profile(
            args.poi_feature_profile, poi_numeric, poi_categorical
        )
        for outer_seed in args.outer_seeds:
            for gt_version in args.gt_versions:
                for class_name in args.classes:
                    print(f"[seed={outer_seed}][{gt_version}] {class_name}", flush=True)
                    try:
                        folds, predictions, diagnostics, permutations = run_class(
                            frame, class_name, gt_version, poi_numeric, poi_categorical,
                            args.quick, args.poi_permutations, model_name, radius, args.search_jobs,
                            args.poi_feature_profile, args.fusion_policy, outer_seed,
                            args.allow_global_fallback,
                        )
                        all_folds.extend(folds)
                        all_predictions.extend(predictions)
                        all_diagnostics.extend(diagnostics)
                        all_permutations.extend(permutations)
                    except Exception as exc:
                        all_diagnostics.append({
                            "Model": model_name, "POI_Radius_m": radius, "GT_Version": gt_version,
                            "Class": class_name, "Outer_Seed": outer_seed, "Fold": -1,
                            "Stage": "class", "Status": "FAILED", "Detail": repr(exc),
                        })
                        print(f"  类别失败但继续: {exc}", flush=True)
    pd.DataFrame(all_folds).to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_frame = pd.DataFrame(all_predictions)
    prediction_output = output / "building_oof_predictions.parquet"
    try:
        prediction_frame.to_parquet(prediction_output, index=False)
    except ImportError:
        prediction_output = output / "building_oof_predictions.csv.gz"
        prediction_frame.to_csv(prediction_output, index=False, encoding="utf-8-sig", compression="gzip")
    pd.DataFrame(all_diagnostics).to_csv(output / "diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_permutations).to_csv(output / "poi_permutations.csv", index=False, encoding="utf-8-sig")
    atomic_json_write(
        output / "run_config.json",
        {
            "run_name": run_name,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "arguments": vars(args),
            "combinations": combinations(args),
            "quick_mode": args.quick,
            "building_predictions": str(prediction_output),
            "note": "quick_mode results are smoke tests and must not be reported as formal results" if args.quick else "formal nested evaluation",
        },
    )
    print(f"结果目录: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

