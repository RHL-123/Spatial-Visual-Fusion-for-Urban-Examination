from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold

from experiment_registry import CLASS_NAMES, NEW_EXP_DIR, PROJECT_ROOT


FUSION_ROOT = NEW_EXP_DIR / "fusion_evaluation"
FUSION_TABLES = FUSION_ROOT / "tables"
FUSION_RESULTS = FUSION_ROOT / "results"
FUSION_REPORTS = FUSION_ROOT / "reports"
EXPECTED_GT_SHA256 = "4071cf85e1f62189e8405e97bd45b8e1f9f729bd9f3077dcccf7d50ceb81ab13"
MODEL_NAMES = [
    "yolov8_balanced",
    "yolov9_640_balanced",
    "yolov9_1280_balanced",
    "yolov10_balanced",
    "yolo11_balanced",
    "rtdetr_balanced",
]
PRIMARY_MODEL = "yolov9_1280_balanced"


def ensure_fusion_dirs() -> None:
    for path in (FUSION_ROOT, FUSION_TABLES, FUSION_RESULTS, FUSION_REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def file_hashes(path: Path) -> dict[str, str | int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def assert_original_gt(path: Path) -> dict[str, str | int]:
    hashes = file_hashes(path)
    if hashes["sha256"] != EXPECTED_GT_SHA256:
        raise ValueError(
            "原始GT SHA-256不匹配，已拒绝继续。"
            f" expected={EXPECTED_GT_SHA256}, actual={hashes['sha256']}"
        )
    return hashes


def normalize_building_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("楼栋ID为空")
    text = unicodedata.normalize("NFKC", str(value)).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"楼栋ID不是数字: {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"楼栋ID不是整数: {value!r}")
    return str(int(number))


def normalize_class_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    aliases = {name.lower(): name for name in CLASS_NAMES}
    if text not in aliases:
        raise ValueError(f"未知类别: {value!r}")
    return aliases[text]


def normalize_image_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].casefold()


def atomic_excel_write(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        with pd.ExcelWriter(temp, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass
class SplitDecision:
    splits: list[tuple[np.ndarray, np.ndarray]] | None
    n_splits: int | None
    status: str
    reason: str


def _validate_binary_fold(
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> str | None:
    if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
        return "训练集或测试集缺少正负类别"
    train_groups = groups[train_idx]
    y_train = y[train_idx]
    positive_groups = np.unique(train_groups[y_train == 1]).size
    negative_groups = np.unique(train_groups[y_train == 0]).size
    if positive_groups < 2 or negative_groups < 2:
        return f"训练集阳性/阴性社区不足: positive={positive_groups}, negative={negative_groups}"
    if set(train_groups).intersection(set(groups[test_idx])):
        return "训练集和测试集社区重叠"
    return None


def choose_group_splits(
    y: Iterable[int],
    groups: Iterable[object],
    candidates: Iterable[int],
    random_state: int = 42,
) -> SplitDecision:
    y_array = np.asarray(list(y), dtype=int)
    group_array = np.asarray([str(item) for item in groups], dtype=object)
    failures: list[str] = []
    for n_splits in candidates:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            splits = list(splitter.split(np.zeros(len(y_array)), y_array, group_array))
            for fold, (train_idx, test_idx) in enumerate(splits):
                reason = _validate_binary_fold(y_array, group_array, train_idx, test_idx)
                if reason:
                    raise ValueError(f"fold={fold}: {reason}")
            return SplitDecision(splits, n_splits, "OK", "")
        except ValueError as exc:
            failures.append(f"{n_splits}折失败: {exc}")
    return SplitDecision(None, None, "SPATIAL_CV_NOT_ESTIMABLE", " | ".join(failures))


def validate_precomputed_splits(
    y: Iterable[int],
    groups: Iterable[object],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[bool, str]:
    y_array = np.asarray(list(y), dtype=int)
    group_array = np.asarray([str(item) for item in groups], dtype=object)
    try:
        for fold, (train_idx, test_idx) in enumerate(splits):
            reason = _validate_binary_fold(y_array, group_array, train_idx, test_idx)
            if reason:
                return False, f"fold={fold}: {reason}"
    except (IndexError, ValueError) as exc:
        return False, str(exc)
    return True, ""


def select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    best: tuple[float, float, float, float] | None = None
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        prediction = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, prediction, zero_division=0)
        precision = precision_score(y_true, prediction, zero_division=0)
        score = (f1, precision, -abs(threshold - 0.5), threshold)
        if best is None or score > best:
            best = score
    assert best is not None
    return float(best[3]), float(best[0]), float(best[1])


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")

