from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_utils import (
    EXPECTED_GT_SHA256,
    assert_original_gt,
    choose_group_splits,
    normalize_building_id,
    normalize_class_name,
    normalize_image_name,
)
from build_auto_gt import OUTPUT_GT
from evaluate_ablation_nested import build_poi_frame_with_missing
from experiment_registry import CLASS_NAMES, GT_EXCEL
from summarize_ablation import collapse_repeated_predictions


class NormalizationTests(unittest.TestCase):
    def test_identifiers(self):
        self.assertEqual(normalize_building_id(" 45219.0 "), "45219")
        self.assertEqual(normalize_class_name(" WALL_DAMAGE "), "wall_damage")
        self.assertEqual(normalize_image_name(r"A\B\Photo.JPG"), "photo.jpg")
        with self.assertRaises(ValueError):
            normalize_building_id("12.5")

    def test_original_gt_hash(self):
        self.assertEqual(assert_original_gt(GT_EXCEL)["sha256"], EXPECTED_GT_SHA256)


class SplitTests(unittest.TestCase):
    def test_group_split_has_no_overlap(self):
        groups = np.repeat([f"g{i}" for i in range(10)], 4)
        y = np.tile([0, 0, 1, 1], 10)
        decision = choose_group_splits(y, groups, (5, 4, 3, 2))
        self.assertEqual(decision.status, "OK")
        self.assertIsNotNone(decision.splits)
        for train, test in decision.splits or []:
            self.assertFalse(set(groups[train]).intersection(set(groups[test])))

    def test_impossible_split_is_reported(self):
        groups = np.array(["positive"] * 3 + ["negative"] * 9)
        y = np.array([1] * 3 + [0] * 9)
        decision = choose_group_splits(y, groups, (3, 2))
        self.assertEqual(decision.status, "SPATIAL_CV_NOT_ESTIMABLE")


class PoiTests(unittest.TestCase):
    def test_missing_indicator_distinguishes_real_zero(self):
        payload = {
            "valid_zero": {"total_count": 0, "category_distribution": {}},
            "invalid": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poi.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            frame = build_poi_frame_with_missing(path).set_index("社区")
        self.assertEqual(int(frame.loc["valid_zero", "poi_missing_indicator"]), 0)
        self.assertEqual(int(frame.loc["invalid", "poi_missing_indicator"]), 1)


class OutputTests(unittest.TestCase):
    def test_gt_preserves_original_columns(self):
        self.assertTrue(OUTPUT_GT.exists())
        original = pd.read_excel(GT_EXCEL)
        ground_truth = pd.read_excel(OUTPUT_GT, sheet_name="GT")
        pd.testing.assert_frame_equal(
            original.reset_index(drop=True),
            ground_truth[original.columns].reset_index(drop=True),
            check_dtype=False,
        )
        expected = {f"{name}" for name in CLASS_NAMES}
        self.assertTrue(expected.issubset(ground_truth.columns))

    def test_repeated_predictions_are_collapsed(self):
        rows = []
        for probability in (0.4, 0.8):
            rows.append(
                {
                    "Model": "m", "POI_Radius_m": 1000, "GT_Version": "original",
                    "Class": "c", "Method": "method", "CV_Type": "GLOBAL_STRATIFIED_CV",
                    "楼栋ID": 1, "社区": "g", "y_true": 1, "probability": probability,
                    "threshold": 0.5, "y_pred": int(probability >= 0.5), "Fold": 0,
                }
            )
        collapsed = collapse_repeated_predictions(pd.DataFrame(rows))
        self.assertEqual(len(collapsed), 1)
        self.assertAlmostEqual(float(collapsed.iloc[0]["probability"]), 0.6)
        self.assertEqual(int(collapsed.iloc[0]["y_pred"]), 1)


if __name__ == "__main__":
    unittest.main()

