import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from new_experiment.export_ablation_building_tables import (
    METHOD_OUTPUT_NAMES,
    build_complete_classification_table,
    get_cached_params,
    should_skip_combo_for_missing_cache,
    output_path_for,
    validate_output_table,
)


class ExportAblationBuildingTablesTests(unittest.TestCase):
    def test_output_path_for_uses_method_prefix_and_model_radius(self):
        out = output_path_for(Path("out"), "yolov8_balanced", 500, "1_Vision_Only_RF")

        self.assertEqual(
            out,
            Path("out")
            / "yolov8_balanced_500m"
            / "1_Vision_Only_RF完整分类表_yolov8_balanced_500m.xlsx",
        )
        self.assertEqual(len(METHOD_OUTPUT_NAMES), 4)

    def test_build_complete_classification_table_preserves_template_layout(self):
        template = pd.DataFrame(
            {
                "楼栋ID": [101, 102],
                "区市": ["A区", "B区"],
                "谷歌经纬度": ["120,36", "121,37"],
                "wall_damage": [0, 0],
                "elevator_addition": [1, 1],
            }
        )
        predictions = pd.DataFrame(
            {
                "楼栋ID": [102, 101],
                "wall_damage": [1, 0],
                "elevator_addition": [0, 1],
            }
        )

        out = build_complete_classification_table(
            template,
            predictions,
            ["wall_damage", "elevator_addition"],
        )

        self.assertEqual(list(out.columns), list(template.columns))
        self.assertEqual(out["楼栋ID"].tolist(), [101, 102])
        self.assertEqual(out["区市"].tolist(), ["A区", "B区"])
        self.assertEqual(out["wall_damage"].tolist(), [0, 1])
        self.assertEqual(out["elevator_addition"].tolist(), [1, 0])
        validate_output_table(out, template, ["wall_damage", "elevator_addition"])

    def test_cached_params_lookup_and_missing_combo_policy(self):
        params = {
            "max_depth": 4,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 150,
        }
        class_params = {
            cls: params
            for cls in [
                "pipeline_damage",
                "addition_balconies_windows",
                "public_spaces_illegal_occupation",
                "no_aging_place_modifications",
                "wall_damage",
                "illegal_renovation_expansion",
                "elevator_addition",
            ]
        }
        cache = {
            "combos": {
                "yolov8_balanced": {
                    "500": class_params
                }
            }
        }

        params = get_cached_params(cache, "yolov8_balanced", 500, "wall_damage")

        self.assertEqual(params["max_depth"], 4)
        self.assertTrue(
            should_skip_combo_for_missing_cache(cache, "yolov9_1280_balanced", 1500)
        )
        self.assertFalse(
            should_skip_combo_for_missing_cache(cache, "yolov8_balanced", 500)
        )


if __name__ == "__main__":
    unittest.main()

