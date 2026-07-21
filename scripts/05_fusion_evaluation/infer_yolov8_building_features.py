"""
Compatibility wrapper.

Use infer_building_features.py for new runs. This wrapper now points to the
formal balanced YOLOv8 experiment instead of the old exp2 unbalanced run.
"""

from infer_building_features import run_inference


if __name__ == "__main__":
    run_inference("yolov8_balanced", device="auto")

