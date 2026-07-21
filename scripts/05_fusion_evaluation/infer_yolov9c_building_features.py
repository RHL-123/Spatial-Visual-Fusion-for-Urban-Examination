"""
Compatibility wrapper.

Use infer_building_features.py for new runs. This wrapper keeps the historical
YOLOv9c high-resolution setting as yolov9_1280_balanced.
"""

from infer_building_features import run_inference


if __name__ == "__main__":
    run_inference("yolov9_1280_balanced", device="auto")

