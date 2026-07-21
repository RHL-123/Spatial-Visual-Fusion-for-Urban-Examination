"""
Compatibility wrapper.

Use evaluate_ablation.py for new runs. This wrapper runs the historical
YOLOv9c high-resolution balanced setting with the 1500m POI file.
"""

from evaluate_ablation import run_ablation


if __name__ == "__main__":
    run_ablation("yolov9_1280_balanced", 1500)

