"""
Compatibility wrapper.

Use evaluate_ablation.py for new runs. This wrapper runs the formal balanced
YOLOv8 experiment with the 1500m POI file.
"""

from evaluate_ablation import run_ablation


if __name__ == "__main__":
    run_ablation("yolov8_balanced", 1500)

