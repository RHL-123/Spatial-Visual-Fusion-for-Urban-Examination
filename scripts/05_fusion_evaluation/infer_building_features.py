"""
Unified building-level inference for the balanced detector experiments.

The model weights are resolved from model_registry.yaml. This prevents
accidentally using non-balanced runs through ambiguous names such as yolo11.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_registry import (
    CLASS_NAMES,
    IMAGES_DIR,
    add_model_name_argument,
    ensure_output_dirs,
    get_model_config,
)

CONF_THRESH = 0.15


def resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "cpu":
        print("使用推理设备: cpu")
        return "cpu"

    try:
        import torch
    except Exception as exc:
        if requested in {"auto", ""}:
            print(f"无法导入 torch，改用 CPU: {exc}")
            return "cpu"
        raise RuntimeError(f"请求 device={device}，但无法导入 torch 检查 CUDA: {exc}") from exc

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {cuda_count}")
    print(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if requested in {"auto", ""}:
        resolved = "0" if cuda_available and cuda_count > 0 else "cpu"
        print(f"使用推理设备: {resolved}")
        return resolved

    if requested != "cpu" and not cuda_available:
        print(f"警告: 请求 device={device}，但 CUDA 当前不可用，改用 CPU。")
        return "cpu"
    print(f"使用推理设备: {device}")
    return str(device)


def parse_filename(filename: str):
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")
    try:
        building_id = int(parts[-3])
    except (IndexError, ValueError):
        return None, None, None, None
    try:
        district = parts[0]
        street = parts[1]
        community = "_".join(parts[2:-3])
    except IndexError:
        district, street, community = "", "", ""
    return building_id, district, street, community


def generate_data_exports(output_excel: Path, raw_csv: Path, building_data, detailed_boxes, quiet=False):
    if not building_data:
        if not quiet:
            print("没有可导出的数据。")
        return

    df = pd.DataFrame(list(building_data.values()))
    for cls in CLASS_NAMES:
        df[f"{cls}_mean_conf"] = df.apply(
            lambda row: row[f"{cls}_sum_conf"] / row[f"{cls}_box_count"]
            if row[f"{cls}_box_count"] > 0
            else 0.0,
            axis=1,
        )
        df[f"{cls}_median_conf"] = df.apply(
            lambda row: float(np.median(row[f"{cls}_all_confs"]))
            if f"{cls}_all_confs" in row
            and isinstance(row[f"{cls}_all_confs"], list)
            and len(row[f"{cls}_all_confs"]) > 0
            else 0.0,
            axis=1,
        )
        df[f"{cls}_std_conf"] = df.apply(
            lambda row: float(np.std(row[f"{cls}_all_confs"]))
            if f"{cls}_all_confs" in row
            and isinstance(row[f"{cls}_all_confs"], list)
            and len(row[f"{cls}_all_confs"]) > 0
            else 0.0,
            axis=1,
        )
        df[f"{cls}_infection_rate"] = df.apply(
            lambda row: row[f"{cls}_infected_imgs"] / row["照片总数"] if row["照片总数"] > 0 else 0.0,
            axis=1,
        )
        df[cls] = df[f"{cls}_box_count"].apply(lambda x: 1 if x > 0 else 0)

    base_cols = ["楼栋ID", "区市", "街道", "社区", "照片总数"]
    ordered_cols = base_cols.copy()
    for cls in CLASS_NAMES:
        ordered_cols.extend(
            [
                cls,
                f"{cls}_box_count",
                f"{cls}_infected_imgs",
                f"{cls}_infection_rate",
                f"{cls}_max_conf",
                f"{cls}_mean_conf",
                f"{cls}_median_conf",
                f"{cls}_std_conf",
            ]
        )

    df = df[[c for c in ordered_cols if c in df.columns]]
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)
    if not quiet:
        print(f"楼栋级特征已保存至: {output_excel}")

    if detailed_boxes:
        pd.DataFrame(detailed_boxes).to_csv(raw_csv, index=False, encoding="utf-8-sig")
        if not quiet:
            print(f"明细级框级数据CSV已保存至: {raw_csv}")


def _load_detector(cfg):
    if cfg.arch == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR(str(cfg.weights))
    if cfg.arch == "yolo":
        from ultralytics import YOLO

        return YOLO(str(cfg.weights))
    raise ValueError(f"不支持的 arch: {cfg.arch}")


def run_inference(model_name: str, device: str, skip_existing: bool = False):
    cfg = get_model_config(model_name)
    ensure_output_dirs()
    resolved_device = resolve_device(device)
    use_half = resolved_device != "cpu"

    if skip_existing and cfg.prediction_excel.exists() and cfg.detailed_boxes_csv.exists():
        print(f"已存在预测结果，跳过: {cfg.prediction_excel}")
        return {}, []

    print(f"加载模型: {cfg.name} -> {cfg.weights}")
    print(f"输出Excel: {cfg.prediction_excel}")
    print(f"输出CSV: {cfg.detailed_boxes_csv}")
    model = _load_detector(cfg)

    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"找到 {len(image_files)} 张待预测图片")
    print(f"模型: {cfg.display_name}, imgsz={cfg.imgsz}, batch={cfg.batch}, conf={CONF_THRESH}, device={resolved_device}")

    building_data = {}
    detailed_boxes = []

    for i in range(0, len(image_files), cfg.batch):
        batch_files = image_files[i : i + cfg.batch]
        batch_paths = [str(IMAGES_DIR / f) for f in batch_files]

        try:
            results = model.predict(
                batch_paths,
                verbose=False,
                conf=CONF_THRESH,
                half=use_half,
                device=resolved_device,
                imgsz=cfg.imgsz,
            )
        except Exception as exc:
            raise RuntimeError(f"Batch推理错误，已停止以避免生成空结果: {exc}") from exc

        for img_file, result in zip(batch_files, results):
            building_id, district, street, community = parse_filename(img_file)
            if building_id is None:
                continue

            if building_id not in building_data:
                building_data[building_id] = {
                    "楼栋ID": building_id,
                    "区市": district,
                    "街道": street,
                    "社区": community,
                    "照片总数": 0,
                }
                for cls in CLASS_NAMES:
                    building_data[building_id][f"{cls}_box_count"] = 0
                    building_data[building_id][f"{cls}_max_conf"] = 0.0
                    building_data[building_id][f"{cls}_sum_conf"] = 0.0
                    building_data[building_id][f"{cls}_infected_imgs"] = 0

            building_data[building_id]["照片总数"] += 1
            img_detected_classes = set()

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if 0 <= cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                    building_data[building_id][f"{cls_name}_box_count"] += 1
                    building_data[building_id][f"{cls_name}_sum_conf"] += conf
                    building_data[building_id][f"{cls_name}_max_conf"] = max(
                        building_data[building_id][f"{cls_name}_max_conf"], conf
                    )
                    building_data[building_id].setdefault(f"{cls_name}_all_confs", []).append(conf)

                    detailed_boxes.append(
                        {
                            "图片名": img_file,
                            "楼栋ID": building_id,
                            "区市": district,
                            "街道": street,
                            "社区": community,
                            "缺陷类别": cls_name,
                            "置信度": conf,
                        }
                    )
                    img_detected_classes.add(cls_name)

            for cls_name in img_detected_classes:
                building_data[building_id][f"{cls_name}_infected_imgs"] += 1

        processed = min(i + cfg.batch, len(image_files))
        if processed % 100 < cfg.batch or processed == len(image_files):
            print(f"已处理 {processed}/{len(image_files)} 张...")

        if (i // cfg.batch + 1) % 5 == 0:
            generate_data_exports(cfg.prediction_excel, cfg.detailed_boxes_csv, building_data, detailed_boxes, quiet=True)

    print("推理完成。")
    generate_data_exports(cfg.prediction_excel, cfg.detailed_boxes_csv, building_data, detailed_boxes, quiet=False)
    return building_data, detailed_boxes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run balanced detector inference on dataset_predict.")
    add_model_name_argument(parser)
    parser.add_argument("--device", default="auto", help="Ultralytics device argument, default: auto")
    parser.add_argument("--skip-existing", action="store_true", help="已有Excel和CSV时跳过推理")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_inference(args.model_name, device=args.device, skip_existing=args.skip_existing)
    except Exception as exc:
        print(f"错误: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

