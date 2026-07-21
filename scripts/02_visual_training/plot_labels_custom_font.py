#!/usr/bin/env python3
"""按 Ultralytics 原版 plot_labels 的数据与绘图逻辑，分别保存四张子图（非裁剪合成图）。

要求：
1. 不是简单的截取，而是通过读取 annotations 数据重新绘制。
2. 字体设置为 Times New Roman (新罗马字体), 13号。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.plotting import colors


def _load_train_boxes_cls(data_yaml: Path):
    data_dict = check_det_dataset(str(data_yaml.resolve()))
    cfg = deepcopy(DEFAULT_CFG)
    cfg.imgsz = 640
    cfg.cache = False
    cfg.rect = False
    cfg.task = "detect"
    cfg.fraction = 1.0
    ds = build_yolo_dataset(cfg, data_dict["train"], batch=16, data=data_dict, mode="train", stride=32)
    boxes = np.concatenate([lb["bboxes"] for lb in ds.labels], 0)
    cls = np.concatenate([lb["cls"] for lb in ds.labels], 0).squeeze()
    names = data_dict["names"]
    return boxes, cls, names


def plot_four_like_ultralytics(boxes: np.ndarray, cls: np.ndarray, names: dict, save_dir: Path) -> list[Path]:
    """与 plot_labels() 相同数值与画法，拆成四个独立文件。
    满足字体：新罗马，13号 的要求。
    """
    import matplotlib.pyplot as plt
    import polars
    from matplotlib.colors import LinearSegmentedColormap
    
    # 字体与字号设置 (统一由这里全局控制，不要在下方硬编码)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    nc = int(cls.max() + 1)
    boxes = boxes[:1000000]
    cls = cls[:1000000]
    x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])
    subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

    out_paths: list[Path] = []

    # ---- ax[0]: class histogram ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    y = ax.hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([c / 255 for c in colors(i)])
    ax.set_ylabel("instances")
    
    if 0 < len(names) < 30:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(list(names.values()), rotation=90)
        ax.bar_label(y[2])
    else:
        ax.set_xlabel("classes")
    for s in {"top", "right", "left", "bottom"}:
        ax.spines[s].set_visible(False)
    p0 = save_dir / "labels_01_class_instances.jpg"
    fig.savefig(p0, dpi=200)
    plt.close(fig)
    out_paths.append(p0)

    # ---- ax[1]: bbox overlay (first 500) ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    b_draw = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for class_id, box in zip(cls[:500], b_draw[:500]):
        ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(class_id))
    ax.imshow(img)
    ax.axis("off")
    for s in {"top", "right", "left", "bottom"}:
        ax.spines[s].set_visible(False)
    p1 = save_dir / "labels_02_bbox_overlay.jpg"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    out_paths.append(p1)

    # ---- ax[2]: center xy hist2d ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    ax.hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    for s in {"top", "right", "left", "bottom"}:
        ax.spines[s].set_visible(False)
    p2 = save_dir / "labels_03_center_xy_hist.jpg"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    out_paths.append(p2)

    # ---- ax[3]: width vs height hist2d ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    ax.hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    for s in {"top", "right", "left", "bottom"}:
        ax.spines[s].set_visible(False)
    p3 = save_dir / "labels_04_width_height_hist.jpg"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    out_paths.append(p3)

    return out_paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets_balanced" / "data.yaml",
        help="与训练相同的 data.yaml",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "labels_separate_output",
        help="输出结果保存的目录",
    )
    args = p.parse_args()

    boxes, cls, names = _load_train_boxes_cls(args.data)
    paths = plot_four_like_ultralytics(boxes, cls, names, args.save_dir)
    for path in paths:
        print(f"已生成: {path}")


if __name__ == "__main__":
    main()

