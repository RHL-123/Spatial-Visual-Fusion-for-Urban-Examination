# `labels.jpg` 是什么、从哪里来

## 生成位置与时机

在 **Ultralytics YOLO 检测训练**（如 `YOLO(...).train(...)`）启动后，数据管线扫描完 `train`/`val` 的标签缓存时，会调用：

`ultralytics.utils.plotting.plot_labels(...)`

并在本次训练的 **`save_dir`**（例如 `02_YOLO_Training/training_runs/yolov9/<run_name>/`）下保存 **`labels.jpg`**。

源码参考：`site-packages/ultralytics/models/yolo/detect/train.py` 中对 `plot_labels` 的调用。

## 图里四块内容的含义（2×2）

整张图为 **`matplotlib` `subplots(2, 2, figsize=(8, 8))`** 合成的一张 PNG/JPEG，按行列对应如下：

| 位置 | 含义 |
|------|------|
| **左上** | 各类别 **实例数量** 直方图（纵轴 `instances`）；类别名不多时会旋转标注并显示柱上计数。 |
| **右上** | 在 **1000×1000** 白底图上绘制 **至多 500 个** 边界框（归一化后的 xywh），用于肉眼查看框的 **尺度与密集程度**（非真实原图叠加）。 |
| **左下** | 边界框 **中心点 (x, y)** 的二维直方图（`hist2d`，蓝白配色）。 |
| **右下** | 边界框 **宽度与高度** 的二维直方图，反映目标形状分布。 |

边界框在绘图函数内部使用的数组格式为 **`[x, y, width, height]`**（与 YOLO 标签一致）。

## 如何得到「与原版完全一致逻辑」的四张单独图片

训练目录里的 **`labels.jpg`** 是把四张子图画在同一张 `8×8` figure 里再 **`savefig` 一次**得到的 JPEG（dpi=200）。若只把 **`labels.jpg` 按中线裁剪**，会与原版略有差别（边界、tight_layout、JPEG 接缝等问题）。

若要 **与 `plot_labels()` 数值与绘图步骤完全一致**，应从 **`data.yaml` 重新聚合标注**，再对每个子图单独 `savefig`。仓库脚本：

```bash
python 02_YOLO_Training/plot_training_labels_separate.py \
  --data datasets_balanced/data.yaml \
  --save-dir 02_YOLO_Training/training_runs/yolov9/<某次run>/
```

该脚本等价于（与 Ultralytics 一致）：

1. `detect/train.py` 里 `plot_training_labels()`：`boxes` / `cls` 来自 **`train` 集** `dataset.labels` 拼接；
2. `plotting.plot_labels()`：同样的直方图 / 白底框图 / 两个 `hist2d`，每个子图单独 **`figsize=(8,8)`, `dpi=200`**。

输出文件名：

- `labels_01_class_instances.jpg`
- `labels_02_bbox_overlay.jpg`
- `labels_03_center_xy_hist.jpg`
- `labels_04_width_height_hist.jpg`

说明：**单独保存的四张图**，与 **`labels.jpg` 里对应象限在视觉上应一致**（同一数据、同一画法）；若你与线上 `labels.jpg` 逐像素对比，仍可能因「单独子图 vs 拼到一个大画布」的 **布局/tight_layout** 存在极小差异，属于正常现象。

## ~~裁剪整张 labels.jpg~~（不推荐）

早期仓库里的 `split_labels_jpg.py` 仅为几何裁剪，**不代表原版绘图逻辑**，已弃用。请统一使用 **`plot_training_labels_separate.py`**。

