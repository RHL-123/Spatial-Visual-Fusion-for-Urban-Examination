#!/usr/bin/env python3
"""已弃用：原先只是对合成后的 labels.jpg 做几何裁剪，与 Ultralytics 绘图逻辑不一致。

请改用「原版逻辑」脚本（从 data.yaml 聚合 boxes/cls，按 plot_labels 画法分别出图）：

  python plot_training_labels_separate.py --data ../datasets_balanced/data.yaml --save-dir <run目录>

若仍需裁剪整张 labels.jpg，可自行用图像编辑器或一行 Pillow crop。
"""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

