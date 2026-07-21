
"""
增强后数据集类别统计脚本

作用：
    统计 YOLO 格式数据集（增强后）中的各类别目标框数量分布，用于验证数据增强和平衡效果。

主要功能：
    1. 扫描 datasets/labels/ 目录下的所有 .txt 标签文件（支持递归子目录）
    2. 解析 YOLO 格式的标签文件（每行格式：class_id x_center y_center w h）
    3. 统计每个类别 ID 出现的总次数（按目标框计数）
    4. 将类别 ID 映射到类别名称，输出详细的分布报告

输入：
    - datasets/labels/：YOLO 格式的标签文件目录（包含 train/ 和 val/ 子目录）
    - class.txt：类别名称列表（ID 到名称的映射）

输出：
    - 控制台输出：每个类别的目标框总数
    - 示例输出格式：
        Augmented Dataset Class Counts:
        pipeline_damage: 1500
        addition_balconies_windows: 1600
        ...

使用场景：
    - 验证增强效果：确认稀缺类别是否通过增强达到目标数量
    - 质量检查：发现异常类别（如未知类别 ID）
    - 对比分析：与原始数据分布对比，评估平衡策略效果

依赖：
    - glob (Python 标准库)

使用方式：
    python count_augmented_classes.py

注意事项：
    - 递归扫描所有子目录（包括 train/ 和 val/）
    - 按目标框（bbox）计数，一张图片可能有多个目标框
    - 遇到未知类别 ID 会输出警告信息
    - 通常在运行 prepare_data.py 之后执行此脚本
"""

import os
import glob

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LABELS_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'labels')
CLASSES_FILE = os.path.join(SCRIPT_DIR, 'class.txt')

# Read classes
with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f.readlines() if line.strip()]

ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

def count_labels_in_dir(label_dir):
    class_counts = {cls: 0 for cls in CLASSES}
    label_files = glob.glob(os.path.join(label_dir, '**', '*.txt'), recursive=True)
    
    print(f"Counting classes in {len(label_files)} label files in {label_dir}...")
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        if class_id in ID_TO_CLASS:
                            class_name = ID_TO_CLASS[class_id]
                            class_counts[class_name] += 1
                        else:
                            print(f"Warning: Unknown class ID {class_id} in {label_file}")

        except Exception as e:
            print(f"Error parsing {label_file}: {e}")
            
    return class_counts

if __name__ == "__main__":
    counts = count_labels_in_dir(LABELS_DIR)
    print("Augmented Dataset Class Counts:")
    for cls, count in counts.items():
        print(f"{cls}: {count}")

