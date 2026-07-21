"""
智能欠采样数据集精简脚本

作用：
    对已准备好的 YOLO 数据集进行智能欠采样，去除冗余数据，生成精简平衡版数据集。
    通过分级策略保留重要样本，大幅减少训练时间和计算资源消耗。

主要功能：
    1. 类别分级处理：
       - 稀有类（大熊猫）：包含类别 0, 5, 6 的图片 100% 保留
       - 中等类：包含类别 2, 3, 4 的图片 50% 随机保留
       - 常见类（泛滥）：纯类别 1 的图片仅 10% 保留
       - 背景图：无任何目标的图片 5% 保留（作为负样本防止误报）
    2. 智能判断：根据图片包含的类别组合决定保留概率
    3. 独立输出：生成到 datasets_balanced/ 目录，不破坏原始 datasets/ 数据
    4. 配置更新：更新 data.yaml 中的路径指向新数据集

类别分级说明：
    - 稀有类（100% 保留）：
      * 0: pipeline_damage (774 个)
      * 5: illegal_renovation_expansion (4080 个)
      * 6: elevator_addition (4962 个)
    - 中等类（50% 保留）：
      * 2: public_spaces_illegal_occupation (9536 个)
      * 3: no_aging_place_modifications (7593 个)
      * 4: wall_damage (6898 个)
    - 常见类（10% 保留）：
      * 1: addition_balconies_windows (47652 个，极其泛滥)

输入：
    - datasets/images/train/：原始训练集图像
    - datasets/labels/train/：原始训练集标签
    - datasets/images/val/：原始验证集图像
    - datasets/labels/val/：原始验证集标签
    - datasets/data.yaml：原始配置文件

输出：
    - datasets_balanced/images/train/：精简后的训练集图像
    - datasets_balanced/labels/train/：精简后的训练集标签
    - datasets_balanced/images/val/：精简后的验证集图像
    - datasets_balanced/labels/val/：精简后的验证集标签
    - datasets_balanced/data.yaml：更新后的配置文件

依赖：
    - pyyaml
    - random (Python 标准库)
    - shutil (Python 标准库)

使用方式：
    python create_balanced_dataset.py

注意事项：
    - 固定随机种子 random.seed(42)，保证每次运行结果可复现
    - 优先保留包含稀有类别的图片（保护大熊猫原则）
    - 保留少量背景图作为负样本，防止模型误报
    - 原始 datasets/ 目录完全不受影响，可安全重复运行
"""

import os
import shutil
import random
import yaml
from pathlib import Path

# 固定随机种子，保证每次欠采样结果一致
random.seed(42)

# 定义路径 (基于脚本所在目录计算绝对路径)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
DST_DATASET_DIR = os.path.join(BASE_DIR, 'datasets_balanced')

# 定义类别分级
# 0: pipeline_damage (774)
# 1: addition_balconies_windows (47652) ---> 极其泛滥
# 2: public_spaces_illegal_occupation (9536)
# 3: no_aging_place_modifications (7593)
# 4: wall_damage (6898)
# 5: illegal_renovation_expansion (4080)
# 6: elevator_addition (4962)

RARE_CLASSES = {0, 5, 6}        # 稀有大熊猫，包含它们的图片 100% 保留
MIDDLE_CLASSES = {2, 3, 4}      # 中等类别，包含它们且没有大熊猫的图片保留 50%
COMMON_CLASSES = {1}            # 纯泛滥类别，如果一张图里【只有】阳台窗户，只保留 10%

def analyze_and_filter(split='train'):
    src_labels_dir = os.path.join(SRC_DATASET_DIR, 'labels', split)
    src_images_dir = os.path.join(SRC_DATASET_DIR, 'images', split)
    
    dst_labels_dir = os.path.join(DST_DATASET_DIR, 'labels', split)
    dst_images_dir = os.path.join(DST_DATASET_DIR, 'images', split)
    
    os.makedirs(dst_labels_dir, exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)
    
    if not os.path.exists(src_labels_dir):
        print(f"Directory {src_labels_dir} does not exist. Skipping {split}.")
        return

    total_count = 0
    kept_count = 0
    
    label_files = [f for f in os.listdir(src_labels_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        total_count += 1
        src_label_path = os.path.join(src_labels_dir, label_file)
        
        # 读取该图片的所有 bbox 类别
        classes_in_img = set()
        with open(src_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        classes_in_img.add(int(parts[0]))
                    except ValueError:
                        pass
        
        # -------- 核心清洗/欠采样逻辑 --------
        keep_image = False
        
        # 1. 如果包含稀有框，像保护大熊猫一样 100% 绝对保留！
        if classes_in_img.intersection(RARE_CLASSES):
            keep_image = True
            
        # 2. 如果包含中等类别，掷骰子 50% 概率保留
        elif classes_in_img.intersection(MIDDLE_CLASSES):
            if random.random() < 0.50:
                keep_image = True
                
        # 3. 如果整张图里全都是烂大街的类别（纯阳台），无情砍掉，只留 10%
        elif classes_in_img.issubset(COMMON_CLASSES):
            if random.random() < 0.10:
                keep_image = True
        
        # 如果一张图是空的（背景图），保留 5% 作为 Negative Samples 防止误报
        elif not classes_in_img:
            if random.random() < 0.05:
                keep_image = True

        # 执行保留动作 (拷贝 image 和 label)
        if keep_image:
            image_exts = ['.jpg', '.png', '.jpeg']
            base_name = os.path.splitext(label_file)[0]
            img_found = False
            
            for ext in image_exts:
                src_img_path = os.path.join(src_images_dir, base_name + ext)
                if os.path.exists(src_img_path):
                    dst_img_path = os.path.join(dst_images_dir, base_name + ext)
                    shutil.copy2(src_img_path, dst_img_path)
                    img_found = True
                    break
            
            if img_found:
                dst_label_path = os.path.join(dst_labels_dir, label_file)
                shutil.copy2(src_label_path, dst_label_path)
                kept_count += 1

    print(f"[{split.upper()}] 原始图片数: {total_count} -> 清洗后保留图片数: {kept_count} (缩减比例: {(1 - kept_count/(total_count+1e-5))*100:.1f}%)")

def copy_and_update_yaml():
    src_yaml = os.path.join(SRC_DATASET_DIR, 'data.yaml')
    dst_yaml = os.path.join(DST_DATASET_DIR, 'data.yaml')
    
    if os.path.exists(src_yaml):
        with open(src_yaml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # 覆写路径为项目根目录相对路径，方便整个项目迁移到服务器。
        data['path'] = 'datasets_balanced'
        
        with open(dst_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"Generated new YAML config at: {dst_yaml}")

if __name__ == '__main__':
    print("================ 开始执行智能数据清洗 (Undersampling) ================")
    print(f"源数据集: {SRC_DATASET_DIR}")
    print(f"目标数据集: {DST_DATASET_DIR}")
    print("------------------------------------------------------------------")
    
    analyze_and_filter('train')
    analyze_and_filter('val')
    copy_and_update_yaml()
    
    print("================ 清洗完成 ============================================")
    print("✨ 注意：原有的 datasets/ 文件夹丝毫未动！新数据已单独存放在 datasets_balanced/ 下！")

