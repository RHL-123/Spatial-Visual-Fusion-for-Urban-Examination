import os
from collections import Counter
import yaml

def count_labels(label_dir):
    counter = Counter()
    if not os.path.exists(label_dir):
        print(f"Warning: Directory {label_dir} not found.")
        return counter
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'): # 读取 YOLO 格式标签文件
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls_id = int(parts[0])
                                counter[cls_id] += 1
                            except ValueError:
                                pass
    return counter

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, 'datasets_balanced', 'data.yaml')
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_info = yaml.safe_load(f)
    
    classes = data_info.get('names', [])
    if isinstance(classes, dict): # 处理不同格式的 yaml
        classes = [classes[i] for i in range(len(classes))]
        
    train_label_dir = os.path.join(base_dir, 'datasets_balanced', 'labels', 'train')
    val_label_dir = os.path.join(base_dir, 'datasets_balanced', 'labels', 'val')
    
    train_counts = count_labels(train_label_dir)
    val_counts = count_labels(val_label_dir)
    
    print("\n================ 数据集真实分布对比 (Train vs Val vs Total) ================\n")
    print(f"{'Class Name':<35} | {'Total (原始总数)':<15} | {'Train (训练集)':<15} | {'Val (验证集)':<15} | {'Train Ratio':<10}")
    print("-" * 105)
    
    for i, cls_name in enumerate(classes):
        train_c = train_counts.get(i, 0)
        val_c = val_counts.get(i, 0)
        total_c = train_c + val_c
        
        train_ratio = f"{(train_c / total_c * 100):.1f}%" if total_c > 0 else "0%"
        print(f"{cls_name:<35} | {total_c:<15} | {train_c:<15} | {val_c:<15} | {train_ratio:<10}")
    
    print("\n-----------------------------------------------------------------------------------")
    print("结论分析：如果 Train Ratio 都在相似的百分比（如 80% 左右），说明当初的拆分是十分健康且均匀的，")
    print("这种断崖式的分布差异纯粹来自于真实世界数据采集阶段的“长尾效应（Long-Tail）”。\n")
