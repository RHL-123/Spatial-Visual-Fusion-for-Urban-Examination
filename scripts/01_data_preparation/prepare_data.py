"""
数据准备与增强脚本 - YOLO 格式数据集生成

作用：
    将原始 VOC 格式数据（XML 标注）转换为 YOLO 格式数据集，并通过数据增强实现类别平衡。

主要功能：
    1. XML 解析：读取 xmls_1/ 目录下的 VOC 格式 XML 标注文件
    2. 格式转换：将边界框从 (xmin, ymin, xmax, ymax) 转换为 YOLO 格式 (x_center, y_center, w, h) 并归一化
    3. 数据增强：
       - 水平翻转（augment_hflip）
       - 高斯噪声（augment_noise）
       - 亮度调整（augment_brightness）
    4. 类别平衡：
       - 下采样：贪心策略保留包含"未达标类别"的图片
       - 增强补充：对稀缺类别的图片进行增强，直到达到目标数量（各类别平均值为目标）
    5. 数据集划分：80% 训练集，20% 验证集
    6. 配置生成：创建 data.yaml 文件供 YOLO 训练使用

输入：
    - images_1/：原始图像文件（.jpg）
    - xmls_1/：VOC 格式 XML 标注文件
    - class.txt：类别名称列表（每行一个类别）

输出：
    - datasets/images/train/：训练集图像
    - datasets/images/val/：验证集图像
    - datasets/labels/train/：训练集 YOLO 格式标签
    - datasets/labels/val/：验证集 YOLO 格式标签
    - datasets/data.yaml：YOLO 训练配置文件

依赖：
    - opencv-python (cv2)
    - numpy
    - tqdm
    - pyyaml
    - openpyxl (pandas 读取 Excel 所需，如果后续使用)

使用方式：
    python prepare_data.py

注意事项：
    - 支持中文路径（使用 cv2.imdecode/cv2.imencode 处理）
    - 临时增强文件保存在 temp_aug/ 目录，完成后清理
"""

import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random
from tqdm import tqdm
import yaml

# 配置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images_1')
XMLS_DIR = os.path.join(SCRIPT_DIR, 'xmls_1')
CLASSES_FILE = os.path.join(SCRIPT_DIR, 'class.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'datasets')
IMAGES_OUT_TRAIN = os.path.join(OUTPUT_DIR, 'images', 'train')
IMAGES_OUT_VAL = os.path.join(OUTPUT_DIR, 'images', 'val')
LABELS_OUT_TRAIN = os.path.join(OUTPUT_DIR, 'labels', 'train')
LABELS_OUT_VAL = os.path.join(OUTPUT_DIR, 'labels', 'val')
TEMP_AUG_DIR = os.path.join(OUTPUT_DIR, 'temp_aug')

# 读取类别
with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f.readlines() if line.strip()]

CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}
ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

print(f"检测到 {len(CLASSES)} 个类别: {CLASSES}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 创建输出目录
for p in [IMAGES_OUT_TRAIN, IMAGES_OUT_VAL, LABELS_OUT_TRAIN, LABELS_OUT_VAL, TEMP_AUG_DIR]:
    ensure_dir(p)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w_img = int(size.find('width').text)
    h_img = int(size.find('height').text)
    
    bboxes = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_TO_ID:
            continue
        
        cls_id = CLASS_TO_ID[name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # 转换为YOLO格式 (x_center, y_center, w, h) 归一化
        x_c = (xmin + xmax) / 2.0 / w_img
        y_c = (ymin + ymax) / 2.0 / h_img
        w = (xmax - xmin) / w_img
        h = (ymax - ymin) / h_img
        
        bboxes.append([cls_id, x_c, y_c, w, h])
        
    return bboxes, w_img, h_img

def save_yolo_label(bboxes, path):
    with open(path, 'w') as f:
        for b in bboxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

# 数据增强函数
def augment_hflip(img, bboxes):
    # 水平翻转
    img_flip = cv2.flip(img, 1)
    new_bboxes = []
    for b in bboxes:
        # x_center 变为 1 - x_center
        new_b = list(b)
        new_b[1] = 1.0 - new_b[1]
        new_bboxes.append(new_b)
    return img_flip, new_bboxes

def augment_noise(img, bboxes):
    # 高斯噪声
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    # 使用 float32 节省内存
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    noisy = img.astype(np.float32) + gauss * 50 # 强度
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, bboxes

def augment_brightness(img, bboxes):
    # 亮度调整
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 随机调整 V
    value = random.randint(-50, 50)
    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = abs(value)
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value
        
    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright, bboxes

# 收集原始数据
print("正在扫描原始数据...")
original_data = [] # item: {'img_path': str, 'xml_path': str, 'bboxes': list, 'classes': set}
class_counts = {i: 0 for i in range(len(CLASSES))}

xml_files = [f for f in os.listdir(XMLS_DIR) if f.endswith('.xml')]

for xml_file in tqdm(xml_files):
    xml_path = os.path.join(XMLS_DIR, xml_file)
    img_file = xml_file.replace('.xml', '.jpg')
    img_path = os.path.join(IMAGES_DIR, img_file)
    
    if not os.path.exists(img_path):
        continue
        
    bboxes, _, _ = parse_xml(xml_path)
    if not bboxes:
        continue
        
    cls_ids = {int(b[0]) for b in bboxes}
    for cid in cls_ids:
        class_counts[cid] += 1
        
    original_data.append({
        'base_name': xml_file.replace('.xml', ''),
        'img_path': img_path,
        'bboxes': bboxes,
        'classes': cls_ids
    })

print("原始类别分布:", {ID_TO_CLASS[k]: v for k, v in class_counts.items()})

# 计算平衡目标
if not class_counts:
    print("没有找到有效数据！")
    exit()

counts_values = list(class_counts.values())
target_count = int(np.mean(counts_values)) # 设定目标为平均值
print(f"设定目标均衡数量: {target_count}")

# 下采样阶段：减少过多类别的样本
print("正在进行下采样...")
random.shuffle(original_data)
balanced_data = []
current_counts = {k: 0 for k in class_counts}

for item in original_data:
    keep = False
    # 如果该图片包含的任意类别数量未达到目标，则保留该图片
    # 贪心策略：优先保留包含"未达标类别"的图片
    for cid in item['classes']:
        if current_counts[cid] < target_count:
            keep = True
            break
            
    if keep:
        balanced_data.append(item)
        for cid in item['classes']:
            current_counts[cid] += 1

original_data = balanced_data
print("下采样后类别分布:", {ID_TO_CLASS[k]: v for k, v in current_counts.items()})

# 增强阶段
augmented_data = []
# current_counts 已更新，直接继续

print("开始数据增强...")
# 简单策略：循环遍历数据，如果包含"稀缺"类别，就进行增强
# 为了防止无限循环，设置最大轮数
max_rounds = 5

for r in range(max_rounds):
    print(f"增强轮次 {r+1}/{max_rounds}")
    added_in_round = 0
    for item in original_data:
        # 检查该图片是否包含稀缺类别
        needs_aug = False
        for cid in item['classes']:
            if current_counts[cid] < target_count:
                needs_aug = True
                break
        
        if not needs_aug:
            continue
            
        # 随机选择一种增强方式
        aug_type = random.choice(['flip', 'noise', 'brightness'])
        
        try:
            # img = cv2.imread(item['img_path']) # cv2.imread不支持中文路径
            img = cv2.imdecode(np.fromfile(item['img_path'], dtype=np.uint8), -1)
            if img is None:
                continue
                
            if aug_type == 'flip':
                aug_img, aug_bboxes = augment_hflip(img, item['bboxes'])
                suffix = '_flip'
            elif aug_type == 'noise':
                aug_img, aug_bboxes = augment_noise(img, item['bboxes'])
                suffix = '_noise'
            else:
                aug_img, aug_bboxes = augment_brightness(img, item['bboxes'])
                suffix = '_bright'
            
            # 记录新数据
            new_base_name = f"{item['base_name']}_r{r}{suffix}"
            
            # 立即保存到临时目录，避免占用内存
            temp_img_path = os.path.join(TEMP_AUG_DIR, new_base_name + '.jpg')
            cv2.imencode('.jpg', aug_img)[1].tofile(temp_img_path)
            
            augmented_data.append({
                'base_name': new_base_name,
                'img_path': temp_img_path, # 记录路径而不是数据
                'bboxes': aug_bboxes,
                'classes': item['classes'] # 假定类别不变
            })
            
            # 释放内存
            del aug_img
            
            # 更新计数
            for cid in item['classes']:
                current_counts[cid] += 1
                
            added_in_round += 1
            
        except Exception as e:
            print(f"Error augmenting {item['base_name']}: {e}")
            continue
            
    print(f"本轮新增 {added_in_round} 张图像")
    if added_in_round == 0:
        break

print("增强后类别分布:", {ID_TO_CLASS[k]: v for k, v in current_counts.items()})

# 保存与划分数据集
all_items = original_data + augmented_data
random.shuffle(all_items)

val_ratio = 0.2
num_val = int(len(all_items) * val_ratio)
train_items = all_items[num_val:]
val_items = all_items[:num_val]

print(f"总数据: {len(all_items)}, 训练集: {len(train_items)}, 验证集: {len(val_items)}")

def save_dataset(items, img_dir, lbl_dir):
    for item in tqdm(items):
        base_name = item['base_name']
        out_img_path = os.path.join(img_dir, base_name + '.jpg')
        out_lbl_path = os.path.join(lbl_dir, base_name + '.txt')
        
        # 保存图片
        if 'img_data' in item:
            # 增强生成的图片
            # cv2.imwrite(out_img_path, item['img_data']) # cv2.imwrite 不支持中文路径
            cv2.imencode('.jpg', item['img_data'])[1].tofile(out_img_path)
        else:
            # 原始图片，复制
            shutil.copy(item['img_path'], out_img_path)
            
        # 保存标签
        save_yolo_label(item['bboxes'], out_lbl_path)

print("保存训练集...")
save_dataset(train_items, IMAGES_OUT_TRAIN, LABELS_OUT_TRAIN)
print("保存验证集...")
save_dataset(val_items, IMAGES_OUT_VAL, LABELS_OUT_VAL)

# 生成 data.yaml
yaml_content = {
    'path': 'datasets',
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(CLASSES),
    'names': CLASSES
}

with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print("数据准备完成！")
print(f"data.yaml 已生成至 {os.path.join(OUTPUT_DIR, 'data.yaml')}")

# 清理临时目录
try:
    shutil.rmtree(TEMP_AUG_DIR)
    print("已清理临时文件")
except Exception as e:
    print(f"清理临时文件失败: {e}")

