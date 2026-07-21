
"""
原始数据类别统计脚本

作用：
    统计原始 VOC 格式 XML 标注文件中的各类别目标框数量分布，用于了解数据的类别不平衡情况。

主要功能：
    1. 扫描 xmls_1/ 目录下的所有 XML 文件
    2. 解析每个 XML 文件中的 <object> 标签
    3. 统计每个类别名称出现的总次数（按目标框计数，非图片数）
    4. 输出详细的类别分布报告

输入：
    - xmls_1/：VOC 格式 XML 标注文件
    - class.txt：类别名称列表（用于验证类别有效性）

输出：
    - 控制台输出：每个类别的目标框总数
    - 示例输出格式：
        Original Class Counts:
        pipeline_damage: 774
        addition_balconies_windows: 47652
        ...

使用场景：
    - 数据探索阶段：了解原始数据的类别分布
    - 决策依据：确定哪些类别需要增强，哪些需要下采样
    - 基线对比：与增强后的数据集分布进行对比

依赖：
    - xml.etree.ElementTree (Python 标准库)
    - glob (Python 标准库)

使用方式：
    python count_original_classes.py

注意事项：
    - 仅统计 XML 文件中存在的类别，跳过 class.txt 中未定义的类别
    - 按目标框（bbox）计数，一张图片可能有多个目标框
"""

import os
import xml.etree.ElementTree as ET
import glob

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XMLS_DIR = os.path.join(BASE_DIR, 'xmls_1')
CLASSES_FILE = os.path.join(BASE_DIR, 'class.txt')

# Read classes
with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    CLASSES = [line.strip() for line in f.readlines() if line.strip()]

CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}
ID_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

# Count function for XML
def count_xml_classes(xml_dir):
    class_counts = {cls: 0 for cls in CLASSES}
    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    
    print(f"Counting classes in {len(xml_files)} XML files in {xml_dir}...")
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in class_counts:
                    class_counts[name] += 1
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            
    return class_counts

if __name__ == "__main__":
    counts = count_xml_classes(XMLS_DIR)
    print("Original Class Counts:")
    for cls, count in counts.items():
        print(f"{cls}: {count}")

