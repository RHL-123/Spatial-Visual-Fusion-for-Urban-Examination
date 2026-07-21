import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def count_target_classes(xml_dir, class_file):
    xml_path = Path(xml_dir)
    class_path = Path(class_file)
    
    if not class_path.exists():
        print(f"Class file {class_file} does not exist.")
        return
        
    # Read target classes
    with open(class_path, 'r', encoding='utf-8') as f:
        target_classes = [line.strip() for line in f if line.strip()]
        
    # Initialize counts for target classes
    class_counts = {cls_name: 0 for cls_name in target_classes}
    
    # Count classes in xmls
    if xml_path.exists():
        for xml_file in xml_path.glob('*.xml'):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in class_counts:
                        class_counts[name] += 1
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                
    print(f"Target Sub-classes Stats for {xml_dir}")
    print("  Classes distribution:")
    for cls_name in target_classes:
        print(f"    - {cls_name}: {class_counts[cls_name]}")
    print("-" * 40)

if __name__ == "__main__":
    xml_dir = ROOT / "01_Data_Preparation" / "xmls_1"
    class_file = ROOT / "01_Data_Preparation" / "class.txt"
    count_target_classes(xml_dir, class_file)

