import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

def count_voc_stats(img_dir, xml_dir):
    img_path = Path(img_dir)
    xml_path = Path(xml_dir)
    
    # Count images
    num_images = 0
    if img_path.exists():
        num_images = len([f for f in img_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
    # Count classes in xmls
    class_counts = defaultdict(int)
    if xml_path.exists():
        for xml_file in xml_path.glob('*.xml'):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    class_counts[name] += 1
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                
    print(f"Dataset Stats for {img_dir} and {xml_dir}")
    print(f"  Images: {num_images}")
    print("  Classes distribution:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cls_name}: {count}")
    print("-" * 40)

if __name__ == "__main__":
    img_dir = ROOT / "01_Data_Preparation" / "images_1"
    xml_dir = ROOT / "01_Data_Preparation" / "xmls_1"
    count_voc_stats(img_dir, xml_dir)

