import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLASSES = [
    "pipeline_damage",
    "addition_balconies_windows",
    "public_spaces_illegal_occupation",
    "no_aging_place_modifications",
    "wall_damage",
    "illegal_renovation_expansion",
    "elevator_addition"
]

def count_dataset_stats(base_dir):
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    splits = ['train', 'val']
    
    for split in splits:
        img_dir = base_path / 'images' / split
        lbl_dir = base_path / 'labels' / split
        
        # Count images
        num_images = 0
        if img_dir.exists():
            num_images = len([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
            
        # Count labels
        class_counts = {cls_name: 0 for cls_name in CLASSES}
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob('*.txt'):
                with open(lbl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(CLASSES):
                                class_counts[CLASSES[cls_id]] += 1
                                
        print(f"[{base_path.name} - {split}]")
        print(f"  Images: {num_images}")
        print("  Classes distribution:")
        for cls_name, count in class_counts.items():
            print(f"    - {cls_name}: {count}")
        print("-" * 40)

if __name__ == "__main__":
    print("=" * 50)
    print("Original Dataset Stats")
    print("=" * 50)
    count_dataset_stats("datasets")
    
    print("\n" + "=" * 50)
    print("Balanced Dataset Stats")
    print("=" * 50)
    count_dataset_stats("datasets_balanced")

