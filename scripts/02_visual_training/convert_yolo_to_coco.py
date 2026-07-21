"""Convert the balanced YOLO dataset into COCO detection annotations.

Many detectors (e.g. Detectron2) expect COCO-style JSON. This script keeps the
original YOLO dataset in place and writes `instances_train.json` /
`instances_val.json` under `datasets_balanced/annotations/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _image_size_match_detectron2(image_path: Path) -> tuple[int, int]:
    """Width/height exactly as Detectron2 training uses (detection_utils.read_image).

    PIL's ImageOps.exif_transpose and D2's _apply_exif_orientation can disagree on
    a few files; a mismatch causes SizeMismatchError during training.
    """
    try:
        from detectron2.data.detection_utils import read_image

        arr = read_image(str(image_path), format="RGB")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        return w, h
    except Exception:
        with Image.open(image_path) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass
            return im.size


def read_data_yaml(path: Path) -> tuple[Path, list[str]]:
    dataset_root: Path | None = None
    names: list[str] = []
    in_names = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            dataset_root = Path(line.split(":", 1)[1].strip().strip("'\""))
            in_names = False
        elif line.startswith("names:"):
            in_names = True
        elif in_names and line.startswith("-"):
            names.append(line[1:].strip().strip("'\""))
        elif ":" in line:
            in_names = False

    if dataset_root is None:
        raise ValueError(f"Missing `path:` in {path}")
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    if not names:
        raise ValueError(f"Missing class `names:` in {path}")
    return dataset_root, names


def yolo_box_to_coco(
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    width = w_norm * image_width
    height = h_norm * image_height
    x = cx_norm * image_width - width / 2
    y = cy_norm * image_height - height / 2

    x = max(0.0, min(x, float(image_width)))
    y = max(0.0, min(y, float(image_height)))
    width = max(0.0, min(width, float(image_width) - x))
    height = max(0.0, min(height, float(image_height) - y))
    return x, y, width, height


def convert_split(dataset_root: Path, names: list[str], split: str) -> dict:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    images = []
    annotations = []
    annotation_id = 1

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    for image_id, image_path in enumerate(image_paths, start=1):
        width, height = _image_size_match_detectron2(image_path)

        images.append(
            {
                "id": image_id,
                "file_name": str(Path("images") / split / image_path.name),
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_no}: {line!r}")

            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(names):
                raise ValueError(f"Class id {class_id} out of range at {label_path}:{line_no}")

            bbox = yolo_box_to_coco(*(float(v) for v in parts[1:]), width, height)
            area = bbox[2] * bbox[3]
            if area <= 0:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [round(v, 4) for v in bbox],
                    "area": round(area, 4),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": "Renhelei balanced detection dataset converted from YOLO format",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name, "supercategory": "defect"} for i, name in enumerate(names)],
    }


def convert_dataset(data_yaml: Path, splits: list[str]) -> list[Path]:
    dataset_root, names = read_data_yaml(data_yaml)
    output_dir = dataset_root / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for split in splits:
        coco = convert_split(dataset_root, names, split)
        output_path = output_dir / f"instances_{split}.json"
        output_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        outputs.append(output_path)
        print(
            f"Wrote {output_path}: "
            f"{len(coco['images'])} images, {len(coco['annotations'])} annotations, {len(coco['categories'])} categories"
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets_balanced" / "data.yaml",
        help="Path to the YOLO data.yaml file.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = parser.parse_args()

    convert_dataset(args.data, args.splits)


if __name__ == "__main__":
    main()

