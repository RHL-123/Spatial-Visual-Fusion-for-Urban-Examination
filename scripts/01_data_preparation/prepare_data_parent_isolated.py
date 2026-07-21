"""Build a parent-isolated YOLO dataset from the original VOC annotations.

The raw images are split before any resampling or offline augmentation. The
validation set is copied without augmentation and retains its natural class
distribution. Only the training subset is downsampled/augmented.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Sample:
    parent_id: str
    image_path: Path
    boxes: tuple[tuple[int, float, float, float, float], ...]
    classes: frozenset[int]


def locate_image(directory: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def parse_voc(path: Path, class_to_id: dict[str, int]) -> tuple[tuple[int, float, float, float, float], ...]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        return ()
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return ()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if name not in class_to_id or box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(
            (
                class_to_id[name],
                (xmin + xmax) / (2.0 * width),
                (ymin + ymax) / (2.0 * height),
                (xmax - xmin) / width,
                (ymax - ymin) / height,
            )
        )
    return tuple(boxes)


def load_samples(images: Path, xmls: Path, classes: list[str]) -> list[Sample]:
    class_to_id = {name: index for index, name in enumerate(classes)}
    samples = []
    for xml_path in sorted(xmls.glob("*.xml")):
        image_path = locate_image(images, xml_path.stem)
        if image_path is None:
            continue
        boxes = parse_voc(xml_path, class_to_id)
        if not boxes:
            continue
        class_ids = frozenset(box[0] for box in boxes)
        samples.append(Sample(xml_path.stem, image_path, boxes, class_ids))
    return samples


def label_matrix(samples: list[Sample], class_count: int) -> np.ndarray:
    matrix = np.zeros((len(samples), class_count), dtype=np.int8)
    for row, sample in enumerate(samples):
        matrix[row, list(sample.classes)] = 1
    return matrix


def choose_split(samples: list[Sample], class_count: int, val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample], int]:
    """Choose the best of deterministic random candidates by multilabel balance."""
    labels = label_matrix(samples, class_count)
    val_size = max(1, round(len(samples) * val_ratio))
    overall_rate = labels.mean(axis=0)
    best: tuple[float, int, np.ndarray] | None = None
    for candidate_seed in range(seed, seed + 256):
        permutation = np.random.default_rng(candidate_seed).permutation(len(samples))
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        val_counts = labels[val_indices].sum(axis=0)
        train_counts = labels[train_indices].sum(axis=0)
        missing_penalty = 1_000.0 * float(np.count_nonzero(val_counts == 0) + np.count_nonzero(train_counts == 0))
        balance_error = float(np.square(labels[val_indices].mean(axis=0) - overall_rate).sum())
        score = missing_penalty + balance_error
        if best is None or score < best[0]:
            best = (score, candidate_seed, permutation)
    assert best is not None
    permutation = best[2]
    val = [samples[index] for index in permutation[:val_size]]
    train = [samples[index] for index in permutation[val_size:]]
    return train, val, best[1]


def image_level_counts(samples: list[Sample], class_count: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(sample.classes)
    for class_id in range(class_count):
        counts[class_id] += 0
    return counts


def downsample_train(samples: list[Sample], class_count: int, rng: random.Random) -> list[Sample]:
    source_counts = image_level_counts(samples, class_count)
    target = round(sum(source_counts.values()) / class_count)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    kept: list[Sample] = []
    current: Counter[int] = Counter()
    for sample in shuffled:
        if any(current[class_id] < target for class_id in sample.classes):
            kept.append(sample)
            current.update(sample.classes)
    return kept


def augment_image(image: np.ndarray, boxes: tuple[tuple[int, float, float, float, float], ...], kind: str, rng: random.Random, np_rng: np.random.Generator):
    if kind == "flip":
        output = cv2.flip(image, 1)
        new_boxes = tuple((cls, 1.0 - x, y, w, h) for cls, x, y, w, h in boxes)
        return output, new_boxes
    if kind == "noise":
        noise = np_rng.normal(0.0, 0.316, image.shape).astype(np.float32) * 50.0
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), boxes
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    value = rng.randint(-50, 50)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR), boxes


def write_label(path: Path, boxes) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for cls, x, y, width, height in boxes:
            handle.write(f"{cls} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")


def copy_original(sample: Sample, split: str, output: Path, manifest: list[dict[str, str]]) -> None:
    image_target = output / "images" / split / f"{sample.parent_id}{sample.image_path.suffix.lower()}"
    label_target = output / "labels" / split / f"{sample.parent_id}.txt"
    shutil.copy2(sample.image_path, image_target)
    write_label(label_target, sample.boxes)
    manifest.append({"parent_id": sample.parent_id, "split": split, "variant": "original", "image": str(image_target)})


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=script_dir / "images_1")
    parser.add_argument("--xmls", type=Path, default=script_dir / "xmls_1")
    parser.add_argument("--classes", type=Path, default=script_dir / "class.txt")
    parser.add_argument("--output", type=Path, default=project_root / "datasets_parent_isolated")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-augmentation-rounds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1")

    classes = [line.strip() for line in args.classes.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = load_samples(args.images, args.xmls, classes)
    train_raw, val, selected_seed = choose_split(samples, len(classes), args.val_ratio, args.seed)
    train = downsample_train(train_raw, len(classes), random.Random(args.seed))
    print(f"raw_samples={len(samples)} train_raw={len(train_raw)} train_after_downsampling={len(train)} val={len(val)}")
    print(f"requested_seed={args.seed} selected_split_seed={selected_seed}")
    print(f"parent_overlap={len({item.parent_id for item in train} & {item.parent_id for item in val})}")
    for class_id, class_name in enumerate(classes):
        raw_count = image_level_counts(samples, len(classes))[class_id]
        train_count = image_level_counts(train, len(classes))[class_id]
        val_count = image_level_counts(val, len(classes))[class_id]
        print(f"class={class_name} raw={raw_count} train={train_count} val={val_count}")
    if args.dry_run:
        return 0
    if args.output.exists():
        raise FileExistsError(f"Output already exists; remove or choose another path: {args.output}")

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=False)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, str]] = []
    for sample in train:
        copy_original(sample, "train", args.output, manifest)
    for sample in val:
        copy_original(sample, "val", args.output, manifest)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    counts = image_level_counts(train, len(classes))
    target = round(sum(image_level_counts(train_raw, len(classes)).values()) / len(classes))
    for round_index in range(args.max_augmentation_rounds):
        added = 0
        for sample in train:
            if not any(counts[class_id] < target for class_id in sample.classes):
                continue
            image = cv2.imdecode(np.fromfile(sample.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            kind = rng.choice(("flip", "noise", "bright"))
            augmented, boxes = augment_image(image, sample.boxes, kind, rng, np_rng)
            stem = f"{sample.parent_id}_r{round_index}_{kind}"
            image_target = args.output / "images" / "train" / f"{stem}.jpg"
            label_target = args.output / "labels" / "train" / f"{stem}.txt"
            cv2.imencode(".jpg", augmented)[1].tofile(image_target)
            write_label(label_target, boxes)
            counts.update(sample.classes)
            manifest.append({"parent_id": sample.parent_id, "split": "train", "variant": kind, "image": str(image_target)})
            added += 1
        print(f"augmentation_round={round_index} added={added}")
        if added == 0:
            break

    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parent_id", "split", "variant", "image"])
        writer.writeheader()
        writer.writerows(manifest)
    with (args.output / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "nc": len(classes), "names": classes},
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

