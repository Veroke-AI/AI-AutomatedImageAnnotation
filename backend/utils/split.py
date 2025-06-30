import random
import shutil
import json
from pathlib import Path
from collections import defaultdict
from typing import Union
import os
from typing import Dict

def split_crops_for_training(
    crops_dir: Union[str, Path],
    split_dir: Union[str, Path],
    val_ratio: float = 0.2
):
    crops_dir = Path(crops_dir)
    split_dir = Path(split_dir)
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    label_to_files = defaultdict(list)
    labelmap = {}

    # Traverse each image subdirectory
    for image_folder in crops_dir.iterdir():
        if not image_folder.is_dir():
            continue
        for file in image_folder.glob("*_boxcrop_*.png"):
            label = file.name.split("_boxcrop_")[0]
            label_to_files[label].append(file)
            labelmap[file.name] = label

    # Split and copy
    for label, files in label_to_files.items():
        random.shuffle(files)
        split_idx = int(len(files) * (1 - val_ratio))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for tfile in train_files:
            label_folder = train_dir / label
            label_folder.mkdir(parents=True, exist_ok=True)
            unique_name = f"{tfile.parent.name}_{tfile.name}"
            shutil.copy(tfile, label_folder / unique_name)
            labelmap[unique_name] = label

        for vfile in val_files:
            label_folder = val_dir / label
            label_folder.mkdir(parents=True, exist_ok=True)
            unique_name = f"{vfile.parent.name}_{vfile.name}"
            shutil.copy(vfile, label_folder / unique_name)
            labelmap[unique_name] = label

    # Save label map
    with open(split_dir / "labelmap.json", "w") as f:
        json.dump(labelmap, f, indent=4)

def get_first_crop_path(crops_dir):
    for root, _, files in os.walk(crops_dir):
        for file in files:
            if file.endswith(".png") and "boxcrop" in file:
                return os.path.join(root, file)
    return None


def split_uploaded_images(image_dir: str, output_dir: str, val_ratio: float = 0.2) -> Dict:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_images = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_exts]

    if not all_images:
        raise ValueError("No images found to split.")

    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    split_dir = os.path.join(output_dir, "original_split")
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(image_dir, img), os.path.join(train_dir, img))
    for img in val_imgs:
        shutil.copy(os.path.join(image_dir, img), os.path.join(val_dir, img))

    return {
        "train_count": len(train_imgs),
        "val_count": len(val_imgs),
        "train_dir": train_dir,
        "val_dir": val_dir,
    }

