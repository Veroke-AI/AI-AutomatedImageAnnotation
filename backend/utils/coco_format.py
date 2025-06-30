import json
import os
import numpy as np
import cv2

def export_to_coco(image_path, detections, masks, output_json_path, image_id=1):
    """
    Export annotations to COCO format.

    Args:
        image_path (str): Path to the original image.
        detections (list): List of dicts with keys 'label', 'confidence', and 'bbox' (in [x1, y1, x2, y2]).
        masks (list): List of binary masks (np.ndarray of shape HxW) corresponding to detections.
        output_json_path (str): Path to save the COCO-format JSON file.
        image_id (int): ID of the image.
    """

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Prepare image info
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    image_filename = os.path.basename(image_path)

    coco["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": width,
        "height": height
    })

    category_name_to_id = {}
    annotation_id = 1

    for i, (det, mask) in enumerate(zip(detections, masks)):
        label = det.get("label", f"object{i}")  # Correct key lookup
        confidence = det.get("confidence", 0.0)
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(float, bbox)
        coco_bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height

        # Assign category id
        if label not in category_name_to_id:
            category_id = len(category_name_to_id) + 1
            category_name_to_id[label] = category_id
            coco["categories"].append({
                "id": category_id,
                "name": label
            })
        else:
            category_id = category_name_to_id[label]

        area = float(np.sum(mask > 0))

        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": coco_bbox,
            "area": area,
            "iscrowd": 0,
            "score": confidence
        })
        annotation_id += 1


        # Encode mask as RLE or polygon (simplified here using cv2.findContours)
        # mask_np = mask.astype(np.uint8)
        # contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # segmentation = []
        # for contour in contours:
        #     if contour.size >= 6:  # At least 3 points
        #         contour = contour.flatten().tolist()
        #         segmentation.append(contour)

        # area = float(np.sum(mask > 0))

        # coco["annotations"].append({
        #     "id": annotation_id,
        #     "image_id": image_id,
        #     "category_id": category_id,
        #     "bbox": coco_bbox,
        #     "area": area,
        #     "iscrowd": 0,
        #     "score": confidence  # Optional, non-standard COCO
        # })
        # annotation_id += 1

    # Write to file
    with open(output_json_path, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"COCO annotations saved to {output_json_path}")

# import json
# import os
# import cv2

# def export_to_coco_bulk(image_root_dir, output_json_path):
#     """
#     Export cropped images (organized by class in train/val folders) to COCO format.
#     Assumes no detection/mask info — only class from folder name.

#     Args:
#         image_root_dir (str): Path to root folder containing 'train' and/or 'val' subfolders.
#         output_json_path (str): Path to save COCO JSON.
#     """
#     coco = {
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }

#     category_name_to_id = {}
#     image_id = 1
#     annotation_id = 1

#     for split in ["train", "val"]:
#         split_dir = os.path.join(image_root_dir, split)
#         if not os.path.exists(split_dir):
#             continue

#         for class_name in os.listdir(split_dir):
#             class_dir = os.path.join(split_dir, class_name)
#             if not os.path.isdir(class_dir):
#                 continue

#             # Assign category id if not already
#             if class_name not in category_name_to_id:
#                 category_id = len(category_name_to_id) + 1
#                 category_name_to_id[class_name] = category_id
#                 coco["categories"].append({
#                     "id": category_id,
#                     "name": class_name
#                 })
#             else:
#                 category_id = category_name_to_id[class_name]

#             for img_file in os.listdir(class_dir):
#                 if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     continue

#                 img_path = os.path.join(class_dir, img_file)
#                 image = cv2.imread(img_path)
#                 if image is None:
#                     continue
#                 height, width = image.shape[:2]

#                 coco["images"].append({
#                     "id": image_id,
#                     "file_name": os.path.relpath(img_path, image_root_dir).replace("\\", "/"),
#                     "width": width,
#                     "height": height
#                 })

#                 coco["annotations"].append({
#                     "id": annotation_id,
#                     "image_id": image_id,
#                     "category_id": category_id,
#                     "bbox": [0, 0, width, height],  # whole crop is the bbox
#                     "area": width * height,
#                     "iscrowd": 0
#                 })

#                 image_id += 1
#                 annotation_id += 1

#     with open(output_json_path, "w") as f:
#         json.dump(coco, f, indent=2)

#     print(f"[✓] Exported COCO file: {output_json_path}")
import os
import cv2
import json
import hashlib

def get_image_id_from_filename(filename):
    # Hash the filename (without extension) to create a reproducible numeric ID
    base = os.path.splitext(filename)[0]
    return int(hashlib.sha256(base.encode()).hexdigest(), 16) % (10 ** 9)

def export_to_coco_bulk(image_root_dir, output_json_path):
    """
    Export cropped images (organized by class in train/val folders) to COCO format.
    Assumes multiple annotations can belong to the same original image.

    Args:
        image_root_dir (str): Path to root folder containing 'train' and/or 'val' subfolders.
        output_json_path (str): Path to save COCO JSON.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_name_to_id = {}
    seen_image_ids = set()
    annotation_id = 1

    for split in ["train", "val"]:
        split_dir = os.path.join(image_root_dir, split)
        if not os.path.exists(split_dir):
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Assign category ID
            if class_name not in category_name_to_id:
                category_id = len(category_name_to_id) + 1
                category_name_to_id[class_name] = category_id
                coco["categories"].append({
                    "id": category_id,
                    "name": class_name
                })
            else:
                category_id = category_name_to_id[class_name]

            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                height, width = image.shape[:2]

                # Use hashed filename for unique, consistent image_id
                image_id = get_image_id_from_filename(img_file)

                # Only add the image entry once
                if image_id not in seen_image_ids:
                    coco["images"].append({
                        "id": image_id,
                        "file_name": os.path.relpath(img_path, image_root_dir).replace("\\", "/"),
                        "width": width,
                        "height": height
                    })
                    seen_image_ids.add(image_id)

                # Each annotation is unique, but refers to the same image
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [0, 0, width, height],  # or use real bbox if available
                    "area": width * height,
                    "iscrowd": 0
                })

                annotation_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"[✓] Exported COCO file: {output_json_path}")       

import json
import os
from glob import glob

import os
import json
from glob import glob

def merge_coco_jsons(json_folder, output_path):
    merged = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_id_map = {}          # Original ID to new ID
    category_name_to_new_id = {}  # Ensure unique name -> ID
    next_category_id = 1
    next_image_id = 1
    next_annotation_id = 1

    json_files = glob(os.path.join(json_folder, "*.json"))

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Remap categories
        local_cat_id_map = {}
        for cat in data.get("categories", []):
            name = cat["name"]
            if name not in category_name_to_new_id:
                category_name_to_new_id[name] = next_category_id
                merged["categories"].append({
                    "id": next_category_id,
                    "name": name
                })
                next_category_id += 1
            local_cat_id_map[cat["id"]] = category_name_to_new_id[name]

        # Remap images
        image_id_map = {}
        for img in data.get("images", []):
            old_id = img["id"]
            img["id"] = next_image_id
            image_id_map[old_id] = next_image_id
            merged["images"].append(img)
            next_image_id += 1

        # Remap annotations
        for ann in data.get("annotations", []):
            ann["image_id"] = image_id_map[ann["image_id"]]
            ann["category_id"] = local_cat_id_map[ann["category_id"]]
            ann["id"] = next_annotation_id
            merged["annotations"].append(ann)
            next_annotation_id += 1

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(json_files)} files into {output_path}")


# Example usage
# merge_coco_jsons("path/to/json/folder", "merged_output.json")


def update_coco_json_with_flags(
    json_path, 
    class_array, 
    confidence_threshold=0.4
):
    # Load existing JSON file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from class names to category IDs
    name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Get valid category IDs based on class_array
    valid_class_ids = [name_to_id[name] for name in class_array if name in name_to_id]

    # Filter and update annotations
    updated_annotations = []
    for ann in coco_data['annotations']:
        if ann['category_id'] in valid_class_ids:
            score = ann.get('score', 1.0)
            ann['misclassified'] = score < confidence_threshold
            updated_annotations.append(ann)

    # Filter categories
    updated_categories = [cat for cat in coco_data['categories'] if cat['id'] in valid_class_ids]

    # Update the original data
    coco_data['annotations'] = updated_annotations
    coco_data['categories'] = updated_categories

    # Save back to the same JSON file
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Updated COCO JSON saved to: {json_path}")


def convert_coco_to_yolo(coco_json_path, output_dir):
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Create ID-to-name and name-to-ID mappings
    categories = coco['categories']
    id_to_name = {cat['id']: cat['name'] for cat in categories}
    name_to_id = {cat['name']: i for i, cat in enumerate(categories)}
    coco_id_to_yolo_id = {cat['id']: name_to_id[cat['name']] for cat in categories}

    # Write class names to file (optional)
    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        for name in name_to_id:
            f.write(f"{name}\n")

    # Map image_id to file_name, width, height
    image_info = {img['id']: img for img in coco['images']}

    # Prepare YOLO annotations
    yolo_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann['bbox']  # [x_min, y_min, width, height]
        x_min, y_min, box_w, box_h = bbox

        image = image_info[img_id]
        img_w, img_h = image['width'], image['height']

        # Convert to YOLO format: normalized [x_center, y_center, width, height]
        x_center = (x_min + box_w / 2) / img_w
        y_center = (y_min + box_h / 2) / img_h
        w = box_w / img_w
        h = box_h / img_h

        yolo_id = coco_id_to_yolo_id[cat_id]
        line = f"{yolo_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        img_filename = os.path.splitext(image['file_name'])[0]
        yolo_annotations.setdefault(img_filename, []).append(line)

    # Write YOLO .txt files
    for img_filename, lines in yolo_annotations.items():
        label_file = os.path.join(labels_dir, f"{img_filename}.txt")
        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))

    print(f"YOLO annotations written to: {labels_dir}")