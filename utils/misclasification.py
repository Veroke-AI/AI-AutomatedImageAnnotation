from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import pandas as pd
import numpy as np
import umap
import torch
from PIL import Image
import os
import json
import shutil
from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# Load CLIP model/processor globally
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor

model, processor = load_clip_model()

def extract_image_embedding(img_path, model, processor):

    device = next(model.parameters()).device  # Get model device
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)  # Move to same device
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # normalize
    return outputs.squeeze().cpu().numpy()  # Return as CPU numpy


def build_clip_df(crops_root="outputs/crops", model=None, processor=None):
    data = []
    for subdir, _, files in os.walk(crops_root):
        for file in files:
            if "_boxcrop_" in file and file.endswith(".png") and "mask" not in file.lower():
                label = file.split("_boxcrop_")[0]
                img_path = os.path.join(subdir, file)
                emb = extract_image_embedding(img_path, model, processor)
                if emb is not None:
                    data.append({"label": label, "embedding": emb, "image_path": img_path})
    print(f"Loaded {len(data)} embeddings.")
    return pd.DataFrame(data)

def apply_umap(df):
    if df.empty or "embedding" not in df.columns:
        raise ValueError("No embeddings found.")
    X = np.vstack(df["embedding"].values)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    reduced = reducer.fit_transform(X)
    df["x"] = reduced[:, 0]
    df["y"] = reduced[:, 1]
    return df

def compute_cluster_centroids(df):
    centroids = {}
    for label in df["clip_top_labels"].explode().unique():
        cluster_points = df[df["clip_top_labels"].apply(lambda labels: label in labels)]
        if not cluster_points.empty:
            centroids[label] = cluster_points[["x", "y"]].mean().values
    return centroids

def flag_outliers(df, centroids, threshold_quantile=0.95):
    distances = []
    for i, row in df.iterrows():
        label = row["clip_top_labels"][0]  # using top-1
        if label in centroids:
            center = centroids[label]
            dist = np.linalg.norm([row["x"] - center[0], row["y"] - center[1]])
        else:
            dist = np.nan
        distances.append(dist)
    
    df["distance_from_cluster"] = distances

    # Threshold by quantile
    threshold = df["distance_from_cluster"].quantile(threshold_quantile)
    df["is_outlier"] = df["distance_from_cluster"] > threshold
    return df

def flag_misclassified(df, k=5):
    clf = KNeighborsClassifier(n_neighbors=k)
    X = df[["x", "y"]]
    y = df["label"]
    clf.fit(X, y)
    preds = clf.predict(X)
    df["predicted_label"] = preds
    df["is_misclassified"] = df["label"] != df["predicted_label"]
    return df

# def visualize_embeddings(df, save_path="outputs/umap_plot.html"):
#     fig = px.scatter(
#         df,
#         x="x",
#         y="y",
#         color="has_missing",  
#         hover_data=["file_name", "clip_top_labels", "predicted_labels", "missing_labels"],
#         title="CLIP-based Embedding Visualization (UMAP)"
#     )

#     fig.write_html(save_path)
def visualize_embeddings(df, save_path="outputs/umap_plot.html"):
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="clip_top_label",  # use string, not list
        symbol="is_outlier",
        hover_data=["file_name", "clip_top_label", "predicted_labels", "distance_from_cluster"],
        title="CLIP UMAP with Outlier Highlighting"
    )
    fig.write_html(save_path)



# def build_full_image_clip_df(images_dir, coco_json_path, model, processor):

#     with open(coco_json_path, "r") as f:
#         coco = json.load(f)

#     annotations = coco["annotations"]
#     image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

#     data = []

#     for ann in annotations:
#         image_id = ann["image_id"]
#         file_name = image_id_to_filename.get(image_id)

#         if not file_name:
#             continue

#         image_path = os.path.join(images_dir, file_name)
#         if not os.path.exists(image_path):
#             continue

#         image = Image.open(image_path).convert("RGB")

#         inputs = processor(images=image, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outputs = model.get_image_features(**inputs)

#         embedding = outputs.cpu().numpy().flatten()
#         data.append({
#             "image_id": image_id,
#             "file_name": file_name,
#             "embedding": embedding
#         })

#     return pd.DataFrame(data)

def predict_clip_labels(df, model, processor, candidate_labels, top_k=3):
    device = next(model.parameters()).device
    inputs = processor(text=candidate_labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    predictions = []
    for i, row in df.iterrows():
        image_vec = torch.tensor(row["embedding"]).to(device)
        image_vec = image_vec / image_vec.norm(p=2)
        similarities = (image_vec @ text_features.T).cpu().numpy()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_labels = [candidate_labels[idx] for idx in top_indices]
        predictions.append(top_labels)
    
    df["clip_top_labels"] = predictions
    return df

def flag_missing_predictions(df):
    def find_missing(row):
        predicted = set(row["predicted_labels"])
        clip_predicted = set(row["clip_top_labels"])
        missing = clip_predicted - predicted
        return list(missing)

    df["missing_labels"] = df.apply(find_missing, axis=1)
    df["has_missing"] = df["missing_labels"].apply(lambda x: len(x) > 0)
    return df

def build_full_image_clip_df(images_dir, coco_json_path, model, processor):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    annotations = coco["annotations"]
    categories = coco["categories"]
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build category_id → name map
    category_map = {cat["id"]: cat["name"] for cat in categories}

    # Aggregate annotations per image
    image_id_to_labels = {}
    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        label = category_map.get(cat_id, "unknown")

        if img_id not in image_id_to_labels:
            image_id_to_labels[img_id] = set()
        image_id_to_labels[img_id].add(label)

    data = []

    for image_id, file_name in image_id_to_filename.items():
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs.cpu().numpy().flatten()
        data.append({
            "image_id": image_id,
            "file_name": file_name,
            "embedding": embedding,
            "predicted_labels": list(image_id_to_labels.get(image_id, []))
        })

    return pd.DataFrame(data)
# candidate_labels = ["car","truck"]  # or use all categories from COCO

# df_clip = build_full_image_clip_df(
#     images_dir="output/original_split/val",
#     coco_json_path="output/coco_dataset.json",
#     model=model,
#     processor=processor
# )

# df_clip = predict_clip_labels(df_clip, model, processor, candidate_labels, top_k=1)
# df_clip = flag_missing_predictions(df_clip)
# df_clip = apply_umap(df_clip)
# print('1',df_clip)
# visualize_embeddings(df_clip, save_path="output/umap_fullimage.html")
# df_clip = predict_clip_labels(df_clip, model, processor, candidate_labels, top_k=1)
# df_clip = flag_missing_predictions(df_clip)
# df_clip = apply_umap(df_clip)

# centroids = compute_cluster_centroids(df_clip)
# df_clip = flag_outliers(df_clip, centroids, threshold_quantile=0.95)
# df_clip["clip_top_label"] = df_clip["clip_top_labels"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown")

# visualize_embeddings(df_clip, save_path="output/umap_fullimage.html")

# candidate_labels = ["car"]

# df_clip = build_full_image_clip_df(images_dir="output/original_split/val", coco_json_path="outputs/coco_dataset.json", model=model, processor=processor)
# print('1',df_clip)
# df_clip = predict_clip_labels(df_clip, model, processor, candidate_labels, top_k=3)
# print('2',df_clip)
# df_clip["predicted_labels"] = [["car"] for _ in range(len(df_clip))]
# df_clip = flag_missing_predictions(df_clip)
# print('3',df_clip)
# df_clip = apply_umap(df_clip)
# visualize_embeddings(df_clip, save_path="output/umap_fullimage.html")

# ---- Run Pipeline ---- #
# print("Extracting embeddings...")
# df_clip = build_clip_df("outputs/crops", model=model, processor=processor)

# print("Applying UMAP...")
# df_clip = apply_umap(df_clip)

# print("Flagging misclassifications...")
# df_clip = flag_misclassified(df_clip)

# print("Saving interactive UMAP visualization...")
# import os

# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level from utils/
# output_dir = os.path.join(base_dir, "outputs")
# os.makedirs(output_dir, exist_ok=True)

# output_path = os.path.join(output_dir, "umap_plot.html")
# visualize_embeddings(df_clip, save_path=output_path)

# print(f"Saved UMAP plot to {output_path}")
# print("Done. Visualization saved to 'outputs/umap_plot.html'")
def create_crops_from_annotations(image_path, annotations, category_mapping, crops_dir):
    """Create crop images from COCO annotations"""
    try:
        # Ensure crops directory exists
        os.makedirs(crops_dir, exist_ok=True)
        
        with Image.open(image_path) as img:
            crops_created = 0
            for ann in annotations:
                x, y, w, h = ann["bbox"]
                category_name = category_mapping.get(ann["category_id"], "unknown")
                
                # Ensure coordinates are within image bounds
                left = max(0, int(x))
                upper = max(0, int(y))
                right = min(img.width, int(x + w))
                lower = min(img.height, int(y + h))
                
                # Skip invalid bounding boxes
                if left >= right or upper >= lower:
                    print(f"Skipping invalid bbox for annotation {ann.get('id', 'unknown')}")
                    continue
                
                # Create crop
                crop = img.crop((left, upper, right, lower))
                
                # Save crop with descriptive filename
                ann_id = ann.get('id', f'ann_{crops_created}')
                crop_filename = f"{ann_id}_{category_name}.jpg"
                crop_path = os.path.join(crops_dir, crop_filename)
                crop.save(crop_path, "JPEG", quality=95)
                crops_created += 1
                
            print(f"Created {crops_created} crops in {crops_dir}")
                
    except Exception as e:
        print(f"Error creating crops for {image_path}: {str(e)}")


def split_updated_dataset(tmp_dir, output_dir, coco_data, val_ratio=0.2):
    """Split dataset into train/validation sets only (no test set)"""
    import random
    
    # Create split directories - only train and validation
    split_dir = os.path.join(output_dir, "split")
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    
    for directory in [train_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, "images"), exist_ok=True)
        os.makedirs(os.path.join(directory, "annotations"), exist_ok=True)
    
    # Split images into train and validation only
    images = coco_data["images"].copy()
    random.seed(42)  # For reproducible splits
    random.shuffle(images)
    
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    train_images = images[val_count:]  # All remaining images go to train
    
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")
    
    val_image_ids = {img["id"] for img in val_images}
    train_image_ids = {img["id"] for img in train_images}
    
    # Split annotations accordingly
    train_annotations = [ann for ann in coco_data["annotations"] 
                        if ann["image_id"] in train_image_ids]
    val_annotations = [ann for ann in coco_data["annotations"] 
                      if ann["image_id"] in val_image_ids]
    
    print(f"Annotations split: {len(train_annotations)} train, {len(val_annotations)} validation")
    
    # Create separate COCO files for train and validation
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"],
        "info": {"description": "Training set"}
    }
    
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"],
        "info": {"description": "Validation set"}
    }
    
    # Save COCO annotation files
    with open(os.path.join(train_dir, "annotations", "instances.json"), "w") as f:
        json.dump(train_coco, f, indent=2)
    
    with open(os.path.join(val_dir, "annotations", "instances.json"), "w") as f:
        json.dump(val_coco, f, indent=2)
    
    # Copy images to respective directories
    train_copied = 0
    for img_info in train_images:
        src_path = os.path.join(tmp_dir, img_info["file_name"])
        dst_path = os.path.join(train_dir, "images", img_info["file_name"])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            train_copied += 1
    
    val_copied = 0
    for img_info in val_images:
        src_path = os.path.join(tmp_dir, img_info["file_name"])
        dst_path = os.path.join(val_dir, "images", img_info["file_name"])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            val_copied += 1
    
    print(f"Images copied: {train_copied} to train, {val_copied} to validation")


def convert_coco_to_voc_format(coco_data, images_dir, output_dir):
    """Convert COCO format to Pascal VOC XML format"""
    os.makedirs(output_dir, exist_ok=True)
    annotations_dir = os.path.join(output_dir, "Annotations")
    images_output_dir = os.path.join(output_dir, "JPEGImages")
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Create mappings
    image_id_to_info = {img["id"]: img for img in coco_data["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Generate XML files
    for img_id, img_info in image_id_to_info.items():
        if img_id in image_annotations:
            # Copy image to VOC structure
            src_img_path = os.path.join(images_dir, img_info["file_name"])
            dst_img_path = os.path.join(images_output_dir, img_info["file_name"])
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            
            # Generate XML annotation
            xml_content = create_voc_xml_content(img_info, image_annotations[img_id], category_id_to_name)
            xml_filename = Path(img_info["file_name"]).stem + ".xml"
            xml_path = os.path.join(annotations_dir, xml_filename)
            
            with open(xml_path, "w") as f:
                f.write(xml_content)


def create_voc_xml_content(image_info, annotations, category_mapping):
    """Create Pascal VOC XML content for an image"""
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<annotation>
    <folder>JPEGImages</folder>
    <filename>{image_info['file_name']}</filename>
    <size>
        <width>{image_info['width']}</width>
        <height>{image_info['height']}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
"""
    
    for ann in annotations:
        category_name = category_mapping.get(ann['category_id'], 'unknown')
        x, y, w, h = ann['bbox']
        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)
        
        xml_content += f"""    <object>
        <name>{category_name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
"""
    
    xml_content += "</annotation>"
    return xml_content