from fastapi import FastAPI, UploadFile, File, Form,Query
from fastapi.responses import FileResponse
from typing import List, Optional
import os, shutil, tempfile, zipfile, gc
from pathlib import Path
# from ultralytics import YOLO
import torch
import cv2
from detector.yolo_model import detect_objects_with_prompt
from zero_detector.G_dino import G_dino_detect,merge_boxes
from segment_anything import sam_model_registry, SamPredictor
from segmentation.sam_fast import annotate_with_efficient_sam
from segmentation.sam import annotate_with_sam,segment_with_sam_clicks
from utils.image_utils import save_annotated_image
from utils.coco_format import export_to_coco, merge_coco_jsons,update_coco_json_with_flags,convert_coco_to_yolo
from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from utils.pascal import coco_to_voc
from utils.split import split_uploaded_images,get_first_crop_path
from utils.misclasification import *
# from utils.annotate_helpers import parse_points
from utils.augment import augment_images
from fastapi.middleware.cors import CORSMiddleware
import json
# from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# import io
# from typing import List
import base64
import logging
from sklearn.cluster import KMeans
# from collections import Counter
from typing import Dict, Any
# import numpy as np
from utils.clip import *
# from fastapi.responses import StreamingResponse
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()


model = None
processor = None
device = "cpu"

model, processor = load_clip_model()


def load_clip_model():
    """Load CLIP model and processor"""
    global model, processor, device
    
    logger.info("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    logger.info(f"CLIP model loaded on device: {device}")
    return model, processor
# from utils.misclasification import *
# Add this middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins like ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods including OPTIONS
    allow_headers=["*"],  # Allows all headers including Content-Type, Authorization, etc.
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_MODELS = {
    "fastsam": lambda: build_efficient_sam_vitt(),
    "sam": lambda: SamPredictor(sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth").to(DEVICE))
}
# YOLO_CLASSES = ["person", "car", "dog", "cat","truck","bus"]  # Extend this list as needed
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_clip_model()

@app.post("/annotate")
async def annotate_images(
    images: List[UploadFile] = File(...),
    classes: Optional[str] = Form(""),
    coordinates: Optional[str] = Form(None),
    sam_type: Optional[str] = Form("sam"),
    export_format: Optional[str] = Form("coco"),  # "voc" or "coco"
    split: Optional[float] = Form(0.2)
):
    output_dir = "outputs"
    crops_dir = os.path.join(output_dir, "crops")
    prompt = classes
    val_split = split
    yolo_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

    # Clean outputs (including crops)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    # Parse coordinates JSON string into Python dict
    coord_dict = None
    if coordinates:
        try:
            coord_dict = json.loads(coordinates)
        except Exception as e:
            return {"error": f"Invalid coordinates format: {str(e)}"}
    tmp_dir = tempfile.mkdtemp()
    all_annotations = []
    
    sam_type = "sam"
    try:
        predictor = SAM_MODELS[sam_type]()

        for idx, img_file in enumerate(images):
            safe_filename = os.path.basename(img_file.filename)
            img_path = os.path.join(tmp_dir, safe_filename)

            with open(img_path, "wb") as f:
                shutil.copyfileobj(img_file.file, f)

            results = []
            changed_dir = output_dir+"/results"
            print("Changed Directory:", changed_dir)
            if not os.path.exists(changed_dir):
                os.makedirs(changed_dir, exist_ok=True)
            detect_output_path = os.path.join(changed_dir, f"annotated_{Path(img_file.filename).stem}.jpg")
            # --- Flow 1: Prompt only ---
            if prompt:
                prompt_list = [p.strip().lower() for p in prompt.split(",") if p.strip()]
                yolo_class_set = set(name.lower() for name in yolo_class_names)

                if all(p in yolo_class_set for p in prompt_list):
                    # Use YOLO if all prompts exist in the YOLO class list
                    yolo_results = detect_objects_with_prompt(img_path, prompt)
                    results = yolo_results
                else:
                    # Use GroundingDINO if any prompt is not in YOLO class list
                    annotated_frame, boxes, logits, phrases = G_dino_detect(img_path, prompt)
                    cv2.imwrite("annotated.jpg", annotated_frame)
                    dino_results = [
                        {"label": phrases[i], "confidence": float(logits[i]), "bbox": list(map(int, boxes[i]))}
                        for i in range(len(phrases))
                    ]
                    results = dino_results

            # --- Flow 2: Click or Polygon (with or without prompt) ---
            elif coord_dict!= None:
                print("Click or Polygon points:", coord_dict)
                # Step 1: Segment using SAM from click/polygon
                mask_output = os.path.join(output_dir, f"masks_{img_file.filename}")
                print("Mask Output Path:", mask_output)
                if sam_type == "fastsam":
                    masks, _ = annotate_with_efficient_sam(img_path, [], "", predictor, device=DEVICE, click_points=coord_dict)
                else:
                    results = segment_with_sam_clicks(img_path, [coord_dict])
                    print("Segmented Results:", results)
                    image = cv2.imread(img_path)

                    # Bounding box in [x_min, y_min, x_max, y_max] format
                    bbox = results
                    x_min, y_min, x_max, y_max = bbox['bbox']

                    # Draw the rectangle
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
                    # Or save it
                    # cv2.imwrite('image_with_bbox.jpg', image)
                    return results
                # Step 2: Get cropped object from mask
                cropped_img_path = get_first_crop_path(crops_dir) 

                # Step 3: Run detection on the crop using G_DINO
                _, boxes, logits, phrases = G_dino_detect(cropped_img_path, prompt="")
                if boxes:
                    results = [{
                        "label": phrases[i],
                        "confidence": float(logits[i]),
                        "bbox": list(map(int, boxes[i])),
                        "mask": masks[i]  # Optional: you can encode mask as RLE or base64 PNG
                    } for i in range(len(phrases))]
                else:
                    results = []
            # Visualize click result
            # print("Boxes:", results)
            save_annotated_image(img_path, results, detect_output_path)


            img_crops_dir = os.path.join(crops_dir, Path(img_file.filename).stem)
            os.makedirs(img_crops_dir, exist_ok=True)

            mask_output = os.path.join(output_dir, f"masks_{img_file.filename}")
            if sam_type == "fastsam":
                masks, _ = annotate_with_efficient_sam(img_path, results, img_crops_dir, predictor, device=DEVICE)
            else:
                masks, _ = annotate_with_sam(img_path, results, mask_output, predictor, save_dir=img_crops_dir)

            coco_output = os.path.join(output_dir, f"coco_{Path(img_file.filename).stem}.json")
            export_to_coco(img_path, results, masks, coco_output, image_id=idx + 1)

            all_annotations.append({
                "filename": img_file.filename,
                "coco_path": coco_output
            })

        torch.cuda.empty_cache()
        gc.collect()

        split_uploaded_images(tmp_dir, output_dir, val_ratio=val_split)
        augment_images(Path(output_dir) / "original_split/train", augmentations_per_image=5)
        if export_format.lower() == "voc":
            coco_to_voc(output_dir, os.path.join(output_dir, "pascal_voc"))
        else:
            merge_coco_jsons(output_dir, os.path.join(output_dir, "coco_dataset.json"))
            convert_coco_to_yolo("outputs/coco_dataset.json", os.path.join(output_dir, "yolo_dataset"))
        # update_coco_json_with_flags("outputs/coco_dataset.json", prompt,0.4)
        try:
            model, processor = load_clip_model()
            df_clip = build_full_image_clip_df(
                images_dir="outputs/original_split/val",
                coco_json_path="outputs/coco_dataset.json",
                model=model,
                processor=processor
            )
            df_clip = predict_clip_labels(df_clip, model, processor, prompt, top_k=1)
            df_clip = flag_missing_predictions(df_clip)
            df_clip = apply_umap(df_clip)

            centroids = compute_cluster_centroids(df_clip)
            df_clip = flag_outliers(df_clip, centroids, threshold_quantile=0.95)
            df_clip["clip_top_label"] = df_clip["clip_top_labels"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown")

            visualize_embeddings(df_clip, save_path="outputs/umap_fullimage.html")
        except Exception as e:
            print(f"Error processing CLIP embeddings: {str(e)}")    
        # Otherwise, return a ZIP
        zip_path = "outputs.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="outputs.zip",
            headers={"X-Annotation-Count": str(len(all_annotations))}
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
# Add these imports to your existing imports
from PIL import Image


@app.post("/update_annotations")
async def update_annotations(
    images: List[UploadFile] = File(...),
    annotations: str = Form(...),  # Updated COCO JSON as string
    export_format: Optional[str] = Form("coco"),
    split: Optional[float] = Form(0.2),
    augment: Optional[bool] = Form(True),
    augmentations_per_image: Optional[int] = Form(5)
):
    """
    Update annotations with user modifications and generate updated dataset
    
    Args:
        images: Original images (should match the ones from /annotate)
        annotations: Updated COCO format JSON string with user modifications
        export_format: "coco" or "voc" 
        split: Train/validation split ratio
        augment: Whether to apply data augmentation
        augmentations_per_image: Number of augmentations per training image
    """
    
    output_dir = "outputs_updated"
    crops_dir = os.path.join(output_dir, "crops")
    
    # Clean previous outputs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # Parse the updated annotations
        try:
            coco_data = json.loads(annotations)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON format in annotations: {str(e)}"}
        
        # Validate required COCO fields
        required_fields = ["images", "annotations", "categories"]
        for field in required_fields:
            if field not in coco_data:
                return {"error": f"Missing required field in COCO data: {field}"}
        
        # Create mappings for easy lookup
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}
        category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        # filename_to_upload = {img.filename: img for img in images}
        filename_to_upload = {Path(img.filename).name: img for img in images}

        # Create results directory
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each image
        processed_images = []
        for img_info in coco_data["images"]:
            filename = img_info["file_name"]
            
            if filename not in filename_to_upload:
                print(f"Warning: Image {filename} not found in uploaded files")
                continue
                
            uploaded_file = filename_to_upload[filename]
            
            # Save original image to tmp and results directories
            safe_filename = os.path.basename(filename)
            img_path = os.path.join(tmp_dir, safe_filename)
            results_path = os.path.join(results_dir, safe_filename)
            
            file_bytes = await uploaded_file.read()

            with open(img_path, "wb") as f:
                f.write(file_bytes)

            with open(results_path, "wb") as f:
                f.write(file_bytes)

            
            # Get annotations for this image
            img_annotations = [ann for ann in coco_data["annotations"] 
                             if ann["image_id"] == img_info["id"]]
            
            if img_annotations:
                # Convert COCO annotations to the format expected by your existing functions
                results_format = []
                for ann in img_annotations:
                    x, y, w, h = ann["bbox"]
                    category_name = category_id_to_name.get(ann["category_id"], "unknown")
                    
                    result = {
                        "label": category_name,
                        "confidence": ann.get("score", 1.0),  # Use score if available, default to 1.0
                        "bbox": [int(x), int(y), int(x + w), int(y + h)]  # Convert to [x1, y1, x2, y2]
                    }
                    results_format.append(result)
                
                # Create annotated visualization
                annotated_path = os.path.join(results_dir, f"annotated_{Path(filename).stem}.jpg")
                save_annotated_image(img_path, results_format, annotated_path)
                
                # Create crops directory for this image
                img_crops_dir = os.path.join(crops_dir, Path(filename).stem)
                os.makedirs(img_crops_dir, exist_ok=True)
                
                # Generate crops from bounding boxes
                create_crops_from_annotations(img_path, img_annotations, category_id_to_name, img_crops_dir)
                
                processed_images.append({
                    "filename": filename,
                    "image_id": img_info["id"],
                    "annotations_count": len(img_annotations)
                })
        
        # Create the complete COCO dataset file
        coco_output_path = os.path.join(output_dir, "annotations.json")
        with open(coco_output_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        
        # Apply train/validation split if requested
        if split > 0:
            split_updated_dataset(tmp_dir, output_dir, coco_data, val_ratio=split)
            
            # Apply augmentation to training set if requested
            if augment:
                train_images_dir = Path(output_dir) / "split" / "train" / "images"
                if train_images_dir.exists():
                    augment_images(train_images_dir, augmentations_per_image=augmentations_per_image)
        
        # Convert to VOC format if requested
        if export_format.lower() == "voc":
            voc_output_dir = os.path.join(output_dir, "pascal_voc")
            convert_coco_to_voc_format(coco_data, results_dir, voc_output_dir)
        
        # Create ZIP file with all contents
        zip_path = "outputs_updated.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from output directory
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create archive path relative to output_dir
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
                    
            # Verify crops are being added
            crops_added = 0
            for root, dirs, files in os.walk(crops_dir):
                crops_added += len(files)
            print(f"Added {crops_added} crop files to ZIP")
            
            # Verify images are being added  
            images_added = 0
            for root, dirs, files in os.walk(results_dir):
                images_added += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"Added {images_added} image files to ZIP")
        
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="updated_dataset.zip",
            headers={
                "X-Images-Processed": str(len(processed_images)),
                "X-Total-Annotations": str(len(coco_data["annotations"])),
                "X-Export-Format": export_format
            }
        )
        
    except Exception as e:
        return {"error": f"Error processing annotations: {str(e)}"}
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)




@app.post("/search_images")
async def search_images(
    description: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Process multiple images and return cosine similarity with search query
    
    Args:
        description: Text query to search for (renamed from search_query)
        images: List of image files in binary format
    
    Returns:
        JSON with similarity scores for each image
    """
    try:
        if not description.strip():
            return JSONResponse(
                status_code=400, 
                content={"error": "Description cannot be empty"}
            )
        
        if not images:
            return JSONResponse(
                status_code=400, 
                content={"error": "No images provided"}
            )
        
        logger.info(f"Processing {len(images)} images with description: '{description}'")
        
        # Extract text embedding for search query
        text_embedding = extract_text_embedding(description, model, processor)
        if text_embedding is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "Failed to process description"}
            )
        
        results = []
        
        # Process each image
        for idx, image_file in enumerate(images):
            try:
                # Read image bytes
                image_bytes = await image_file.read()
                
                # Extract image embedding
                image_embedding = extract_image_embedding(image_bytes, model, processor)
                
                if image_embedding is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [text_embedding], 
                        [image_embedding]
                    )[0][0]
                    
                    results.append({
                        "image_index": idx,
                        "filename": image_file.filename,
                        "similarity_score": float(similarity),
                        "status": "success"
                    })
                else:
                    results.append({
                        "image_index": idx,
                        "filename": image_file.filename,
                        "similarity_score": 0.0,
                        "status": "error",
                        "error_message": "Failed to extract image embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append({
                    "image_index": idx,
                    "filename": image_file.filename if image_file else f"image_{idx}",
                    "similarity_score": 0.0,
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return JSONResponse(content={
            "description": description,
            "total_images": len(images),
            "processed_images": len([r for r in results if r["status"] == "success"]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/reverse_search")
async def reverse_search(
    reference_image: UploadFile = File(...),
    images: List[UploadFile] = File(...)
):
    """
    Reverse image search: Find similar images to a reference image
    
    Args:
        reference_image: The reference image to search for similar images
        images: List of image files to search through
    
    Returns:
        JSON with similarity scores for each image compared to reference
    """
    try:
        if not images:
            return JSONResponse(
                status_code=400, 
                content={"error": "No images provided for comparison"}
            )
        
        logger.info(f"Processing reverse search with reference image '{reference_image.filename}' against {len(images)} images")
        
        # Extract embedding for reference image
        reference_bytes = await reference_image.read()
        reference_embedding = extract_image_embedding(reference_bytes, model, processor)
        
        if reference_embedding is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "Failed to process reference image"}
            )
        
        results = []
        
        # Process each comparison image
        for idx, image_file in enumerate(images):
            try:
                # Read image bytes
                image_bytes = await image_file.read()
                
                # Extract image embedding
                image_embedding = extract_image_embedding(image_bytes, model, processor)
                
                if image_embedding is not None:
                    # Calculate cosine similarity between reference and current image
                    similarity = cosine_similarity(
                        [reference_embedding], 
                        [image_embedding]
                    )[0][0]
                    
                    results.append({
                        "image_index": idx,
                        "filename": image_file.filename,
                        "similarity_score": float(similarity),
                        "status": "success"
                    })
                else:
                    results.append({
                        "image_index": idx,
                        "filename": image_file.filename,
                        "similarity_score": 0.0,
                        "status": "error",
                        "error_message": "Failed to extract image embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append({
                    "image_index": idx,
                    "filename": image_file.filename if image_file else f"image_{idx}",
                    "similarity_score": 0.0,
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return JSONResponse(content={
            "reference_image": reference_image.filename,
            "total_images": len(images),
            "processed_images": len([r for r in results if r["status"] == "success"]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in reverse search: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/reverse_search_base64")
async def reverse_search_base64(
    request_data: dict
):
    """
    Reverse image search with base64 encoded images
    
    Expected format:
    {
        "reference_image": {
            "filename": "reference.jpg",
            "data": "base64_encoded_reference_image"
        },
        "images": [
            {
                "filename": "image1.jpg",
                "data": "base64_encoded_image_data"
            },
            ...
        ]
    }
    """
    try:
        reference_data = request_data.get("reference_image", {})
        images_data = request_data.get("images", [])
        
        if not reference_data or not reference_data.get("data"):
            return JSONResponse(
                status_code=400, 
                content={"error": "Reference image is required"}
            )
        
        if not images_data:
            return JSONResponse(
                status_code=400, 
                content={"error": "No images provided for comparison"}
            )
        
        reference_filename = reference_data.get("filename", "reference_image")
        logger.info(f"Processing reverse search with reference image '{reference_filename}' against {len(images_data)} base64 images")
        
        # Decode and process reference image
        reference_bytes = base64.b64decode(reference_data.get("data"))
        reference_embedding = extract_image_embedding(reference_bytes, model, processor)
        
        if reference_embedding is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "Failed to process reference image"}
            )
        
        results = []
        
        # Process each comparison image
        for idx, image_data in enumerate(images_data):
            try:
                filename = image_data.get("filename", f"image_{idx}")
                base64_data = image_data.get("data", "")
                
                # Decode base64 image
                image_bytes = base64.b64decode(base64_data)
                
                # Extract image embedding
                image_embedding = extract_image_embedding(image_bytes, model, processor)
                
                if image_embedding is not None:
                    # Calculate cosine similarity between reference and current image
                    similarity = cosine_similarity(
                        [reference_embedding], 
                        [image_embedding]
                    )[0][0]
                    
                    results.append({
                        "image_index": idx,
                        "filename": filename,
                        "similarity_score": float(similarity),
                        "status": "success"
                    })
                else:
                    results.append({
                        "image_index": idx,
                        "filename": filename,
                        "similarity_score": 0.0,
                        "status": "error",
                        "error_message": "Failed to extract image embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append({
                    "image_index": idx,
                    "filename": image_data.get("filename", f"image_{idx}"),
                    "similarity_score": 0.0,
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return JSONResponse(content={
            "reference_image": reference_filename,
            "total_images": len(images_data),
            "processed_images": len([r for r in results if r["status"] == "success"]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in reverse search: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/search_images_base64")
async def search_images_base64(
    request_data: dict
):
    """
    Alternative endpoint that accepts base64 encoded images
    
    Expected format:
    {
        "description": "your search text",
        "images": [
            {
                "filename": "image1.jpg",
                "data": "base64_encoded_image_data"
            },
            ...
        ]
    }
    """
    try:
        description = request_data.get("description", "").strip()
        images_data = request_data.get("images", [])
        
        if not description:
            return JSONResponse(
                status_code=400, 
                content={"error": "Description cannot be empty"}
            )
        
        if not images_data:
            return JSONResponse(
                status_code=400, 
                content={"error": "No images provided"}
            )
        
        logger.info(f"Processing {len(images_data)} base64 images with description: '{description}'")
        
        # Extract text embedding for search query
        text_embedding = extract_text_embedding(description, model, processor)
        if text_embedding is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "Failed to process description"}
            )
        
        results = []
        
        # Process each image
        for idx, image_data in enumerate(images_data):
            try:
                filename = image_data.get("filename", f"image_{idx}")
                base64_data = image_data.get("data", "")
                
                # Decode base64 image
                image_bytes = base64.b64decode(base64_data)
                
                # Extract image embedding
                image_embedding = extract_image_embedding(image_bytes, model, processor)
                
                if image_embedding is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [text_embedding], 
                        [image_embedding]
                    )[0][0]
                    
                    results.append({
                        "image_index": idx,
                        "filename": filename,
                        "similarity_score": float(similarity),
                        "status": "success"
                    })
                else:
                    results.append({
                        "image_index": idx,
                        "filename": filename,
                        "similarity_score": 0.0,
                        "status": "error",
                        "error_message": "Failed to extract image embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results.append({
                    "image_index": idx,
                    "filename": image_data.get("filename", f"image_{idx}"),
                    "similarity_score": 0.0,
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return JSONResponse(content={
            "description": description,
            "total_images": len(images_data),
            "processed_images": len([r for r in results if r["status"] == "success"]),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}


        )
    

@app.post("/find_duplicates")
async def find_duplicates(
    images: List[UploadFile] = File(...),
    similarity_threshold: float = Form(0.95)
):
    """
    Find duplicate/similar images and group them together
    
    Args:
        images: List of image files to analyze for duplicates
        similarity_threshold: Threshold for considering images as duplicates (0.0 to 1.0)
    
    Returns:
        JSON with groups of duplicate images
    """
    try:
        if not images:
            return JSONResponse(
                status_code=400, 
                content={"error": "No images provided"}
            )
        
        if not (0.0 <= similarity_threshold <= 1.0):
            return JSONResponse(
                status_code=400, 
                content={"error": "Similarity threshold must be between 0.0 and 1.0"}
            )
        
        logger.info(f"Finding duplicates in {len(images)} images with threshold: {similarity_threshold}")
        
        # Extract embeddings for all images
        image_embeddings = []
        image_info = []
        
        for idx, image_file in enumerate(images):
            try:
                # Read image bytes
                image_bytes = await image_file.read()
                
                # Extract image embedding
                embedding = extract_image_embedding(image_bytes, model, processor)
                
                if embedding is not None:
                    image_embeddings.append(embedding)
                    image_info.append({
                        "index": idx,
                        "filename": image_file.filename,
                        "status": "success"
                    })
                else:
                    image_info.append({
                        "index": idx,
                        "filename": image_file.filename,
                        "status": "error",
                        "error_message": "Failed to extract embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                image_info.append({
                    "index": idx,
                    "filename": image_file.filename if image_file else f"image_{idx}",
                    "status": "error",
                    "error_message": str(e)
                })
        
        if len(image_embeddings) < 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Need at least 2 valid images to find duplicates"}
            )
        
        # Calculate similarity matrix for valid embeddings only
        valid_images = [info for info in image_info if info["status"] == "success"]
        similarity_matrix = cosine_similarity(image_embeddings)
        
        # Find duplicate groups using the specified logic
        duplicate_groups = []
        processed = set()
        
        for i in range(len(valid_images)):
            if i in processed:
                continue
                
            # Find all images with similarity > threshold (not >=)
            duplicates = [i]
            for j in range(len(valid_images)):
                if i != j and similarity_matrix[i, j] > similarity_threshold:
                    duplicates.append(j)
            
            # If duplicates found, add to groups
            if len(duplicates) > 1:
                # Create group info for each duplicate
                group_info = []
                for idx in duplicates:
                    group_info.append({
                        "original_index": valid_images[idx]["index"],
                        "filename": valid_images[idx]["filename"]
                    })
                
                # Calculate average similarity within the group
                group_similarities = []
                for idx1 in duplicates:
                    for idx2 in duplicates:
                        if idx1 != idx2:
                            group_similarities.append(similarity_matrix[idx1][idx2])
                
                avg_similarity = sum(group_similarities) / len(group_similarities) if group_similarities else 0.0
                
                duplicate_groups.append({
                    "group_id": len(duplicate_groups) + 1,
                    "image_count": len(duplicates),
                    "average_similarity": float(avg_similarity),
                    "images": group_info
                })
                
                # Mark all duplicates as processed
                processed.update(duplicates)
        
        # Find images with no duplicates (singletons)
        singleton_images = []
        for info in valid_images:
            is_singleton = True
            for group in duplicate_groups:
                if any(img["original_index"] == info["index"] for img in group["images"]):
                    is_singleton = False
                    break
            
            if is_singleton:
                singleton_images.append({
                    "original_index": info["index"],
                    "filename": info["filename"]
                })
        
        # Collect error images
        error_images = [info for info in image_info if info["status"] == "error"]
        
        # Format response as requested: group_1, group_2, etc.
        formatted_groups = {}
        for i, group in enumerate(duplicate_groups, 1):
            image_paths = [img["filename"] for img in group["images"]]
            formatted_groups[f"group_{i}"] = image_paths
        
        return JSONResponse(content=formatted_groups)
        
    except Exception as e:
        logger.error(f"Unexpected error in duplicate detection: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )


@app.post("/find_duplicates_base64")
async def find_duplicates_base64(
    request_data: dict
):
    """
    Find duplicate/similar images using base64 encoded images
    
    Expected format:
    {
        "images": [
            {
                "filename": "image1.jpg",
                "data": "base64_encoded_image_data"
            },
            ...
        ],
        "similarity_threshold": 0.95
    }
    """
    try:
        images_data = request_data.get("images", [])
        similarity_threshold = request_data.get("similarity_threshold", 0.95)
        
        if not images_data:
            return JSONResponse(
                status_code=400, 
                content={"error": "No images provided"}
            )
        
        if not (0.0 <= similarity_threshold <= 1.0):
            return JSONResponse(
                status_code=400, 
                content={"error": "Similarity threshold must be between 0.0 and 1.0"}
            )
        
        logger.info(f"Finding duplicates in {len(images_data)} base64 images with threshold: {similarity_threshold}")
        
        # Extract embeddings for all images
        image_embeddings = []
        image_info = []
        
        for idx, image_data in enumerate(images_data):
            try:
                filename = image_data.get("filename", f"image_{idx}")
                base64_data = image_data.get("data", "")
                
                # Decode base64 image
                image_bytes = base64.b64decode(base64_data)
                
                # Extract image embedding
                embedding = extract_image_embedding(image_bytes, model, processor)
                
                if embedding is not None:
                    image_embeddings.append(embedding)
                    image_info.append({
                        "index": idx,
                        "filename": filename,
                        "status": "success"
                    })
                else:
                    image_info.append({
                        "index": idx,
                        "filename": filename,
                        "status": "error",
                        "error_message": "Failed to extract embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                image_info.append({
                    "index": idx,
                    "filename": image_data.get("filename", f"image_{idx}"),
                    "status": "error",
                    "error_message": str(e)
                })
        
        if len(image_embeddings) < 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Need at least 2 valid images to find duplicates"}
            )
        
        # Calculate similarity matrix for valid embeddings only
        valid_images = [info for info in image_info if info["status"] == "success"]
        similarity_matrix = cosine_similarity(image_embeddings)
        
        # Find duplicate groups using the specified logic
        duplicate_groups = []
        processed = set()
        
        for i in range(len(valid_images)):
            if i in processed:
                continue
                
            # Find all images with similarity > threshold (not >=)
            duplicates = [i]
            for j in range(len(valid_images)):
                if i != j and similarity_matrix[i, j] > similarity_threshold:
                    duplicates.append(j)
            
            # If duplicates found, add to groups
            if len(duplicates) > 1:
                # Create group info for each duplicate
                group_info = []
                for idx in duplicates:
                    group_info.append({
                        "original_index": valid_images[idx]["index"],
                        "filename": valid_images[idx]["filename"]
                    })
                
                # Calculate average similarity within the group
                group_similarities = []
                for idx1 in duplicates:
                    for idx2 in duplicates:
                        if idx1 != idx2:
                            group_similarities.append(similarity_matrix[idx1][idx2])
                
                avg_similarity = sum(group_similarities) / len(group_similarities) if group_similarities else 0.0
                
                duplicate_groups.append({
                    "group_id": len(duplicate_groups) + 1,
                    "image_count": len(duplicates),
                    "average_similarity": float(avg_similarity),
                    "images": group_info
                })
                
                # Mark all duplicates as processed
                processed.update(duplicates)
        
        # Find images with no duplicates (singletons)
        singleton_images = []
        for info in valid_images:
            is_singleton = True
            for group in duplicate_groups:
                if any(img["original_index"] == info["index"] for img in group["images"]):
                    is_singleton = False
                    break
            
            if is_singleton:
                singleton_images.append({
                    "original_index": info["index"],
                    "filename": info["filename"]
                })
        
        # Collect error images
        error_images = [info for info in image_info if info["status"] == "error"]
        
        # Format response as requested: group_1, group_2, etc.
        formatted_groups = {}
        for i, group in enumerate(duplicate_groups, 1):
            image_paths = [img["filename"] for img in group["images"]]
            formatted_groups[f"group_{i}"] = image_paths
        
        return JSONResponse(content=formatted_groups)
        
    except Exception as e:
        logger.error(f"Unexpected error in duplicate detection: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )


def generate_cluster_name(image_embeddings: List[np.ndarray], model, processor) -> str:
    """
    Generate a descriptive name for a cluster based on image embeddings
    using zero-shot classification with CLIP
    """
    # Candidate labels for classification
    candidate_labels = [
        "landscape", "portrait", "architecture", "food", "animals", "people",
        "nature", "urban", "indoor", "outdoor", "abstract", "text",
        "vehicles", "art", "technology", "sports", "black and white", "colorful",
        "water", "sky", "night", "day", "flowers", "beach", "mountains",
        "forest", "sunset", "buildings", "pets", "wildlife", "close-up"
    ]
    
    # For clustering, we'll use the average embedding to represent the cluster
    # and classify it against our candidate labels
    try:
        # Calculate average embedding for the cluster
        avg_embedding = np.mean(image_embeddings, axis=0)
        
        # Get text embeddings for candidate labels
        text_inputs = processor(text=candidate_labels, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
            text_embeddings = text_embeddings.cpu().numpy()
        
        # Calculate similarity between average cluster embedding and text embeddings
        similarities = cosine_similarity([avg_embedding], text_embeddings)[0]
        
        # Get top 2 most similar labels
        top_indices = np.argsort(similarities)[-2:][::-1]
        top_labels = [(candidate_labels[i], similarities[i]) for i in top_indices]
        
        # Create cluster name based on top labels
        if len(top_labels) > 1 and top_labels[0][1] > 1.5 * top_labels[1][1]:
            cluster_name = top_labels[0][0].capitalize()
        elif len(top_labels) > 1:
            cluster_name = f"{top_labels[0][0].capitalize()} & {top_labels[1][0]}"
        else:
            cluster_name = top_labels[0][0].capitalize()
            
        return cluster_name
        
    except Exception as e:
        logger.error(f"Error generating cluster name: {e}")
        return "Miscellaneous"

def cluster_images_by_embeddings(embeddings: List[np.ndarray], image_info: List[Dict], num_clusters: int = 5) -> Dict[str, Any]:
    """
    Cluster images based on their embeddings using KMeans
    Returns: Dictionary mapping cluster names to image information
    """
    try:
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Adjust number of clusters if we have fewer images
        actual_num_clusters = min(num_clusters, len(embeddings))
        if actual_num_clusters < 2:
            actual_num_clusters = 2
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Group images by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    "embeddings": [],
                    "images": []
                }
            clusters[label]["embeddings"].append(embeddings[i])
            clusters[label]["images"].append(image_info[i])
        
        # Generate names for each cluster and format response
        named_clusters = {}
        for cluster_id, cluster_data in clusters.items():
            cluster_name = generate_cluster_name(cluster_data["embeddings"], model, processor)
            
            # Format images information
            cluster_images = []
            for img in cluster_data["images"]:
                cluster_images.append({
                    "filename": img["filename"],
                    "image_index": img["image_index"]
                })
            
            named_clusters[cluster_name] = {
                "cluster_id": int(cluster_id),
                "image_count": len(cluster_images),
                "images": cluster_images
            }
        
        return named_clusters
        
    except Exception as e:
        logger.error(f"Error in clustering: {e}")
        raise e

@app.post("/cluster_images")
async def cluster_images(
    images: List[UploadFile] = File(...),
    num_clusters: int = Form(5)
):
    """
    Cluster uploaded images based on visual similarity using CLIP embeddings
    
    Args:
        images: List of image files to cluster
        num_clusters: Number of clusters to create (default: 5)
    
    Returns:
        JSON with cluster names and images in each cluster
    """
    try:
        if not images:
            return JSONResponse(
                status_code=400,
                content={"error": "No images provided"}
            )
        
        if num_clusters < 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Number of clusters must be at least 2"}
            )
        
        logger.info(f"Clustering {len(images)} images into {num_clusters} clusters")
        
        # Extract embeddings for all images
        embeddings = []
        image_info = []
        failed_images = []
        
        for idx, image_file in enumerate(images):
            try:
                # Read image bytes
                image_bytes = await image_file.read()
                
                # Extract image embedding
                embedding = extract_image_embedding(image_bytes, model, processor)
                
                if embedding is not None:
                    embeddings.append(embedding)
                    image_info.append({
                        "filename": image_file.filename,
                        "image_index": idx
                    })
                else:
                    failed_images.append({
                        "filename": image_file.filename,
                        "image_index": idx,
                        "error": "Failed to extract embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                failed_images.append({
                    "filename": image_file.filename if image_file else f"image_{idx}",
                    "image_index": idx,
                    "error": str(e)
                })
        
        if len(embeddings) < 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Need at least 2 valid images to perform clustering"}
            )
        
        # Perform clustering
        clusters = cluster_images_by_embeddings(embeddings, image_info, num_clusters)
        
        return JSONResponse(content={
            "total_images": len(images),
            "processed_images": len(embeddings),
            "failed_images": len(failed_images),
            "num_clusters": len(clusters),
            "clusters": clusters,
            "failed_images_details": failed_images if failed_images else None
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in clustering: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/cluster_images_base64")
async def cluster_images_base64(request_data: dict):
    """
    Cluster base64 encoded images based on visual similarity
    
    Expected format:
    {
        "images": [
            {
                "filename": "image1.jpg",
                "data": "base64_encoded_image_data"
            },
            ...
        ],
        "num_clusters": 5
    }
    """
    try:
        images_data = request_data.get("images", [])
        num_clusters = request_data.get("num_clusters", 5)
        
        if not images_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No images provided"}
            )
        
        if num_clusters < 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Number of clusters must be at least 2"}
            )
        
        logger.info(f"Clustering {len(images_data)} base64 images into {num_clusters} clusters")
        
        # Extract embeddings for all images
        embeddings = []
        image_info = []
        failed_images = []
        
        for idx, image_data in enumerate(images_data):
            try:
                filename = image_data.get("filename", f"image_{idx}")
                base64_data = image_data.get("data", "")
                
                # Decode base64 image
                image_bytes = base64.b64decode(base64_data)
                
                # Extract image embedding
                embedding = extract_image_embedding(image_bytes, model, processor)
                
                if embedding is not None:
                    embeddings.append(embedding)
                    image_info.append({
                        "filename": filename,
                        "image_index": idx
                    })
                else:
                    failed_images.append({
                        "filename": filename,
                        "image_index": idx,
                        "error": "Failed to extract embedding"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                failed_images.append({
                    "filename": image_data.get("filename", f"image_{idx}"),
                    "image_index": idx,
                    "error": str(e)
                })
        
        if len(embeddings) < 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Need at least 2 valid images to perform clustering"}
            )
        
        # Perform clustering
        clusters = cluster_images_by_embeddings(embeddings, image_info, num_clusters)
        
        return JSONResponse(content={
            "total_images": len(images_data),
            "processed_images": len(embeddings),
            "failed_images": len(failed_images),
            "num_clusters": len(clusters),
            "clusters": clusters,
            "failed_images_details": failed_images if failed_images else None
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in clustering: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CLIP Image Search API",
        "endpoints": {
            "/search_images": "POST - Upload images and description (multipart/form-data)",
            "/search_images_base64": "POST - Send base64 encoded images (JSON)",
            "/reverse_search": "POST - Find similar images to reference image (multipart/form-data)",
            "/reverse_search_base64": "POST - Reverse search with base64 encoded images (JSON)",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # load_clip_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)