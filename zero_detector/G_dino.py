from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.config import GroundingDINO_SwinT_OGC
import torch
# import os
# import supervision as sv
# import numpy as np

WEIGHTS_NAME = "models/groundingdino_swint_ogc.pth"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.1
GroundingDINO_SwinT_OGC ="zero_detector/config_dino.py"
model = load_model(GroundingDINO_SwinT_OGC, WEIGHTS_NAME)
import re

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()


def clean_and_lemmatize(p):
    clean = re.sub(r"<[^>]+>", "", p).strip().lower()
    return " ".join([lemmatizer.lemmatize(word) for word in clean.split()])



def G_dino_detect(image_path, text_prompt):
    print("Grounding DINO",text_prompt)
    text_prompt = text_prompt.replace("_", " ")
    # Lemmatize prompt
    # raw_classes = [x.strip().lower() for x in re.split(r"[;,]", text_prompt.replace("<OD>", ""))]
    # lemmatized_prompt = ", ".join([lemmatizer.lemmatize(cls) for cls in raw_classes if cls])
    # text_prompt = lemmatized_prompt
    print(f"prompt: {text_prompt}")
    """
    Detects objects in an image based on a text prompt using the Grounding DINO model."""
    image_source, image = load_image(image_path)
    # Ensure the <OD> token is included in custom prompts
    # if "<OD>" not in text_prompt:
    #     text_prompt = "<OD> " + text_prompt
        # print(f"Updated text prompt: {text_prompt}")
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    print(phrases)
    # clean_phrases = [clean_and_lemmatize(p) for p in phrases]
    # print(f"Clean phrases: {clean_phrases}")
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # Convert normalized boxes to absolute pixel coordinates
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    xy_boxes = boxes.numpy()
    return annotated_frame, xy_boxes, logits, phrases


# import cv2
# from pathlib import Path
# from ultralytics import YOLO
# from utils.prompt_validator import is_valid_yolo_class

# # Get the root directory (project/)
# root_dir = Path(_file_).resolve().parent.parent
# model_path = root_dir /"models"/"yolov8x-worldv2.pt"

# model = YOLO(str(model_path))
# # Load YOLO model once globally
# # model = YOLO("/models/yolov8x-worldv2.pt")  # You can change to yolov8s.pt, etc.

# def detect_objects_with_prompt(image_path: str, prompt: str):
#     print("yolo")
#     """
#     Detects objects in an image based on a prompt (comma-separated classes).
    
#     Args:
#         image_path (str): Path to the input image.
#         prompt (str): Comma-separated class names to detect (e.g., "person, car").
    
#     Returns:
#         List[Dict]: List of detected objects matching the prompt.
#     """
#     # Parse and clean prompt
#     requested_classes = [cls.strip().lower() for cls in prompt.split(",")]

#     valid, invalid = is_valid_yolo_class(requested_classes, model)

#     print("✅ Valid:", valid)
#     print("❌ Invalid:", invalid)
    
#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Failed to load image: {image_path}")

#     # Run YOLO inference
#     results = model(image)[0]
#     # print("dedede",results)
#     # Get predictions filtered by prompt classes
#     detections = []
#     for box in results.boxes:
#         class_id = int(box.cls)
#         class_name = model.names[class_id].lower()

#         if class_name in requested_classes:
#             xyxy = box.xyxy[0].cpu().numpy().tolist()
#             conf = float(box.conf)
#             detections.append({
#                 "label": class_name,
#                 "confidence": round(conf, 3),
#                 "bbox": xyxy  # Format: [x1, y1, x2, y2]
#             })
#     print(detections)
#     return detections

def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# def consensus_detection(image_path, prompt, iou_threshold=0.5):
#     # Run both models
#     yolo_detections = detect_objects_with_prompt(image_path, prompt)
#     _, dino_boxes, _, dino_phrases = G_dino_detect(image_path, prompt)

#     # Normalize Grounding DINO output to match YOLO format
#     dino_detections = []
#     for box, label in zip(dino_boxes, dino_phrases):
#         label = label.lower().strip()
#         box = box.tolist()
#         dino_detections.append({
#             "label": label,
#             "bbox": box
#         })

#     # Match YOLO and DINO detections
#     consensus = []
#     for yolo_det in yolo_detections:
#         for dino_det in dino_detections:
#             if yolo_det["label"] in dino_det["label"]:  # loose match (can make stricter)
#                 iou = compute_iou(yolo_det["bbox"], dino_det["bbox"])
#                 if iou >= iou_threshold:
#                     consensus.append({
#                         "label": yolo_det["label"],
#                         "bbox": yolo_det["bbox"],
#                         "confidence": yolo_det["confidence"],
#                         "iou_with_dino": round(iou, 2)
#                     })
#                     break  # avoid duplicate matches

#     return consensus

def merge_boxes(detections, iou_threshold=0.6, conf_threshold=0.45):
    """
    Merge overlapping bounding boxes from multiple detections with confidence filtering.
    
    Args:
        detections (list of dict): Each dict contains keys: 'label', 'confidence', 'bbox'
        iou_threshold (float): IoU threshold for considering two boxes as overlapping.
        conf_threshold (float): Minimum confidence to keep a detection.
        
    Returns:
        merged_results (list of dict): Merged detections with combined boxes.
    """
    
    # Step 1: Filter out low confidence detections
    detections = [d for d in detections if d['confidence'] >= conf_threshold]
    
    merged_results = []
    used = [False] * len(detections)
    
    for i, det_i in enumerate(detections):
        if used[i]:
            continue
        
        label_i = det_i['label']
        bbox_i = det_i['bbox']
        conf_i = det_i['confidence']
        
        # Initialize merge group
        merged_bbox = bbox_i[:]
        merged_conf = conf_i
        count = 1
        
        for j, det_j in enumerate(detections[i+1:], start=i+1):
            if used[j]:
                continue
            
            if det_j['label'] == label_i:
                iou = compute_iou(bbox_i, det_j['bbox'])
                if iou >= iou_threshold:
                    # Merge boxes by averaging coordinates weighted by confidence
                    conf_j = det_j['confidence']
                    merged_bbox = [
                        (merged_bbox[k]*merged_conf + det_j['bbox'][k]*conf_j) / (merged_conf + conf_j)
                        for k in range(4)
                    ]
                    merged_conf += conf_j
                    count += 1
                    used[j] = True
        
        # Average the confidence
        avg_conf = merged_conf / count
        
        merged_results.append({
            "label": label_i,
            "confidence": avg_conf,
            "bbox": [int(coord) for coord in merged_bbox]
        })
        
        used[i] = True
    
    return merged_results