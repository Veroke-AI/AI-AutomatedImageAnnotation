import cv2
from pathlib import Path
from ultralytics import YOLO
from utils.prompt_validator import is_valid_yolo_class

# Get the root directory (project/)
root_dir = Path(__file__).resolve().parent.parent
model_path = root_dir /"models"/"yolov8x-worldv2.pt"

model = YOLO(str(model_path))
# Load YOLO model once globally
# model = YOLO("/models/yolov8x-worldv2.pt")  # You can change to yolov8s.pt, etc.

def detect_objects_with_prompt(image_path: str, prompt: str):
    print("yolo")
    """
    Detects objects in an image based on a prompt (comma-separated classes).
    
    Args:
        image_path (str): Path to the input image.
        prompt (str): Comma-separated class names to detect (e.g., "person, car").
    
    Returns:
        List[Dict]: List of detected objects matching the prompt.
    """
    # Parse and clean prompt
    requested_classes = [cls.strip().lower() for cls in prompt.split(",")]

    valid, invalid = is_valid_yolo_class(requested_classes, model)

    print("✅ Valid:", valid)
    print("❌ Invalid:", invalid)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Run YOLO inference
    results = model(image)[0]
    # print("dedede",results)
    # Get predictions filtered by prompt classes
    detections = []
    for box in results.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id].lower()

        if class_name in requested_classes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf)
            detections.append({
                "label": class_name,
                "confidence": round(conf, 3),
                "bbox": xyxy  # Format: [x1, y1, x2, y2]
            })
    print(detections)
    return detections




