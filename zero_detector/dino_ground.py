import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from torch.amp import autocast
class GroundingDinoHF:
    def __init__(self, model_name="IDEA-Research/grounding-dino-tiny", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(self.device)

    def detect(self, image_path: str, prompt: str, threshold: float = 0.35):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            with autocast(device_type=self.device if self.device == "cuda" else "cpu"):
                outputs = self.model(**inputs)
        # outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        print("results", results)
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "label": self.model.config.id2label.get(label.item(), str(label.item())),
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2]
            })

        for box, score, text_label in zip(results["boxes"], results["scores"], results["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")

        return detections
    
# dino_detector = GroundingDinoHF()

# prompt = [prompt]
# detections = dino_detector.detect(image, prompt)
# print("detections", detections)

# Run with Object Detection task
# task_prompt = '<OD>'
# results = florence_detect(task_prompt)
# print(results)    

