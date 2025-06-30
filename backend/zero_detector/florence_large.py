import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# model_id = 'microsoft/Florence-2-base'
# model_id = 'microsoft/Florence-2-large'
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     torch_dtype='auto'
# ).eval().cuda()
# import supervision as sv
from PIL import Image
# CHECKPOINT = "microsoft/Florence-2-base-ft"
CHECKPOINT = "microsoft/Florence-2-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
image = Image.open(f"static/download2.jpg")
text = "<OD>"
task = "<OD>"

inputs = processor(text=text, images=image, return_tensors="pt").to(DEVICE)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
response = processor.post_process_generation(generated_text, task=task, image_size=image.size)
print(f"Generated text: {generated_text}")
print(f"Response: {response}")


# detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

# bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
# label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

# image = bounding_box_annotator.annotate(image, detections)
# image = label_annotator.annotate(image, detections)
# image.thumbnail((600, 600))
# image
# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def detect_classes_bboxes_fast(image_path, class_names=None):
    if class_names is None:
        class_names = []
    if isinstance(class_names, str):
        class_names = [class_names]
    class_names = [cls.lower() for cls in class_names]

    image = Image.open(image_path).convert("RGB")

    if len(class_names) == 0:
        task_prompt = "Detect all objects in the image with their bounding boxes and scores."
    else:
        classes_string = ", ".join(class_names)
        task_prompt = f"Detect {classes_string} in the image. Provide bounding boxes and scores."

    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to('cuda', torch.float16)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=512,
        do_sample=False,
        num_beams=1,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"Generated text: {generated_text}")
    # Match pattern with 4 <loc_*> values and optional score
    pattern = re.compile(
        r'([a-zA-Z0-9_]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>\s*(?:\(score:\s*([0-9.]+)\))?',
        re.IGNORECASE
    )

    results = {}
    for match in pattern.finditer(generated_text):
        label = match.group(1).lower()
        if label in class_names or not class_names:
            try:
                locs = [int(match.group(i)) for i in range(2, 6)]
                score = float(match.group(6)) if match.group(6) else 1.0
                results.setdefault(label, []).append({
                    "bbox": locs,
                    "score": score
                })
            except Exception as e:
                print(f"Skipping match due to parse error: {e}")

    return results
from PIL import Image, ImageDraw, ImageFont

def draw_bboxes_on_image(image_path, detections, output_path="output_annotated.jpg"):
    image = Image.open(image_path).convert("RGB")
    # image = Image.open("static/download.jpg").convert("RGB")
    print(f"Image size: {image.size}")  # (width, height)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    width, height = image.size

    for label, boxes in detections.items():
        for item in boxes:
            x1, y1, x2, y2 = item['bbox']
            # Clamp values to image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{label} ({item['score']:.2f})"
            draw.text((x1, max(0, y1 - 15)), text, fill="yellow", font=font)


    image.save(output_path)
    print(f"Annotated image saved to {output_path}")
    return image

# detections = detect_classes_bboxes_fast("static/download2.jpg", ["dog","cat"])
# print(detections)
# annotated_image = draw_bboxes_on_image("static/download.jpg", detections)
# annotated_image.show()