import cv2
import numpy as np
import random
from utils.image_utils import save_mask_and_bbox_crops
from segment_anything import sam_model_registry, SamPredictor



def annotate_with_sam(image_path, detections, output_path, predictor, save_dir="outputs/crops"):
    print(f"Annotating image with SAM: {image_path}")
    print(f"Detections: {detections}")
    image = cv2.imread(image_path)
    original = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # base_name = "Sam"

    predictor.set_image(image_rgb)

    all_masks = []
    all_scores = []

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        label_name = det.get("label", f"object{idx}")
        print(f"Label name: {label_name}")
        # labels = det.get("label", "object")
        confidence = det.get("confidence", 0.0)
        label = f"{label_name} ({confidence * 100:.1f}%)"
        # Run SAM on the bounding box
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )

        mask = masks[0]
        all_masks.append(mask)
        all_scores.append(scores[0])

        # Generate a random color for the mask
        color = [random.randint(0, 255) for _ in range(3)]
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[mask] = color

        image = cv2.addWeighted(image, 1.0, color_mask, 0.4, 0)

        # Draw the bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save mask and crops if requested
        if save_dir:
            save_mask_and_bbox_crops(
                mask_resized=(mask.astype(np.uint8) * 255),
                image_cv2=original,
                box=(x1, y1, x2, y2),
                base_name=label_name,
                idx=idx,
                output_dir=save_dir
            )

    # Save annotated image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

    return all_masks, all_scores

def segment_with_sam_clicks(image_path, click_points, sam_checkpoint="models/sam_vit_b_01ec64.pth", model_type="vit_b", device="cuda"):
    """
    Segments objects using SAM based on click point(s) without any text prompt.

    Parameters:
        image_path (str): Path to the input image.
        click_points (List[Dict]): List of points with 'x' and 'y' coordinates.
        sam_checkpoint (str): Path to the SAM checkpoint.
        model_type (str): Model type for SAM. Options: "vit_h", "vit_l", "vit_b".
        device (str): Device to run SAM on ("cuda" or "cpu").

    Returns:
        List of dicts with keys: "bbox" and "mask" (numpy array)
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Convert click points to required format
    input_points = np.array([[pt["x"], pt["y"]] for pt in click_points])
    input_labels = np.array([1] * len(input_points))  # 1 means "foreground"

    # Predict mask(s)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True  # You can change to False for a single best mask
    )

    results = []
    for i, mask in enumerate(masks):
        # Get bounding box from mask
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue  # skip empty masks
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        results.append({
            "bbox": [x1, y1, x2, y2],
            "mask": mask.astype(np.uint8)
        })

    return results


def segment_with_sam_clicks(
    image_path,
    click_points,
    sam_checkpoint="models/sam_vit_b_01ec64.pth",
    model_type="vit_b",
    device="cuda"
):
    """
    Segments objects using SAM based on click point(s) without any text prompt.

    Parameters:
        image_path (str): Path to the input image.
        click_points (List[Dict]): List of points with 'x' and 'y' coordinates.
        sam_checkpoint (str): Path to the SAM checkpoint.
        model_type (str): Model type for SAM. Options: "vit_h", "vit_l", "vit_b".
        device (str): Device to run SAM on ("cuda" or "cpu").

    Returns:
        List of dicts with keys: "bbox" and "mask" (numpy array)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    results = []

    for point in click_points:
        x, y = point["x"], point["y"]
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 for foreground

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False  # only best mask
        )

        mask = masks[0]
        # Get bounding box from mask
        y_indices, x_indices = np.where(mask)
        if y_indices.size == 0 or x_indices.size == 0:
            continue  # skip empty masks

        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        bbox = [x_min, y_min, x_max, y_max]

        results.append({
            "bbox": bbox
        })

    return {"bbox": bbox}

