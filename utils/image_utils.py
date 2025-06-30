import cv2
import numpy as np
import os
# from fastapi import UploadFile
# import tempfile

def save_mask_and_bbox_crops(mask_resized, image_cv2, box, base_name, idx, output_dir):
    x1, y1, x2, y2 = map(int, box)

    # Crop mask and bbox region
    mask_crop = mask_resized[y1:y2, x1:x2]
    box_crop = image_cv2[y1:y2, x1:x2]

    # Define filenames
    mask_crop_path = os.path.join(output_dir, f"{base_name}_mask_{idx}.png")
    box_crop_path = os.path.join(output_dir, f"{base_name}_box_{idx}.png")

    # Save crops
    cv2.imwrite(mask_crop_path, mask_crop)
    cv2.imwrite(box_crop_path, box_crop)

    print(f"Saved mask crop: {mask_crop_path}")
    print(f"Saved box crop: {box_crop_path}")

def save_annotated_image(img_file, results, output_path: str):
    # # Create a temporary file to save the uploaded image
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
    #     # Ensure file content is written before accessing
    #     tmp_file.write(img_file.file.read())
    #     tmp_file_path = tmp_file.name

    # Now use cv2.imread with the path to the temporary file
    image = cv2.imread(img_file)
    
    if image is None:
        raise ValueError("Failed to read the image. The file might be corrupted or empty.")

    for det in results:
        x1, y1, x2, y2 = map(int, det["bbox"])
        labels = det.get("label", "object")
        # print(f"Labels: {labels}")
        confidence = det.get("confidence", 0.0)
        label = f"{labels} ({confidence*100:.1f}%)"
        # print(f"Label: {label}")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save image to specified format (e.g., .jpg, .png)
    cv2.imwrite(output_path, image)
def save_annotated_image_click(img_file, results, output_path: str):
    # # Create a temporary file to save the uploaded image
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
    #     # Ensure file content is written before accessing
    #     tmp_file.write(img_file.file.read())
    #     tmp_file_path = tmp_file.name

    # Now use cv2.imread with the path to the temporary file
    image = cv2.imread(img_file)
    
    if image is None:
        raise ValueError("Failed to read the image. The file might be corrupted or empty.")

    for det in results:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save image to specified format (e.g., .jpg, .png)
    cv2.imwrite(output_path, image)


def crop_with_mask(image, mask):
    # Get bounding box
    y_indices, x_indices = np.where(mask)
    if y_indices.size == 0 or x_indices.size == 0:
        return None

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    cropped = image[y_min:y_max, x_min:x_max]
    masked = cv2.bitwise_and(cropped, cropped, mask=mask[y_min:y_max, x_min:x_max].astype(np.uint8))
    return masked

def apply_mask(image, mask, alpha=0.4):
    color = (0, 255, 0)
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = ((1 - alpha) * image[mask] + alpha * np.array(color)).astype(np.uint8)
    return overlay

def draw_label(image, mask, label: str):
    overlay = apply_mask(image.copy(), mask)
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return overlay

import os
import cv2

def save_mask_and_bbox_crops(mask_resized, image_cv2, box, base_name, idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    x1, y1, x2, y2 = map(int, box)

    # Full mask
    full_mask_path = os.path.join(output_dir, f"{base_name}_fullmask_{idx}.png")
    cv2.imwrite(full_mask_path, mask_resized)

    # Cropped mask and bbox region
    mask_crop = mask_resized[y1:y2, x1:x2]
    box_crop = image_cv2[y1:y2, x1:x2]

    mask_crop_path = os.path.join(output_dir, f"{base_name}_maskcrop_{idx}.png")
    box_crop_path = os.path.join(output_dir, f"{base_name}_boxcrop_{idx}.png")

    cv2.imwrite(mask_crop_path, mask_crop)
    cv2.imwrite(box_crop_path, box_crop)

    print(f"[Saved] Full mask: {full_mask_path}")
    print(f"[Saved] Mask crop: {mask_crop_path}")
    print(f"[Saved] Box crop: {box_crop_path}")




