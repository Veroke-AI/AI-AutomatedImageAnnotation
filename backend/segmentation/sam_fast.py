import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import random
from utils.image_utils import save_mask_and_bbox_crops

def annotate_with_efficient_sam(image_path, detections, output_path, model, device='cpu'):
    # Load original image
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    # Resize to model's expected input size
    target_size = model.image_encoder.img_size  # Should be 1024
    resized_image = image_pil.resize((target_size, target_size))
    image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0).to(device)

    print("Original size:", (orig_w, orig_h), "| Model input size:", target_size)

    # Get image embedding
    # image_embedding = model.image_encoder(image_tensor)

    # Load OpenCV version of original image for drawing
    image_cv2 = cv2.imread(image_path)
    image_copy = image_cv2.copy()

    # Scaling factors for converting detection box to resized image coordinates
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    for idx,det in enumerate(detections):
        label = det['label']
        score = det['confidence']
        box = det['bbox']
        x1, y1, x2, y2 = map(int, box)

        # Scale box to resized image coordinates
        cx = ((x1 + x2) / 2) * scale_x
        cy = ((y1 + y2) / 2) * scale_y

        # Prepare prompt
        point_coords = torch.tensor([[[[cx, cy]]]], device=device)
        point_labels = torch.tensor([[[1]]], device=device)

        # Run mask decoder
        predicted_masks, predicted_iou = model(
            image_tensor,
            point_coords,
            point_labels,
        )
        mask = predicted_masks[0, 0].detach().cpu().numpy()

        # If shape is (3, H, W), reduce it
        if mask.ndim == 3 and mask.shape[0] == 3:
            mask = mask[0]  # or np.max(mask, axis=0) if combining all channels

        mask = (mask > 0).astype(np.uint8) * 255

        # Resize mask back to original size
        print("Mask shape before resize:", mask.shape, "| Target size:", orig_w, orig_h)

        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        print("Mask shape after resize:", mask_resized.shape)
        # Apply mask to original image
        # colored_mask = np.zeros_like(image_cv2)
        # colored_mask[:, :, 1] = mask_resized  # Green channel
        random_color = [random.randint(0, 255) for _ in range(3)]
        save_mask_and_bbox_crops(mask_resized, image_copy, box, "efficient_Sam", idx, "outputs/crops")

        # Apply mask to original image
        colored_mask = np.zeros_like(image_cv2)
        colored_mask[:, :, 0] = mask_resized * random_color[0]  # Red channel
        colored_mask[:, :, 1] = mask_resized * random_color[1]  # Green channel
        colored_mask[:, :, 2] = mask_resized * random_color[2]  # Blue channel

        image_cv2 = cv2.addWeighted(image_cv2, 1.0, colored_mask, 0.2, 0)
        print("Colored mask shape:", colored_mask.shape)
        # Draw box and label with confidence
        cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{label} ({score:.2f})"
        cv2.putText(image_cv2, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print("Text:", text)

    # Save final annotated image
    cv2.imwrite(output_path, image_cv2)
    print(f"Annotated image saved to {output_path}")
    return predicted_masks, predicted_iou
