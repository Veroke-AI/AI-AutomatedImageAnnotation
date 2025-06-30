import cv2
import albumentations as A
from pathlib import Path
import uuid
from typing import Union

def augment_images(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path] = None,
    augmentations_per_image: int = 3,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    # Define augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=25, p=0.5),
        A.GaussianBlur(p=0.2),
        A.HueSaturationValue(p=0.3)
    ])

    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        output_class_dir = output_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*.png"):
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(augmentations_per_image):
                augmented = transform(image=image)["image"]
                aug_filename = f"{img_path.stem}_aug_{uuid.uuid4().hex[:8]}.png"
                aug_path = output_class_dir / aug_filename
                cv2.imwrite(str(aug_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
