from transformers import CLIPProcessor, CLIPModel
import torch
import io
from PIL import Image
import logging


# Global variables for model and processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_image_embedding(image_bytes, model, processor):
    """Extract embedding from image bytes using CLIP"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get image embeddings
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
            
        return image_embeddings.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def extract_text_embedding(text, model, processor):
    print("text: ",text)
    """Extract embedding from text using CLIP"""
    try:
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get text embeddings
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings
            text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
            
        return text_embeddings.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return None
    

    