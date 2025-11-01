from sentence_transformers import SentenceTransformer
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the multimodal model
model = SentenceTransformer('clip-ViT-B-32')


def embed_text(text: str):
    """Generates an embedding for a text chunk."""
    try:
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error embedding text: {e}")
        return None

def embed_image(image_path: str):
    """Generates an embedding for an image."""
    try:
        # Open the image using PIL
        img = Image.open(image_path)
        return model.encode(img)
    except FileNotFoundError:
        logger.error(f"Image file not found at: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error embedding image at {image_path}: {e}")
        return None