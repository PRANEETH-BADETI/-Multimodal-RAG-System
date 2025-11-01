import fitz  # PyMuPDF
import os
import time
import logging
import shutil
from embedder import embed_text, embed_image
from vector_store import add_item

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
IMG_SAVE_DIR = os.path.join(UPLOAD_DIR, "processed_images")

os.makedirs(IMG_SAVE_DIR, exist_ok=True)


# --- Text Processing ---
def process_text_file(file_path: str):  # file_path is now ABSOLUTE
    """Processes a plain text file, chunks it, and adds to vector store."""
    logger.info(f"Processing text file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = []
        for paragraph in text.split('\n\n'):
            if len(paragraph) > 1000:
                for i in range(0, len(paragraph), 1000):
                    chunks.append(paragraph[i:i + 1000])
            elif paragraph.strip():
                chunks.append(paragraph)

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = embed_text(chunk)
                if embedding is not None:
                    metadata = {
                        "source_file": file_path,
                        "file_type": "txt",
                        "chunk_num": i,
                        "content_type": "text",
                        "upload_timestamp": int(time.time())
                    }
                    add_item(embedding.tolist(), chunk, metadata)

    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {e}")


# --- Image Processing ---
def process_image_file(file_path: str):
    """
    Processes a single image file, copies it to the static directory,
    and adds to vector store.
    """
    logger.info(f"Processing image file: {file_path}")
    try:
        new_file_name = os.path.basename(file_path)
        static_path = os.path.join(IMG_SAVE_DIR, new_file_name)

        logger.info(f"Copying file from {file_path} to {static_path}")
        shutil.copy(file_path, static_path)

        logger.info(f"Embedding image from: {static_path}")
        embedding = embed_image(static_path)

        if embedding is not None:
            metadata = {
                "source_file": file_path,
                "file_type": os.path.splitext(file_path)[1],
                "content_type": "image",
                "upload_timestamp": int(time.time())
            }
            add_item(embedding.tolist(), static_path, metadata)
        else:
            logger.warning(f"Failed to get embedding for image {static_path}. Skipping.")

    except Exception as e:
        logger.error(f"Error processing image file {file_path}: {e}", exc_info=True)


# --- PDF Processing ---
def process_pdf_file(file_path: str):  # file_path is now ABSOLUTE
    """Processes a PDF, extracting and embedding both text and images."""
    logger.info(f"Processing PDF file: {file_path}")
    try:
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # 1. Extract Text
            text = page.get_text()
            if text.strip():
                chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        embedding = embed_text(chunk)
                        if embedding is not None:
                            metadata = {
                                "source_file": file_path,
                                "file_type": "pdf",
                                "page_num": page_num + 1,
                                "chunk_num": i,
                                "content_type": "text",
                                "upload_timestamp": int(time.time())
                            }
                            add_item(embedding.tolist(), chunk, metadata)

            # 2. Extract Images
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                img_filename = f"{os.path.basename(file_path)}_p{page_num}_{img_index}.{image_ext}"
                img_save_path = os.path.join(IMG_SAVE_DIR, img_filename)

                with open(img_save_path, "wb") as img_file:
                    img_file.write(image_bytes)

                logger.info(f"Embedding PDF image from: {img_save_path}")
                embedding = embed_image(img_save_path)  # Pass absolute path

                if embedding is not None:
                    metadata = {
                        "source_file": file_path,
                        "file_type": "pdf",
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "content_type": "image",
                        "upload_timestamp": int(time.time())
                    }
                    add_item(embedding.tolist(), img_save_path, metadata)
                else:
                    logger.warning(f"Failed to get embedding for PDF image {img_save_path}. Skipping.")

    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
    finally:
        if 'doc' in locals():
            doc.close()


# --- Router Function ---
def process_file(file_path: str, file_type: str):  # file_path is now ABSOLUTE
    """Router function to call the correct processor based on file type."""
    if file_type == 'txt':
        process_text_file(file_path)
    elif file_type in ['png', 'jpg', 'jpeg']:
        process_image_file(file_path)
    elif file_type == 'pdf':
        process_pdf_file(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_type} for file: {file_path}")