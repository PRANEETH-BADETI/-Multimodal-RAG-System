from pinecone import Pinecone
import os
import uuid
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "multimodal-rag"

if not PINECONE_API_KEY:
    logger.critical("PINECONE_API_KEY must be set in .env")
    raise EnvironmentError("PINECONE_API_KEY must be set.")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        logger.error(f"Index '{INDEX_NAME}' does not exist.")
        raise ConnectionError(
            f"Index '{INDEX_NAME}' does not exist. "
            f"Please create it in the Pinecone console with 512 dimensions and cosine metric."
        )

    index = pc.Index(INDEX_NAME)

    logger.info(f"Successfully connected to Pinecone index '{INDEX_NAME}'.")
    logger.info(index.describe_index_stats())

except Exception as e:
    logger.critical(f"Failed to initialize Pinecone: {e}")
    raise


def add_item(embedding, document, metadata, item_id=None):
    """
    Upserts a single item (text chunk or image) to the Pinecone index.
    """
    if not item_id:
        item_id = str(uuid.uuid4())

    metadata['content'] = document

    try:
        index.upsert(
            vectors=[
                (item_id, embedding, metadata)
            ]
        )
        logger.info(f"Upserted item {item_id} from {metadata.get('source_file')}")
    except Exception as e:
        logger.error(f"Error upserting to Pinecone: {e}")


def query(query_text: str, k: int = 5):
    """Queries the Pinecone index with a text query."""
    from embedder import embed_text

    # This prefix is standard practice for improving CLIP text-to-image search
    query_for_embedding = f"A photo of: {query_text}"
    logger.info(f"Embedding modified query: {query_for_embedding}")

    query_embedding = embed_text(query_for_embedding)

    if query_embedding is None:
        logger.error("Failed to generate query embedding.")
        return None

    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )
        return results
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return None