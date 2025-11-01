from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import logging
from processor import process_file
from vector_store import query as query_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal RAG API",
    description="Upload documents (txt, pdf, img) and query them.",
    version="1.0.0"
)

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
STATIC_DIR = os.path.join(UPLOAD_DIR, "processed_images")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)  # Ensure this exists

# Mount the 'uploads' directory to serve images
app.mount("/static_images", StaticFiles(directory=STATIC_DIR), name="static_images")


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class UploadResponse(BaseModel):
    message: str
    filename: str


class QueryResult(BaseModel):
    source: str
    page: int | None = None
    content_type: str
    content: str
    image_url: str | None = None
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    results: list[QueryResult]


# --- Exception Handlers ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


# --- API Endpoints ---
@app.post("/upload/", response_model=UploadResponse)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a file (txt, pdf, png, jpg, jpeg).
    Processing is done in the background.
    """
    try:
        # Save the uploaded file to the UPLOAD_DIR
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['txt', 'png', 'jpg', 'jpeg', 'pdf']:
            logger.warning(f"Unsupported file type uploaded: {file_ext}")
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        absolute_file_path = os.path.abspath(file_path)
        background_tasks.add_task(process_file, absolute_file_path, file_ext)

        logger.info(f"File {file.filename} uploaded, starting background processing.")
        return {"message": "File uploaded successfully. Processing in background.", "filename": file.filename}

    except Exception as e:
        logger.error(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during upload: {str(e)}")


@app.post("/query/", response_model=QueryResponse)
async def query_system(request: QueryRequest, http_request: Request):
    """
    Query the system with a text prompt.
    Returns the most relevant text chunks and/or image paths.
    """
    try:
        logger.info(f"Received query: {request.query}")
        results = query_vector_store(request.query, k=request.top_k)

        if results is None:
            logger.error("Query to vector store returned None.")
            raise HTTPException(status_code=500, detail="Error querying vector store.")

        formatted_results = []
        base_url = str(http_request.base_url)

        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            score = match.get('score', 0)

            doc_path = metadata.get('content', '')

            content_display = ""
            image_url = None

            if metadata.get('content_type') == 'text':
                content_display = doc_path
            elif metadata.get('content_type') == 'image':
                content_display = f"Retrieved image from {metadata.get('source_file')}"
                image_filename = os.path.basename(doc_path)
                image_url = f"{base_url}static_images/{image_filename}"

            formatted_results.append(
                QueryResult(
                    source=metadata.get('source_file', 'unknown'),
                    page=metadata.get('page_num'),
                    content_type=metadata.get('content_type', 'unknown'),
                    content=content_display,
                    image_url=image_url,
                    relevance_score=score
                )
            )

        return QueryResponse(query=request.query, results=formatted_results)

    except Exception as e:
        logger.error(f"Error during query processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {str(e)}")


# --- Run Application ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)