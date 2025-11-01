# Multimodal RAG System - AI Intern Assignment

This project is a Retrieval-Augmented Generation (RAG) system capable of processing and querying multiple data formats, including text documents, images, and mixed-content PDFs. It features a FastAPI backend, a Pinecone vector database, and a Streamlit frontend for demonstration.

The core of this project is the **`clip-ViT-B-32`** multimodal model, which embeds both text and images into the same vector space, enabling true **cross-modal search**.

## 🚀 Key Features

* **Multimodal Ingestion:** Process `.txt`, `.pdf`, `.png`, `.jpg`, and `.jpeg` files.
* **True Cross-Modal Search:** Use a text query (e.g., "a picture of a red car") to find relevant images.
* **Advanced PDF Processing:** Extracts and embeds *both* text and images from mixed-content PDFs.
* **Async Backend:** Built with **FastAPI** for high-performance, asynchronous file processing.
* **Vector Storage:** Uses **Pinecone** as a scalable, managed vector database.
* **Interactive Demo:** A **Streamlit** app provides an easy-to-use interface for uploading files and testing queries.

---

## 🛠️ Architecture Overview

The system is composed of two main parts that run simultaneously:

1.  **FastAPI Backend (`main.py`)**:
    * `/upload/`: Handles file uploads, saves them, and (asynchronously) passes their absolute paths to the `processor`.
    * `/query/`: Receives a text query, modifies it for better image retrieval, embeds it, and returns results from Pinecone.
    * `/static_images/`: Serves processed images so they can be displayed in the frontend.

2.  **Streamlit Frontend (`app.py`)**:
    * Provides a simple UI to upload files to the `/upload/` endpoint.
    * Sends text queries to the `/query/` endpoint and displays the results, rendering images if found.

### Core Components:

* **Embedder (`embedder.py`)**: Loads the `clip-ViT-B-32` model to create 512-dimension vector embeddings for both text and images.
* **Processor (`processor.py`)**: The "brains" of the ingestion pipeline. It uses `PyMuPDF` to extract text and images from PDFs, copies files to a static directory, and orchestrates the embedding and database insertion.
* **Vector Store (`vector_store.py`)**: Manages all communication with the Pinecone index, including upserting new vectors and executing search queries.

---

## 🏁 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

* Python 3.8+
* A Pinecone account (free tier)

### 2. Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/PRANEETH-BADETI/-Multimodal-RAG-System.git
    cd -Multimodal-RAG-System
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Pinecone:**
    * Log in to your Pinecone account.
    * Create a new **manual configuration** index.
    * **Index Name:** `multimodal-rag`
    * **Dimensions:** `512`
    * **Metric:** `cosine`
    * Get your API key.

5.  **Create `.env` File:**
    * Create a file named `.env` in the project root.
    * Add your Pinecone API key:
        ```
        PINECONE_API_KEY="your-pinecone-api-key-here"
        ```

### 3. Running the Application

You must run **two** servers simultaneously in two separate terminals.

* **Terminal 1 (Backend - FastAPI):**
    ```bash
    uvicorn main:app --reload
    ```
    *Server will be running at `http://127.0.0.1:8000`*

* **Terminal 2 (Frontend - Streamlit):**
    ```bash
    streamlit run app.py
    ```
    *Streamlit app will open in your browser.*

---

## 🧪 Sample Queries & Results

Use the Streamlit UI to upload the files from the `/sample_data` folder, then run any queries.

---

