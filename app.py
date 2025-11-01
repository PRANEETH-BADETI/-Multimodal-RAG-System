import streamlit as st
import requests
import os

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload/"
QUERY_ENDPOINT = f"{API_BASE_URL}/query/"

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Multimodal RAG Demo",
    layout="wide"
)
st.title("Multimodal RAG System Demo 🚀")

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("1. Upload Your Files")
    uploaded_files = st.file_uploader(
        "Upload TXT, PDF, or Image files",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Process Uploaded Files"):
        if uploaded_files:
            with st.spinner("Processing files... This may take a moment."):
                for file in uploaded_files:
                    # Prepare the file for the API
                    files = {"file": (file.name, file.getvalue(), file.type)}

                    try:
                        response = requests.post(UPLOAD_ENDPOINT, files=files)
                        if response.status_code == 200:
                            st.success(f"Successfully processed: {file.name}")
                        else:
                            st.error(f"Error processing {file.name}: {response.json().get('detail')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"API connection error for {file.name}: {e}")
            st.success("All files processed!")
        else:
            st.warning("Please upload files first.")

# --- Main Area for Querying ---
st.header("2. Query Your Documents")
query_text = st.text_input("Enter your query (e.g., 'a picture of a blue car' or 'what is this project about?'):")

if st.button("Get Answer"):
    if query_text:
        with st.spinner("Searching for answers..."):
            try:
                # Send query to the API
                response = requests.post(QUERY_ENDPOINT, json={"query": query_text, "top_k": 5})

                if response.status_code == 200:
                    results = response.json()
                    st.success("Query successful!")

                    # Display the results
                    st.subheader("Search Results")
                    if not results.get("results"):
                        st.warning("No relevant documents found for your query.")

                    for i, res in enumerate(results.get("results", [])):
                        st.markdown(f"---")
                        st.markdown(f"**Result {i + 1} (Score: {res['relevance_score']:.4f})**")

                        # Display text content
                        st.info(f"**Content:** {res['content']}")

                        # Display metadata
                        with st.expander("Show Details"):
                            st.json({
                                "source": res['source'],
                                "page": res.get('page'),
                                "content_type": res['content_type']
                            })

                        # Display image if URL is present
                        if res.get("image_url"):
                            # --- THIS IS THE FIXED LINE ---
                            st.image(res["image_url"], caption=f"Retrieved Image (from {res['source']})",
                                     use_container_width=True)

                else:
                    st.error(f"Error during query: {response.json().get('detail')}")
            except requests.exceptions.RequestException as e:
                st.error(f"API connection error: {e}")
    else:
        st.warning("Please enter a query.")