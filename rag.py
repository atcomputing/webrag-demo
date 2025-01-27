import streamlit as st
import requests
import os
import base64
from typing import List
from help_content import HELP_CONTENT
from tutorial_content import TUTORIAL_CONTENT

import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Custom CSS for UI elements
st.markdown(
    """
<style>
.download-button {
    display: inline-block;
    padding: 8px 16px;
    background-color: #4CAF50;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    margin: 5px 0;
}
.download-button:hover {
    background-color: #45a049;
}
.stButton > button {
    width: 100%;
    margin: 0;
}
.response-box {
    border: 2px solid #4CAF50;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}
.context-box {
    border: 2px solid #2196F3;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}
</style>
""",
    unsafe_allow_html=True
)

# Fetch available Ollama models
def get_ollama_models():
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            return [model["name"] for model in models]
        else:
            st.error(f"Failed to fetch models: {response.status_code}")
            return []
    except requests.RequestException as e:
        st.error(f"Error fetching models: {e}")
        return []

# Initialize ChromaDB client
def get_chroma_client():
    try:
        return chromadb.Client()
    except ValueError:
        # Fallback with tenant if needed
        return chromadb.Client(chromadb.config.Settings(tenant="default_tenant"))

# Retrieve Chroma collection names
def get_collections() -> List[str]:
    chroma_client = get_chroma_client()
    return chroma_client.list_collections()  # v0.6.0 returns just names (strings)

# Create a new Chroma collection
def create_collection(name: str):
    chroma_client = get_chroma_client()
    chroma_client.create_collection(name)

# Get ChromaDB statistics
def get_chromadb_stats():
    chroma_client = get_chroma_client()
    collections = chroma_client.list_collections()

    stats = {
        "num_collections": len(collections),
        "total_vectors": 0,
    }
    for coll_name in collections:
        collection_obj = chroma_client.get_collection(coll_name)
        stats["total_vectors"] += collection_obj.count()

    return stats

# Upload and index files into a specified collection
def upload_files(files, collection_name: str, chunk_size: int, chunk_overlap: int):
    try:
        with st.spinner(f'Uploading and indexing files to collection "{collection_name}"... This may take a few moments.'):
            # Get or create collection
            chroma_client = chromadb.Client()
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
            all_documents = []
    
            # Process each file
            for file in files:
                try:
                    # Save uploaded file temporarily
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())
                        
                    # Read the file
                    documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
                    all_documents.extend(documents)
                    
                    # Remove temporary file
                    os.remove(file.name)
                    
                    st.success(f"Processed file: {file.name}")
                except Exception as e:
                    st.error(f"Error processing file {file.name}: {str(e)}")
                    continue

            if not all_documents:
                st.error("No documents were successfully processed.")
                return None

            # Create index with all documents
            index = VectorStoreIndex.from_documents(
                all_documents,
                storage_context=storage_context,
                embed_model=Settings.embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            st.success(f"All files uploaded and indexed in collection: {collection_name}")
            return index

    except Exception as e:
        st.error(f"Error during upload and indexing: {str(e)}")
        return None

# Show demo files with download buttons
def show_demo_files():
    st.header("Available Demo Files")
    demo_dir = "demo_docs"
    
    for filename in os.listdir(demo_dir):
        filepath = os.path.join(demo_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{filename}**")
        with col2:
            b64 = base64.b64encode(content.encode()).decode()
            href = f'data:text/plain;base64,{b64}'
            st.download_button(
                label="Download",
                data=content,
                file_name=filename,
                mime="text/plain",
                key=filename
            )

# Ingest demo documents into Chroma
def ingest_demo_data():
    with st.spinner('Ingesting demo data... This may take a few moments.'):
        try:
            chroma_client = chromadb.Client()
        except ValueError as e:
            if "Could not connect to tenant default_tenant" in str(e):
                chroma_client = chromadb.Client(tenant="default_tenant")
            else:
                raise e

        demo_collection = chroma_client.get_or_create_collection("demo")
        vector_store = ChromaVectorStore(chroma_collection=demo_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = SimpleDirectoryReader("demo_docs").load_data()

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )

        st.session_state.chromadb_stats = get_chromadb_stats()

        return index

# Main function to set up the Streamlit app
def main():
    st.title("Retrieval Augmented Generation")

    # Add logo to the sidebar
    with st.sidebar:
        st.image("AT-LOGO-WEB-BL-Z-v1.png", use_container_width=True)

    # Controls for demo data and help/tutorial
    with st.sidebar:

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Ingest Demo Data", use_container_width=True):
                st.session_state.demo_index = ingest_demo_data()
                st.success("Demo data ingested successfully!")
                st.session_state.chromadb_stats = get_chromadb_stats()
        
        with col2:
            if st.button("Show Demo Files", use_container_width=True):
                st.session_state.show_demo_files = True

        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("Help", use_container_width=True):
                st.session_state.show_help = not st.session_state.get('show_help', False)
        with col4:
            if st.button("Tutorial", use_container_width=True):
                st.session_state.show_tutorial = not st.session_state.get('show_tutorial', False)

    if st.session_state.get('show_help', False):
        with st.expander("Help", expanded=True):
            st.markdown(HELP_CONTENT)
            if st.button("Close Help"):
                st.session_state.show_help = False

    if st.session_state.get('show_tutorial', False):
        with st.expander("Tutorial", expanded=True):
            st.markdown(TUTORIAL_CONTENT)
            if st.button("Close Tutorial"):
                st.session_state.show_tutorial = False

    if st.session_state.get('show_demo_files', False):
        show_demo_files()
        if st.button("Close Demo Files"):
            st.session_state.show_demo_files = False
        st.divider()

    # Fetch available models
    available_models = get_ollama_models()

    # Sidebar settings for LLM and embedding models
    st.sidebar.header("Settings")
    default_model_index = 0
    default_embed_index = 0
    if "llama3:latest" in available_models:
        default_model_index = available_models.index("llama3:latest")
    if "mxbai-embed-large:latest" in available_models:
        default_embed_index = available_models.index("mxbai-embed-large:latest")

    model = st.sidebar.selectbox("LLM Model", options=available_models, index=default_model_index)
    embed_model = st.sidebar.selectbox("Embedding Model", options=available_models, index=default_embed_index)

    # Advanced settings configuration
    with st.sidebar.expander("Advanced Settings"):
        similarity_top_k = st.number_input("Number of similar documents", value=4, min_value=1)
        context_window = st.number_input("Maximum input size to LLM", value=4096, min_value=1)
        num_output = st.number_input("Number of tokens for generation", value=256, min_value=1)
        chunk_size = st.number_input("Chunk size for document parsing", value=1024, min_value=128)
        chunk_overlap = st.number_input("Chunk overlap for document parsing", value=128, min_value=0)

    # File upload and collection management
    st.sidebar.header("Document Upload")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True)

    collections = get_collections()
    collection_name = st.sidebar.selectbox("Select Collection", options=collections + ["New Collection"])

    if collection_name == "New Collection":
        new_collection_name = st.sidebar.text_input("Enter new collection name")
        if st.sidebar.button("Create Collection"):
            if new_collection_name:
                create_collection(new_collection_name)
                st.sidebar.success(f"Collection '{new_collection_name}' created.")
                collection_name = new_collection_name
                st.session_state.chromadb_stats = get_chromadb_stats()
            else:
                st.sidebar.error("Please enter a name for the new collection.")

    if uploaded_files and collection_name != "New Collection":
        if st.sidebar.button("Upload and Index"):
            st.session_state.index = upload_files(uploaded_files, collection_name, chunk_size, chunk_overlap)
            st.session_state.chromadb_stats = get_chromadb_stats()

    # Configure global settings for embeddings and LLM
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model,
        base_url=ollama_base_url,
        ollama_additional_kwargs={"mirostat": 0},
    )
    Settings.llm = Ollama(model=model, base_url=ollama_base_url, request_timeout=360.0)
    Settings.context_window = context_window
    Settings.num_output = num_output
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    # Initialize index
    if "index" not in st.session_state:
        st.session_state.index = None

    # Collection selection for querying
    all_collections_for_query = get_collections()
    if "demo_index" in st.session_state:
        all_collections_for_query.append("demo")

    query_collection = st.selectbox("Select Collection for Querying", options=all_collections_for_query)

    # Show tutorial if no collection is selected
    if not query_collection:
        st.markdown(TUTORIAL_CONTENT)
        return

    # Get the index for the selected collection
    if query_collection == "demo":
        if "demo_index" in st.session_state:
            index = st.session_state.demo_index
        else:
            st.warning("Please ingest the demo data first.")
            return
    else:
        chroma_client = chromadb.Client()
        chroma_collection = chroma_client.get_collection(query_collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # Query input field
    query = st.text_input("Enter your query:")

    # Options for query processing
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rag = st.checkbox("Use RAG", value=True)
    with col2:
        compare = st.checkbox("Without RAG", value=True)
    with col3:
        print_context = st.checkbox("Print RAG context", value=True)

    # Handle query submission
    if st.button("Submit Query"):
        if query:
            with st.spinner('Generating response...'):

                query_engine = index.as_query_engine(
                    similarity_top_k=similarity_top_k,
                    use_async=True,
                    llm=Settings.llm,
                    streaming=True
                )

                if not use_rag:
                    st.subheader("Response without RAG:")
                    response_placeholder = st.empty()
                    full_response = ""
                    for response_chunk in Settings.llm.stream_complete(query):
                        full_response += response_chunk.delta
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}</div>',
                            unsafe_allow_html=True
                        )

                elif compare:
                    col1, col2 = st.columns(2)

                    # Left column: no RAG
                    with col1:
                        st.subheader("Response without RAG:")
                        response_placeholder_norag = st.empty()
                        full_response_norag = ""
                        for response_chunk in Settings.llm.stream_complete(query):
                            full_response_norag += response_chunk.delta
                            response_placeholder_norag.markdown(
                                f'<div class="response-box">{full_response_norag}</div>',
                                unsafe_allow_html=True
                            )

                    # Right column: with RAG
                    with col2:
                        st.subheader("Response with RAG:")
                        response_placeholder_rag = st.empty()
                        full_response_rag = ""
                        response_rag = query_engine.query(query)
                        for text in response_rag.response_gen:
                            full_response_rag += text
                            response_placeholder_rag.markdown(
                                f'<div class="response-box">{full_response_rag}</div>',
                                unsafe_allow_html=True
                            )

                    # Print RAG context if requested
                    if print_context and response_rag.source_nodes:
                        st.subheader("RAG Context:")
                        context_placeholder = st.empty()
                        full_context = ""
                        for node_entry in response_rag.source_nodes:
                            full_context += f"File: {node_entry.node.metadata.get('file_path', 'N/A')}\n"
                            full_context += f"Text: {node_entry.node.text}\n\n"
                        context_placeholder.markdown(
                            f'<div class="context-box">{full_context}</div>',
                            unsafe_allow_html=True
                        )

                else:
                    st.subheader("Response with RAG:")
                    response_placeholder = st.empty()
                    full_response = ""
                    response = query_engine.query(query)
                    for text in response.response_gen:
                        full_response += text
                        response_placeholder.markdown(
                            f'<div class="response-box">{full_response}</div>',
                            unsafe_allow_html=True
                        )

                    if print_context and response.source_nodes:
                        st.subheader("RAG Context:")
                        context_placeholder = st.empty()
                        full_context = ""
                        for node_entry in response.source_nodes:
                            full_context += f"File: {node_entry.node.metadata.get('file_path', 'N/A')}\n"
                            full_context += f"Text: {node_entry.node.text}\n\n"
                        context_placeholder.markdown(
                            f'<div class="context-box">{full_context}</div>',
                            unsafe_allow_html=True
                        )
        else:
            st.warning("Please enter a query.")
            
    # Display ChromaDB statistics
    if 'chromadb_stats' not in st.session_state:
        st.session_state.chromadb_stats = get_chromadb_stats()

    stats = st.session_state.chromadb_stats
    st.sidebar.header("ChromaDB Stats")
    st.sidebar.write(f"Number of Collections: {stats['num_collections']}")
    st.sidebar.write(f"Total Vectors: {stats['total_vectors']}")

    # Copyright notice
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        © 2024 Dennis Kruyt<br>
        AT Computing
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
