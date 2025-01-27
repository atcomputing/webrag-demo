# WebRAG Demo

A Streamlit-based web application demonstrating Retrieval Augmented Generation (RAG) using local LLMs via Ollama.

## Features

- Upload and process PDF, TXT, and DOCX documents
- Organize documents in collections using ChromaDB
- Query documents using natural language
- Compare responses with and without RAG
- View source context for responses
- Built-in demo documents and tutorial
- Configurable LLM and embedding models
- Advanced settings for fine-tuning

## Prerequisites

- Python 3.11+
- Docker (optional)
- [Ollama](https://ollama.ai/) running locally or accessible via network

## Quick Start

### Using Docker

1. Start Ollama on your machine
2. Pull and run the container:
```bash
docker-compose up
```

### Manual Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ollama and pull required models:
```bash
ollama pull llama2
ollama pull nomic-embed-text
```

3. Run the application:
```bash
streamlit run rag.py
```

## Usage

1. Access the web interface at http://localhost:8501
2. Click "Ingest Demo Data" to load sample documents
3. Or upload your own documents using the sidebar
4. Select a collection and enter your query
5. Experiment with different models and settings

## Configuration

- `OLLAMA_BASE_URL`: Set this environment variable to connect to Ollama running on a different host
- Advanced settings available in the UI for:
  - Chunk size and overlap
  - Context window size
  - Number of similar documents
  - Response length

## Demo Documents

Includes sample documents covering:
- Solar system and space exploration
- Science fiction (Star Trek)
- Children's stories
- Calendar data
- Technical documentation

## Development

The project uses:
- Streamlit for the web interface
- LlamaIndex for document processing
- ChromaDB for vector storage
- Ollama for local LLM integration

## License

© 2024 Dennis Kruyt, AT Computing
