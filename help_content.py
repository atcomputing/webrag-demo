HELP_CONTENT = """
# Retrieval-Augmented Generation (RAG) Demo App

## What is RAG?
Retrieval-Augmented Generation (RAG) is an AI technique that enhances language models by providing them with relevant information retrieved from a knowledge base. This allows the model to generate more accurate and contextually appropriate responses.

## What is ChromaDB?
ChromaDB is an open-source vector database that powers the document storage and retrieval in this app. It:
- Stores document chunks as vectors (numerical representations of text)
- Organizes documents into collections for better management
- Enables fast similarity search to find relevant context
- Tracks statistics about your document collections

**Note**: In this demo, ChromaDB runs in-memory, which means all data (collections, vectors, etc.) will be lost when you restart the application. This is ideal for testing and demonstration purposes, but in a production environment, you would typically configure ChromaDB to persist data to disk.

### How ChromaDB is Used Here:
1. **Document Storage**: When you upload files, they're split into chunks and stored in ChromaDB collections
2. **Vector Search**: When you ask a question, ChromaDB finds the most similar document chunks
3. **Collection Management**: You can create multiple collections to organize different sets of documents
4. **Statistics Tracking**: The ChromaDB Stats panel shows your total collections and vectors

## How to Use This App

1. **Explore Demo Data**: 
   - Click "Show Demo Files" to view available demo documents
   - Use "Ingest Demo Data" to load these documents into the system
   - View the "ChromaDB Stats" at the bottom of the sidebar to track document ingestion

2. **Or Upload Your Own Data**:
   - Use the file uploader in the sidebar to add your documents
   - Create a new collection or select an existing one
   - Click "Upload and Index" to process your documents

3. **Configure Settings**:
   - Select LLM and Embedding models from the dropdowns
   - Adjust advanced settings if needed (click to expand)

4. **Query Your Data**:
   - Select a collection to query from the dropdown
   - Type your question in the text input field
   - Choose query options:
     - Use RAG: Enable/disable context-enhanced responses
     - Without RAG: Compare responses with/without context
     - Print RAG context: View the retrieved document snippets

5. **Get Results**: Click "Submit Query" to see the response(s)

## Settings Explained

### Models
- **LLM Model**: The AI model that generates responses
- **Embedding Model**: Creates vector representations for similarity search

### Advanced Settings
- **Number of similar documents**: Controls how many relevant document chunks are retrieved for context. Increasing this gets more context but may dilute relevance. Decreasing focuses on the most relevant chunks but might miss important context.

- **Maximum input size to LLM**: The maximum number of tokens (words/subwords) the model can process at once. Increasing allows more context but uses more memory and may be slower. Decreasing reduces memory usage but might truncate important context.

- **Number of tokens for generation**: Maximum length of the model's response. Increasing allows longer, more detailed responses but takes longer to generate. Decreasing gives shorter, more concise responses and generates faster.

- **Chunk size**: How many characters to include in each document segment during indexing. Larger chunks preserve more context but may retrieve irrelevant information. Smaller chunks are more precise but might break up related information.

- **Chunk overlap**: How many characters should overlap between consecutive chunks. More overlap helps maintain context across chunk boundaries but increases storage size. Less overlap saves space but might miss connections between chunks.

### Collections
- Groups of related documents
- Create new collections or use existing ones
- View collection statistics in ChromaDB Stats

## Tips
- Compare responses with and without RAG to see the impact of context
- Adjust chunk settings for different document types
- Monitor ChromaDB Stats to track your document collections
- Use the tutorial for step-by-step guidance
"""
