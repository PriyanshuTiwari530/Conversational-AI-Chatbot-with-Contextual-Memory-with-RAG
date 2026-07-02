# Conversational AI Chatbot Contextual Memory With RAG.

A conversational AI chatbot powered by Retrieval-Augmented Generation (RAG) using 
LangGraph, FAISS, and Cohere LLM. Supports multi-turn conversations with contextual 
memory via MemorySaver. Built with HuggingFace embeddings and available as both a 
Streamlit and Flask app.

## Overview

This chatbot retrieves relevant context from documents using FAISS vector search and 
generates accurate, grounded responses using Cohere's `command-r7b-12-2024` model — 
reducing hallucinations compared to standard LLM usage. LangGraph manages the 
conversation flow and memory across multiple turns, making it feel like a real 
back-and-forth conversation rather than isolated Q&A.

## Features
- **RAG Pipeline**: Retrieves relevant document chunks before generating responses
- **Contextual Memory**: Uses LangGraph's `MemorySaver` to maintain context across turns
- **Fast Vector Search**: FAISS index for efficient semantic similarity search
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` for document and query embeddings
- **Interfaces**: Available as a Streamlit app web app

## Tech Stack
- Python
- LangGraph + LangChain
- FAISS (vector store)
- HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- Cohere LLM (`command-r7b-12-2024`)
- Streamlit 

## How It Works
1. Documents are chunked and converted into embeddings
2. Embeddings are stored in a FAISS vector index
3. On each user query, the most relevant chunks are retrieved
4. Retrieved context + conversation history is passed to Cohere's LLM
5. LangGraph manages the conversation flow and memory across turns

## Installation
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

Add your API key in a `.env` file:
