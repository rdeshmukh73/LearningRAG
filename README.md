# 🧠 RAG-Learning-Hub: Learn Retrieval-Augmented Generation by Doing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-%F0%9F%A6%8A-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-%F0%9F%90%AD%20Indexing-orange)
![MistralAI](https://img.shields.io/badge/Mistral-AI-orange?logo=ai)
[![Ollama](https://img.shields.io/badge/Ollama-%F0%9F%96%A5%EF%B8%8F%20Local%20LLMs-darkgreen)](https://ollama.com)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%9D%A4-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌟 Introduction

Welcome to **RAG-Learning-Hub**, a hands-on summer project designed to teach you **Retrieval-Augmented Generation (RAG)** through practical, working examples.

We'll guide you step-by-step from fundamental concepts to building production-ready chat interfaces powered by **LangChain**, **Mistral AI**, **FAISS**, and more.

---

## 🧭 Learning Journey

Here’s what you’ll explore in this project:

### 🔹 1. RAG Fundamentals
- What is RAG and why it matters?
- Understanding context-aware AI with retrieval

### 📄 2. Document Handling
- 🗂️ Loading documents using LangChain
- ✂️ Chunking strategies for better embeddings

### 📦 3. Vector Stores
- 📥 Adding chunks to Vector Databases (FAISS, Qdrant, Weaviate)
- 🔍 Using them as Retrievers for your LLM

### 💬 4. Chat Interfaces
- Integrating retrievers with LLMs via LangChain
- Building interactive RAG pipelines

### 🧪 5. Performance & Evaluation
- Precision/recall on retrieved chunks
- How to benchmark and debug RAG flows

### 🔌 6. API Layer
- Build a lightweight API on top of RAG using **FastAPI** or **Flask** or **Relevant** technologies.  We will default to **FastAPI** for most of this project

### 🎛️ 7. User Interfaces
- Streamlit or Gradio frontend for:
  - Document upload
  - Chat experience
  - Source document tracing

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Core language |
| 🔗 LangChain | RAG orchestration |
| 🧠 Mistral AI (via Ollama) | LLM backend |
| 📚 FAISS / Qdrant / Weaviate | Vector search |
| 🌐 Flask / FastAPI | APIs |
| 📊 Streamlit / Gradio | Frontend UI |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone 


# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
