# 🧬 BioLaySumm RAG Demo

A biomedical article summarization system based on RAG (Retrieval-Augmented Generation) technology. This system generates scientific summaries with three different readability levels according to user needs.

## 📋 Project Overview

**BioLaySumm** is an intelligent summarization system with the following features:

- ✅ **Multi-Query RAG Strategy**: Extract multiple queries from abstracts for multi-angle retrieval
- ✅ **Two-Stage Retrieval**: Dense Retrieval (FAISS) + Cross-Encoder Reranking
- ✅ **Dynamic Chunking**: Automatically adjust chunk size based on article length
- ✅ **Three Readability Styles**: Formal (academic), Plain (accessible), High Readability (simple)
- ✅ **Fully Transparent**: Visualize the RAG retrieval process

## 🚀 Quick Start

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, CPU mode also supported but slower)
- At least 8GB VRAM (for loading Qwen2.5-3B model)

### Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd BioLaySumm
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Note**: First installation may take a few minutes as it needs to download large packages (e.g., PyTorch).

3. **NLTK data** (automatically downloaded, no manual action needed)

The code will automatically download necessary NLTK data (punkt, punkt_tab, stopwords) on first run.

### Running the Demo (Backend + Frontend Version)

This version includes the full interface with RAG process visualization, multi-style comparison, and detailed chunk analysis.

1. **Install dependencies:**

```bash
pip install -r backend/requirements.txt
```

2. **Start the backend server:**

```bash
cd backend
python api.py
```

The backend will:
- Load all models (takes 5-10 minutes on first run)
- Start FastAPI server on port 8000
- Serve the frontend automatically

3. **Access the demo:**

Open your browser and navigate to:
```
http://localhost:8000
```

You will see the full interface with:
- ✅ Input section (title, article, style selector)
- ✅ Generated summaries with three tabs (Formal/Plain/Simple)
- ✅ RAG process visualization
- ✅ Query overlap analysis
- ✅ Original text highlighting

**Note**: The first run will take 5-10 minutes to download and load models. Subsequent runs will be faster.

### Usage

1. **Input Article**:
   - **Article Title**: Enter the article title (optional)
   - **Full Article Text**: Paste the complete article text
     - **Important**: The first line must be the abstract, followed by the main text
     - Example format:
       ```
       This is the abstract of the paper. It describes the main findings...
       
       Introduction: The study aims to...
       Methods: We conducted...
       Results: Our findings show...
       Conclusion: In summary...
       ```

2. **Select Style**:
   - **formal**: Academic style, preserves technical terminology, suitable for researchers
   - **plain**: Clear and accessible, suitable for general readers
   - **high_readability**: High readability, uses simple vocabulary, suitable for public

3. **Generate Summary**:
   - Click the "🚀 Generate Summary" button
   - First generation may take 30-60 seconds (building FAISS index)
   - Subsequent generations will be faster (about 10-20 seconds)

## 🔧 Technical Architecture

### Model Information

- **LLM**: Qwen/Qwen2.5-3B-Instruct
- **Embedding**: BAAI/bge-small-en-v1.5
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2

### Core Pipeline

1. **Text Preprocessing**: Normalize, clean, extract abstract and main text
2. **Dynamic Chunking**: Adjust chunk size based on article length
3. **Multi-Query Generation**: Split abstract into multiple sentence queries
4. **Dense Retrieval**: Vector retrieval using FAISS
5. **Cross-Encoder Reranking**: Precise scoring and ranking
6. **LLM Generation**: Generate summary based on retrieved context

## 📁 Project Structure

```
BioLaySumm/
├── backend/                    # Backend API server
│   ├── __init__.py            # Python package marker
│   ├── rag_engine.py          # Core RAG logic
│   ├── api.py                 # FastAPI REST API server
│   └── requirements.txt       # Backend dependencies
├── frontend/                   # Frontend web interface
│   └── public/
│       ├── index.html         # Main HTML file
│       ├── app.js             # JavaScript logic
│       └── style.css          # CSS styling
├── README.md                  # This file
└── .gitignore                # Git ignore file
```

## ⚠️ Troubleshooting

### Q: First run is very slow?

A: First run needs to download model files (~6GB), which takes 5-10 minutes. Subsequent runs will use cached files and be much faster.

### Q: Out of memory?

A: If you encounter OOM (Out of Memory) errors:
- Use CPU mode (will automatically downgrade, but slower)
- Reduce `max_new_tokens` parameter
- Use a smaller model

### Q: Cannot access the web interface?

A:
- Make sure the backend server is running (`python backend/api.py`)
- Check that you're accessing `http://localhost:8000` 
- If running on a server: Make sure firewall allows port 8000
- Check backend logs for any error messages

### Q: Model download failed?

A:
- Check network connection
- If in mainland China, you may need to set HuggingFace mirror:
  ```python
  import os
  os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
  ```


## 🎯 Features

### Three Readability Levels

1. **Formal**: 
   - Preserves technical terminology
   - Complex sentence structures
   - Academic press release style

2. **Plain**:
   - Explains technical terms
   - Moderate sentence length (15-20 words)
   - Clear and logical

3. **High Readability (Simple)**:
   - Short sentences (10-14 words)
   - Common everyday vocabulary
   - No jargon

### RAG Pipeline

- Multi-query retrieval from abstract sentences
- Dynamic chunking based on article length
- Two-stage retrieval (FAISS + Cross-Encoder)
- Context-aware summary generation

## 📄 License

This project is for academic research purposes only.


## 🙏 Acknowledgments

- Qwen Team for the LLM model
- BAAI for the embedding model
- LangChain for the RAG framework

---

**Note**: If you encounter any issues, please check:
1. Python version is 3.8+
2. All dependencies are installed
3. GPU drivers are correctly installed (if using GPU)
4. Network connection is normal (for downloading models)


