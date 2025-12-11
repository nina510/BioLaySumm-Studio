"""
FastAPI Backend for BioLaySumm RAG Demo
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from rag_engine import RAGEngine

app = FastAPI(title="BioLaySumm RAG API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG engine (loaded once at startup)
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global rag_engine
    print("\n" + "="*60)
    print("🚀 Starting BioLaySumm RAG API")
    print("="*60 + "\n")
    rag_engine = RAGEngine()
    print("✓ API ready to serve requests\n")


class SummarizeRequest(BaseModel):
    title: str
    article: str
    style: str = "plain"  # formal, plain, or high_readability


class RAGProcessInfo(BaseModel):
    num_queries: int
    queries: List[str]
    num_total_chunks: int
    dense_candidates: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    final_chunks: List[Dict[str, Any]]


class SummarizeResponse(BaseModel):
    summary: str
    word_count: int
    chunks_used: int
    queries: int
    style: str
    abstract: str = ""
    main_text: str = ""
    rag_process: Optional[RAGProcessInfo] = None


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate summary for an article"""
    
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    if not request.article.strip():
        raise HTTPException(status_code=400, detail="Article cannot be empty")
    
    try:
        result = rag_engine.generate_summary(
            title=request.title,
            article=request.article,
            version=request.style
        )
        
        response_data = {
            "summary": result["summary"],
            "word_count": result["word_count"],
            "chunks_used": result["chunks_used"],
            "queries": result["queries"],
            "style": request.style,
            "abstract": result.get("abstract", ""),
            "main_text": result.get("main_text", ""),
        }
        
        # Add RAG process info if available
        if "rag_process" in result:
            response_data["rag_process"] = result["rag_process"]
        
        return SummarizeResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": rag_engine is not None,
        "device": rag_engine.device if rag_engine else "unknown"
    }


# Serve React frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    # Development mode: serve from frontend/public
    frontend_dev = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
    if os.path.exists(frontend_dev):
        app.mount("/", StaticFiles(directory=frontend_dev, html=True), name="frontend")
    else:
        @app.get("/")
        async def root():
            return {"message": "Frontend not built. Run: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

