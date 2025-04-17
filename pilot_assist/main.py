from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
from collections import defaultdict

from config import settings
from vector_store import VectorStore
from data_processor import DataProcessor

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
data_processor = DataProcessor()

# Rate limiting
request_counts = defaultdict(lambda: {"count": 0, "reset_time": time.time()})

class Query(BaseModel):
    text: str

class Document(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class Response(BaseModel):
    answer: str
    sources: List[Dict]

def check_rate_limit():
    """Check if the request should be rate limited."""
    current_time = time.time()
    client_id = "default"  # In production, use actual client identification
    
    if current_time - request_counts[client_id]["reset_time"] >= settings.RATE_LIMIT_PERIOD:
        request_counts[client_id] = {"count": 0, "reset_time": current_time}
    
    request_counts[client_id]["count"] += 1
    
    if request_counts[client_id]["count"] > settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

@app.post("/api/query", response_model=Response)
async def query_endpoint(query: Query, _: None = Depends(check_rate_limit)):
    """
    Query the RAG system with a question.
    """
    try:
        # Generate embedding for the query
        query_embedding = data_processor.get_embeddings([query.text])[0]
        
        # Search for relevant documents
        relevant_docs = vector_store.search(query_embedding)
        
        if not relevant_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for your query."
            )
        
        # Extract documents from results
        context = [doc for doc, _ in relevant_docs]
        
        # Generate response
        answer = data_processor.generate_response(query.text, context)
        
        return Response(
            answer=answer,
            sources=[{"text": doc["text"], "metadata": doc["metadata"]} for doc in context]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)}"
        )

@app.post("/api/ingest")
async def ingest_endpoint(document: Document, _: None = Depends(check_rate_limit)):
    """
    Ingest a new document into the knowledge base.
    """
    try:
        # Process the document
        processed_docs = data_processor.process_document(document.text, document.metadata)
        
        # Prepare documents for storage
        docs, embeddings = data_processor.prepare_documents_for_storage(processed_docs)
        
        # Add to vector store
        vector_store.add_documents(docs, embeddings)
        
        return {"message": "Document successfully ingested", "chunks": len(docs)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while ingesting the document: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "version": settings.API_VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 