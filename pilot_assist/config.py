import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Vector Store Configuration
    VECTOR_DIMENSION: int = 1536  # OpenAI ada-002 embedding dimension
    INDEX_TYPE: str = "L2"  # FAISS index type
    TOP_K_RESULTS: int = 5  # Number of relevant documents to retrieve
    
    # API Configuration
    API_TITLE: str = "Aviation RAG System"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # 1 hour in seconds
    
    class Config:
        env_file = ".env"

settings = Settings() 