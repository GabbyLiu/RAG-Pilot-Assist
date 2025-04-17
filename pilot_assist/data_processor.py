from typing import List, Dict, Generator
import openai
from config import settings
import tiktoken
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        openai.api_key = settings.OPENAI_API_KEY

    def chunk_text(self, text: str) -> Generator[str, None, None]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: Input text to chunk
            
        Yields:
            Text chunks
        """
        tokens = self.encoding.encode(text)
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk = tokens[start:end]
            yield self.encoding.decode(chunk)
            start = end - overlap

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the embedding model.
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embeddings
        """
        return self.embedding_model.encode(texts).tolist()

    def process_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Process a document by chunking it and preparing it for storage.
        
        Args:
            text: Document text
            metadata: Additional document metadata
            
        Returns:
            List of processed document chunks with metadata
        """
        if metadata is None:
            metadata = {}
            
        chunks = list(self.chunk_text(text))
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "chunk_index": i,
                "metadata": metadata
            }
            documents.append(doc)
            
        return documents

    def prepare_documents_for_storage(self, documents: List[Dict]) -> tuple[List[Dict], List[List[float]]]:
        """
        Prepare documents for storage by generating embeddings.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (documents, embeddings)
        """
        texts = [doc["text"] for doc in documents]
        embeddings = self.get_embeddings(texts)
        return documents, embeddings

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """
        Generate a response using GPT-4 with the provided context.
        
        Args:
            query: User query
            context: Retrieved relevant documents
            
        Returns:
            Generated response
        """
        # Prepare context for the prompt
        context_text = "\n\n".join([doc["text"] for doc in context])
        
        # Create the prompt
        prompt = f"""You are an aviation expert assistant. Use the following context to answer the question.
        If you cannot answer the question based on the context, say so.

        Context:
        {context_text}

        Question: {query}

        Answer:"""

        # Generate response using GPT-4
        response = openai.ChatCompletion.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an aviation expert assistant. Provide accurate, concise, and professional answers based on the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip() 