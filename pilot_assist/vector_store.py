import numpy as np
import faiss
from typing import List, Dict, Tuple
import pickle
import os
from config import settings

class VectorStore:
    def __init__(self):
        self.dimension = settings.VECTOR_DIMENSION
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[Dict] = []
        self.index_path = "faiss_index"
        self.documents_path = "documents.pkl"
        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing index and documents if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.documents_path, 'rb') as f:
                self.documents = pickle.load(f)

    def _save(self):
        """Save the index and documents to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.documents_path, 'wb') as f:
            pickle.dump(self.documents, f)

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries containing text and metadata
            embeddings: List of document embeddings
        """
        if not documents or not embeddings:
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents.extend(documents)
        
        # Save to disk
        self._save()

    def search(self, query_embedding: List[float], k: int = None) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents using the query embedding.
        
        Args:
            query_embedding: The embedding of the query
            k: Number of results to return (defaults to settings.TOP_K_RESULTS)
            
        Returns:
            List of tuples containing (document, distance)
        """
        if k is None:
            k = settings.TOP_K_RESULTS
            
        # Convert query embedding to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_array, k)
        
        # Return documents with their distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):  # Ensure index is valid
                results.append((self.documents[idx], float(distance)))
                
        return results

    def clear(self):
        """Clear the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self._save() 