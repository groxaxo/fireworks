"""ChromaDB service for vector storage and retrieval."""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class ChromaService:
    """Service for ChromaDB operations."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "fireworks_findings"
    ):
        """Initialize ChromaDB service."""
        self.host = host or os.getenv("CHROMA_HOST", "localhost")
        self.port = port or int(os.getenv("CHROMA_PORT", "8000"))
        self.collection_name = collection_name
        
        # Initialize client
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Storage for search findings and results"}
        )
    
    def add_finding(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Add a finding to the collection."""
        if document_id is None:
            import uuid
            document_id = str(uuid.uuid4())
        
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}],
            ids=[document_id]
        )
        
        return document_id
    
    def add_findings_batch(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add multiple findings in batch."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in contents]
        
        if metadatas is None:
            metadatas = [{} for _ in contents]
        
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search_findings(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar findings."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """Search findings by text query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return results
    
    def get_finding(self, document_id: str) -> Dict[str, Any]:
        """Get a specific finding by ID."""
        result = self.collection.get(ids=[document_id])
        return result
    
    def update_finding(
        self,
        document_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update an existing finding."""
        update_data = {"ids": [document_id]}
        
        if content is not None:
            update_data["documents"] = [content]
        if embedding is not None:
            update_data["embeddings"] = [embedding]
        if metadata is not None:
            update_data["metadatas"] = [metadata]
        
        self.collection.update(**update_data)
    
    def delete_finding(self, document_id: str):
        """Delete a finding."""
        self.collection.delete(ids=[document_id])
    
    def count_findings(self) -> int:
        """Get total number of findings."""
        return self.collection.count()
    
    def clear_collection(self):
        """Clear all findings from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Storage for search findings and results"}
        )
