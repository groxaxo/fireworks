"""Database service for PostgreSQL with pgvector."""

import os
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from typing import List, Dict, Any, Optional
import json

class DatabaseService:
    """Service for PostgreSQL database operations."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize database service."""
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "fireworks_db")
        self.user = user or os.getenv("POSTGRES_USER", "fireworks_user")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "fireworks_password")
        
        self.connection = None
    
    def connect(self):
        """Connect to the database."""
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
    
    def close(self):
        """Close database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def store_embedding(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store an embedding in the database."""
        self.connect()
        
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embeddings (content, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (content, embedding, Json(metadata or {}))
            )
            embedding_id = cursor.fetchone()[0]
            self.connection.commit()
        
        return embedding_id
    
    def search_similar_embeddings(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity."""
        self.connect()
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM embeddings
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, threshold, query_embedding, limit)
            )
            results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def store_search_history(
        self,
        query: str,
        results: Dict[str, Any],
        model_used: str
    ) -> int:
        """Store search history."""
        self.connect()
        
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO search_history (query, results, model_used)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (query, Json(results), model_used)
            )
            search_id = cursor.fetchone()[0]
            self.connection.commit()
        
        return search_id
    
    def store_deep_search_results(
        self,
        query: str,
        search_depth: int,
        results: List[Dict[str, Any]]
    ) -> int:
        """Store deep search results."""
        self.connect()
        
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO deep_search_results (query, search_depth, results)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (query, search_depth, Json(results))
            )
            result_id = cursor.fetchone()[0]
            self.connection.commit()
        
        return result_id
    
    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent search history."""
        self.connect()
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, query, model_used, created_at
                FROM search_history
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            results = cursor.fetchall()
        
        return [dict(row) for row in results]
