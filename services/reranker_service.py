"""Reranker service for improving search result relevance."""

import os
from typing import List, Dict, Any, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class RerankerService:
    """
    Service for reranking search results using various strategies.
    
    Supports:
    - Embedding-based reranking
    - Cross-encoder models (via external API)
    - Hybrid scoring (combining multiple signals)
    """
    
    def __init__(self):
        """Initialize reranker service."""
        self.strategies = {
            "cosine": self._cosine_similarity,
            "reciprocal": self._reciprocal_rank_fusion,
            "hybrid": self._hybrid_score
        }
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        query_embedding: List[float] = None,
        strategy: str = "cosine",
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using the specified strategy.
        
        Args:
            query: The search query
            results: List of search results
            query_embedding: Query embedding vector (if available)
            strategy: Reranking strategy to use
            top_k: Return only top k results
            
        Returns:
            Reranked list of results
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if strategy == "cosine" and query_embedding is None:
            raise ValueError("Query embedding required for cosine strategy")
        
        # Apply reranking strategy
        if strategy == "cosine":
            reranked = self._rerank_by_cosine(query_embedding, results)
        elif strategy == "reciprocal":
            reranked = self._rerank_by_reciprocal(results)
        else:
            reranked = self._rerank_by_hybrid(query, query_embedding, results)
        
        # Return top k if specified
        if top_k:
            return reranked[:top_k]
        
        return reranked
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if NUMPY_AVAILABLE:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        else:
            # Fallback implementation without numpy
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def _rerank_by_cosine(
        self,
        query_embedding: List[float],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results by cosine similarity with query embedding."""
        scored_results = []
        
        for result in results:
            if "embedding" in result:
                similarity = self._cosine_similarity(
                    query_embedding,
                    result["embedding"]
                )
                result["rerank_score"] = similarity
                scored_results.append((similarity, result))
            else:
                # Keep results without embeddings at the end
                result["rerank_score"] = 0.0
                scored_results.append((0.0, result))
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [result for _, result in scored_results]
    
    def _reciprocal_rank_fusion(
        self,
        rankings: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine multiple rankings.
        
        RRF formula: score(d) = sum(1 / (k + rank(d)))
        """
        scores = {}
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                doc_id = doc.get("id", str(doc))
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0, "doc": doc}
                scores[doc_id]["score"] += 1 / (k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        results = []
        for item in sorted_docs:
            doc = item["doc"]
            doc["rerank_score"] = item["score"]
            results.append(doc)
        
        return results
    
    def _rerank_by_reciprocal(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank using reciprocal rank (simple version)."""
        for rank, result in enumerate(results, start=1):
            result["rerank_score"] = 1 / rank
        
        return results
    
    def _hybrid_score(
        self,
        query: str,
        query_embedding: List[float],
        results: List[Dict[str, Any]],
        weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """Calculate hybrid score combining multiple signals."""
        if weights is None:
            weights = {
                "semantic": 0.5,  # Embedding similarity
                "lexical": 0.3,   # Keyword match
                "position": 0.2   # Original position
            }
        
        scored_results = []
        
        for idx, result in enumerate(results):
            score = 0.0
            
            # Semantic similarity (if embedding available)
            if query_embedding and "embedding" in result:
                semantic_score = self._cosine_similarity(
                    query_embedding,
                    result["embedding"]
                )
                score += weights["semantic"] * semantic_score
            
            # Lexical match (simple keyword overlap)
            if "content" in result or "snippet" in result:
                content = result.get("content", result.get("snippet", ""))
                lexical_score = self._keyword_overlap(query, content)
                score += weights["lexical"] * lexical_score
            
            # Position score (reciprocal of position)
            position_score = 1 / (idx + 1)
            score += weights["position"] * position_score
            
            result["rerank_score"] = score
            scored_results.append((score, result))
        
        # Sort by hybrid score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [result for _, result in scored_results]
    
    def _rerank_by_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank using hybrid strategy."""
        return self._hybrid_score(query, query_embedding, results)
    
    def _keyword_overlap(self, query: str, text: str) -> float:
        """Calculate simple keyword overlap score."""
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & text_terms)
        return overlap / len(query_terms)
