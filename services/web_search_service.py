"""Web search service using SerpAPI."""

import os
import requests
from typing import Dict, Any, List, Optional

class WebSearchService:
    """Service for web search functionality."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize web search service."""
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SerpAPI key is required for web search")
        
        self.base_url = "https://serpapi.com/search"
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search"
    ) -> Dict[str, Any]:
        """Perform web search."""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "engine": "google"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "error": str(e),
                "organic_results": []
            }
    
    def extract_snippets(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract text snippets from search results."""
        snippets = []
        
        if "organic_results" in search_results:
            for result in search_results["organic_results"]:
                if "snippet" in result:
                    snippets.append(result["snippet"])
        
        return snippets
    
    def deep_search(
        self,
        query: str,
        depth: int = 3,
        results_per_level: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform deep search by following links and extracting related queries.
        
        Args:
            query: Initial search query
            depth: How many levels deep to search
            results_per_level: Number of results to process per level
            
        Returns:
            List of search results across all levels
        """
        all_results = []
        queries_to_process = [(query, 0)]
        processed_queries = set()
        
        while queries_to_process and len(all_results) < depth * results_per_level:
            current_query, level = queries_to_process.pop(0)
            
            if current_query in processed_queries or level >= depth:
                continue
            
            processed_queries.add(current_query)
            
            # Search for current query
            results = self.search(current_query, num_results=results_per_level)
            
            if "organic_results" in results:
                for result in results["organic_results"][:results_per_level]:
                    all_results.append({
                        "query": current_query,
                        "level": level,
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })
                    
                # Extract related searches for next level
                if level < depth - 1 and "related_searches" in results:
                    for related in results["related_searches"][:2]:
                        if "query" in related:
                            queries_to_process.append((related["query"], level + 1))
        
        return all_results
