"""Search service wrappers for web search APIs."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from ..config import settings
from ..models.schemas import Evidence

logger = logging.getLogger(__name__)


class SearchService(ABC):
    """Abstract base class for search services."""
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Evidence]:
        """Execute a web search query.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of Evidence objects from search results
        """
        pass
    
    @abstractmethod
    async def asearch(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Evidence]:
        """Async version of search."""
        pass


class TavilySearchService(SearchService):
    """Search service using Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily search service.
        
        Args:
            api_key: Tavily API key (defaults to settings)
        """
        self.api_key = api_key or settings.TAVILY_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY environment variable."
            )
        
        self._client = None
    
    @property
    def client(self):
        """Lazy load Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "tavily-python package required. Install with: pip install tavily-python"
                )
        return self._client
    
    def search(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Evidence]:
        """Execute search using Tavily API.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of Evidence objects
        """
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results,
                include_answer=False,
                include_raw_content=False
            )
            
            evidence_list = []
            for result in response.get("results", []):
                # Estimate relevance based on position and score
                relevance = result.get("score", 0.5)
                
                # Assess source quality
                url = result.get("url", "")
                source_quality = self._assess_source_quality(url)
                
                evidence = Evidence(
                    url=url,
                    title=result.get("title", "Unknown"),
                    snippet=result.get("content", "")[:1000],  # Limit snippet length
                    relevance_score=relevance,
                    source_quality=source_quality
                )
                evidence_list.append(evidence)
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    async def asearch(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Evidence]:
        """Async search using Tavily API.
        
        Note: Tavily doesn't have native async support, 
        so we wrap the sync call.
        """
        import asyncio
        return await asyncio.to_thread(self.search, query, num_results)
    
    def _assess_source_quality(self, url: str) -> str:
        """Assess the quality of a source based on URL.
        
        Args:
            url: URL of the source
            
        Returns:
            Quality rating: "high", "medium", or "low"
        """
        high_quality_domains = [
            "wikipedia.org", "britannica.com", "reuters.com",
            "bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com",
            "theguardian.com", "nature.com", "science.org", "sciencedirect.com",
            "gov", "edu", "who.int", "un.org", "nasa.gov", "cdc.gov",
            "nih.gov", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"
        ]
        
        medium_quality_domains = [
            "cnn.com", "foxnews.com", "nbcnews.com", "abcnews.go.com",
            "forbes.com", "businessinsider.com", "techcrunch.com",
            "wired.com", "arstechnica.com", "theverge.com"
        ]
        
        url_lower = url.lower()
        
        for domain in high_quality_domains:
            if domain in url_lower:
                return "high"
        
        for domain in medium_quality_domains:
            if domain in url_lower:
                return "medium"
        
        return "low"


class SerpAPISearchService(SearchService):
    """Search service using SerpAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize SerpAPI search service.
        
        Args:
            api_key: SerpAPI key (defaults to settings)
        """
        self.api_key = api_key or settings.SERPAPI_KEY
        
        if not self.api_key:
            raise ValueError(
                "SerpAPI key is required. Set SERPAPI_KEY environment variable."
            )
    
    def search(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Evidence]:
        """Execute search using SerpAPI.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of Evidence objects
        """
        try:
            from serpapi import GoogleSearch
        except ImportError:
            raise ImportError(
                "google-search-results package required. "
                "Install with: pip install google-search-results"
            )
        
        try:
            params = {
                "api_key": self.api_key,
                "q": query,
                "num": num_results,
                "engine": "google"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            evidence_list = []
            organic_results = results.get("organic_results", [])
            
            for i, result in enumerate(organic_results[:num_results]):
                # Calculate relevance score based on position
                relevance = 1.0 - (i * 0.1)  # Decreasing by position
                relevance = max(0.5, relevance)
                
                url = result.get("link", "")
                source_quality = self._assess_source_quality(url)
                
                evidence = Evidence(
                    url=url,
                    title=result.get("title", "Unknown"),
                    snippet=result.get("snippet", "")[:1000],
                    relevance_score=relevance,
                    source_quality=source_quality
                )
                evidence_list.append(evidence)
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    async def asearch(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[Evidence]:
        """Async search using SerpAPI."""
        import asyncio
        return await asyncio.to_thread(self.search, query, num_results)
    
    def _assess_source_quality(self, url: str) -> str:
        """Assess source quality (same logic as Tavily)."""
        # Reuse the same quality assessment logic
        high_quality_domains = [
            "wikipedia.org", "britannica.com", "reuters.com",
            "bbc.com", "bbc.co.uk", "nytimes.com", "gov", "edu"
        ]
        
        url_lower = url.lower()
        
        for domain in high_quality_domains:
            if domain in url_lower:
                return "high"
        
        return "medium"


def get_search_service(provider: Optional[str] = None) -> SearchService:
    """Factory function to get the appropriate search service.
    
    Args:
        provider: Search provider ("tavily" or "serpapi")
        
    Returns:
        SearchService instance
    """
    provider = provider or settings.SEARCH_PROVIDER
    
    if provider == "tavily":
        return TavilySearchService()
    elif provider == "serpapi":
        return SerpAPISearchService()
    else:
        raise ValueError(
            f"Unknown search provider: {provider}. "
            "Choose from: tavily, serpapi"
        )
