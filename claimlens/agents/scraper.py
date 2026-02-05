"""Scraper & Filter Agent for retrieving and filtering evidence."""

import logging
from typing import List, Optional

from ..models.schemas import Evidence, Claim
from ..services.llm_service import LLMService
from ..services.search_service import SearchService, get_search_service
from ..config import settings

logger = logging.getLogger(__name__)


class ScraperAgent:
    """Agent responsible for executing searches and filtering evidence.
    
    This agent takes search queries, executes web searches, retrieves results,
    and filters them for relevance and quality.
    """
    
    RELEVANCE_PROMPT = """You are an expert evidence evaluator for fact-checking. Assess how relevant each piece of evidence is to verifying the given claim.

For each evidence snippet, provide:
1. relevance_score (0.0 to 1.0): How directly relevant is this to the claim?
2. is_relevant (boolean): Is this evidence useful for verification?
3. brief_reason: One sentence explaining relevance assessment

Scoring guide:
- 0.9-1.0: Directly addresses the claim with specific information
- 0.7-0.8: Related and useful, provides good context
- 0.5-0.6: Tangentially related, might help
- 0.3-0.4: Loosely related, limited usefulness
- 0.0-0.2: Not relevant to the claim"""

    def __init__(
        self, 
        search_service: Optional[SearchService] = None,
        llm_service: Optional[LLMService] = None
    ):
        """Initialize the Scraper Agent.
        
        Args:
            search_service: Search service for web queries
            llm_service: LLM service for relevance filtering
        """
        self.search_service = search_service or get_search_service()
        self.llm_service = llm_service or LLMService()
    
    def search_and_filter(
        self,
        claim: Claim,
        queries: List[str],
        max_results_per_query: int = None,
        max_total_evidence: int = None
    ) -> List[Evidence]:
        """Execute searches and filter results for relevance.
        
        Args:
            claim: The claim being verified
            queries: List of search queries to execute
            max_results_per_query: Max results per query (defaults to config)
            max_total_evidence: Max total evidence to return (defaults to config)
            
        Returns:
            Filtered and scored list of Evidence objects
        """
        max_results_per_query = max_results_per_query or settings.SEARCH_RESULTS_PER_QUERY
        max_total_evidence = max_total_evidence or settings.MAX_EVIDENCE_PER_CLAIM
        
        all_evidence = []
        seen_urls = set()
        
        # Execute searches for each query
        for query in queries:
            logger.debug(f"Executing search: {query}")
            
            try:
                results = self.search_service.search(
                    query=query,
                    num_results=max_results_per_query
                )
                
                # Deduplicate by URL
                for evidence in results:
                    if evidence.url not in seen_urls:
                        seen_urls.add(evidence.url)
                        all_evidence.append(evidence)
                        
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                continue
        
        if not all_evidence:
            logger.warning(f"No evidence found for claim: {claim.text[:50]}...")
            return []
        
        logger.info(f"Retrieved {len(all_evidence)} unique evidence pieces")
        
        # Filter and score evidence for relevance
        filtered_evidence = self._filter_by_relevance(claim, all_evidence)
        
        # Sort by relevance and return top results
        filtered_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        
        return filtered_evidence[:max_total_evidence]
    
    def _filter_by_relevance(
        self, 
        claim: Claim, 
        evidence_list: List[Evidence]
    ) -> List[Evidence]:
        """Filter evidence by relevance to the claim using LLM.
        
        Args:
            claim: The claim being verified
            evidence_list: List of evidence to filter
            
        Returns:
            Filtered list with updated relevance scores
        """
        if not evidence_list:
            return []
        
        # Format evidence for the prompt
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}]\nURL: {e.url}\nTitle: {e.title}\nContent: {e.snippet[:500]}"
            for i, e in enumerate(evidence_list)
        ])
        
        user_prompt = f"""Evaluate the relevance of each evidence piece to this claim:

CLAIM: "{claim.text}"

EVIDENCE:
{evidence_text}

Assess each evidence piece's relevance to verifying or refuting the claim."""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_index": {"type": "integer"},
                                "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                                "is_relevant": {"type": "boolean"},
                                "brief_reason": {"type": "string"}
                            },
                            "required": ["evidence_index", "relevance_score", "is_relevant"]
                        }
                    }
                },
                "required": ["evaluations"]
            }
            
            result = self.llm_service.generate_structured(
                system_prompt=self.RELEVANCE_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.1
            )
            
            # Update evidence with new relevance scores
            evaluations = {
                e["evidence_index"]: e 
                for e in result.get("evaluations", [])
            }
            
            filtered = []
            for i, evidence in enumerate(evidence_list):
                eval_data = evaluations.get(i + 1)  # 1-indexed in prompt
                
                if eval_data:
                    evidence.relevance_score = eval_data.get("relevance_score", evidence.relevance_score)
                    
                    if eval_data.get("is_relevant", True):
                        filtered.append(evidence)
                    else:
                        logger.debug(
                            f"Filtered out evidence: {evidence.url} "
                            f"(reason: {eval_data.get('brief_reason', 'not relevant')})"
                        )
                else:
                    # Keep evidence if no evaluation (fallback)
                    filtered.append(evidence)
            
            logger.info(f"Filtered to {len(filtered)} relevant evidence pieces")
            return filtered
            
        except Exception as e:
            logger.error(f"Relevance filtering failed: {e}")
            # Return all evidence with original scores on failure
            return evidence_list
    
    async def asearch_and_filter(
        self,
        claim: Claim,
        queries: List[str],
        max_results_per_query: int = None,
        max_total_evidence: int = None
    ) -> List[Evidence]:
        """Async version of search_and_filter.
        
        Args:
            claim: The claim being verified
            queries: List of search queries
            max_results_per_query: Max results per query
            max_total_evidence: Max total evidence
            
        Returns:
            Filtered list of Evidence objects
        """
        import asyncio
        
        max_results_per_query = max_results_per_query or settings.SEARCH_RESULTS_PER_QUERY
        max_total_evidence = max_total_evidence or settings.MAX_EVIDENCE_PER_CLAIM
        
        # Execute searches in parallel
        search_tasks = [
            self.search_service.asearch(query, max_results_per_query)
            for query in queries
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        all_evidence = []
        seen_urls = set()
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
                continue
            
            for evidence in result:
                if evidence.url not in seen_urls:
                    seen_urls.add(evidence.url)
                    all_evidence.append(evidence)
        
        if not all_evidence:
            return []
        
        # Filter in thread (LLM call is sync)
        filtered = await asyncio.to_thread(
            self._filter_by_relevance, claim, all_evidence
        )
        
        filtered.sort(key=lambda e: e.relevance_score, reverse=True)
        return filtered[:max_total_evidence]
    
    def extract_key_snippets(
        self,
        claim: Claim,
        evidence_list: List[Evidence]
    ) -> List[Evidence]:
        """Extract the most relevant snippets from evidence.
        
        Args:
            claim: The claim being verified
            evidence_list: List of evidence with full snippets
            
        Returns:
            Evidence list with refined, focused snippets
        """
        if not evidence_list:
            return []
        
        user_prompt = f"""Extract the most relevant sentences from each evidence piece for this claim:

CLAIM: "{claim.text}"

For each evidence, identify 1-2 sentences that are most relevant to verifying the claim.

EVIDENCE:
{chr(10).join([f'[{i+1}] {e.snippet}' for i, e in enumerate(evidence_list)])}"""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "extracted": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_index": {"type": "integer"},
                                "key_sentences": {"type": "string"}
                            },
                            "required": ["evidence_index", "key_sentences"]
                        }
                    }
                },
                "required": ["extracted"]
            }
            
            result = self.llm_service.generate_structured(
                system_prompt="You extract key evidence sentences for fact-checking.",
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.1
            )
            
            extractions = {
                e["evidence_index"]: e["key_sentences"]
                for e in result.get("extracted", [])
            }
            
            for i, evidence in enumerate(evidence_list):
                if (i + 1) in extractions:
                    evidence.snippet = extractions[i + 1]
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Snippet extraction failed: {e}")
            return evidence_list
