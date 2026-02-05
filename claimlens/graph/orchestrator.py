"""LangGraph orchestrator for the ClaimLens verification pipeline."""

import logging
import time
from typing import Any, Dict, List, Optional, Literal, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from ..models.schemas import (
    Claim, 
    ClaimStatus,
    Evidence, 
    VerificationResult, 
    FinalReport,
    Verdict
)
from ..agents.decomposition import DecompositionAgent
from ..agents.search_architect import SearchArchitectAgent
from ..agents.scraper import ScraperAgent
from ..agents.verifier import VerifierAgent
from ..services.llm_service import LLMService
from ..config import settings

logger = logging.getLogger(__name__)


# Type definitions for state
class GraphState(dict):
    """State schema for the LangGraph verification pipeline.
    
    This TypedDict-like class defines all state variables that flow
    through the graph nodes.
    """
    original_text: str
    claims: List[Claim]
    current_claim_index: int
    search_queries: Dict[str, List[str]]
    evidence_buffer: Dict[str, List[Evidence]]
    verification_results: List[VerificationResult]
    iteration_counts: Dict[str, int]
    final_report: Optional[FinalReport]
    error: Optional[str]
    start_time: float


def create_initial_state(text: str) -> GraphState:
    """Create the initial state for a verification run.
    
    Args:
        text: The input text to verify
        
    Returns:
        Initial GraphState dictionary
    """
    return GraphState(
        original_text=text,
        claims=[],
        current_claim_index=0,
        search_queries={},
        evidence_buffer={},
        verification_results=[],
        iteration_counts={},
        final_report=None,
        error=None,
        start_time=time.time()
    )


class ClaimLensGraph:
    """LangGraph-based orchestrator for the claim verification pipeline.
    
    This class builds and manages the state machine that coordinates
    the decomposition, search, and verification agents.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        decomposition_agent: Optional[DecompositionAgent] = None,
        search_architect: Optional[SearchArchitectAgent] = None,
        scraper_agent: Optional[ScraperAgent] = None,
        verifier_agent: Optional[VerifierAgent] = None
    ):
        """Initialize the orchestrator with agents.
        
        Args:
            llm_service: Shared LLM service instance
            decomposition_agent: Custom decomposition agent
            search_architect: Custom search architect agent
            scraper_agent: Custom scraper agent
            verifier_agent: Custom verifier agent
        """
        self.llm_service = llm_service or LLMService()
        
        # Initialize agents (allow dependency injection)
        self.decomposition_agent = decomposition_agent or DecompositionAgent(self.llm_service)
        self.search_architect = search_architect or SearchArchitectAgent(self.llm_service)
        self.scraper_agent = scraper_agent or ScraperAgent(llm_service=self.llm_service)
        self.verifier_agent = verifier_agent or VerifierAgent(llm_service=self.llm_service)
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
        
        logger.info("ClaimLensGraph initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine.
        
        Returns:
            Configured StateGraph instance
        """
        # Create the graph with state schema
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("decompose_claims", self._decompose_claims_node)
        graph.add_node("prepare_claim", self._prepare_claim_node)
        graph.add_node("generate_queries", self._generate_queries_node)
        graph.add_node("search_evidence", self._search_evidence_node)
        graph.add_node("verify_claim", self._verify_claim_node)
        graph.add_node("aggregate_results", self._aggregate_results_node)
        graph.add_node("generate_report", self._generate_report_node)
        
        # Add edges
        # START -> decompose_claims
        graph.add_edge(START, "decompose_claims")
        
        # decompose_claims -> prepare_claim or generate_report (if no claims)
        graph.add_conditional_edges(
            "decompose_claims",
            self._route_after_decomposition,
            {
                "prepare_claim": "prepare_claim",
                "generate_report": "generate_report"
            }
        )
        
        # prepare_claim -> generate_queries
        graph.add_edge("prepare_claim", "generate_queries")
        
        # generate_queries -> search_evidence
        graph.add_edge("generate_queries", "search_evidence")
        
        # search_evidence -> verify_claim
        graph.add_edge("search_evidence", "verify_claim")
        
        # verify_claim -> decision: continue searching, next claim, or aggregate
        graph.add_conditional_edges(
            "verify_claim",
            self._route_after_verification,
            {
                "generate_queries": "generate_queries",  # Continue searching
                "prepare_claim": "prepare_claim",         # Next claim
                "aggregate_results": "aggregate_results"  # All claims done
            }
        )
        
        # aggregate_results -> generate_report
        graph.add_edge("aggregate_results", "generate_report")
        
        # generate_report -> END
        graph.add_edge("generate_report", END)
        
        return graph
    
    # ==================== Node Functions ====================
    
    def _decompose_claims_node(self, state: GraphState) -> GraphState:
        """Decompose input text into atomic claims.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with extracted claims
        """
        logger.info("Decomposing text into claims...")
        
        try:
            text = state["original_text"]
            claims = self.decomposition_agent.decompose(text)
            
            # Validate claims
            claims = self.decomposition_agent.validate_claims(claims)
            
            # Initialize iteration counts for each claim
            iteration_counts = {claim.id: 0 for claim in claims}
            
            logger.info(f"Extracted {len(claims)} valid claims")
            
            return {
                **state,
                "claims": claims,
                "iteration_counts": iteration_counts
            }
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return {
                **state,
                "error": f"Decomposition failed: {str(e)}"
            }
    
    def _prepare_claim_node(self, state: GraphState) -> GraphState:
        """Prepare the current claim for verification.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with current claim marked as processing
        """
        claims = state["claims"]
        current_index = state["current_claim_index"]
        
        if current_index < len(claims):
            current_claim = claims[current_index]
            current_claim.status = ClaimStatus.SEARCHING
            
            logger.info(
                f"Processing claim {current_index + 1}/{len(claims)}: "
                f"{current_claim.text[:50]}..."
            )
        
        return state
    
    def _generate_queries_node(self, state: GraphState) -> GraphState:
        """Generate search queries for the current claim.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with search queries
        """
        claims = state["claims"]
        current_index = state["current_claim_index"]
        current_claim = claims[current_index]
        
        try:
            # Check if this is a retry (need refined queries)
            iteration = state["iteration_counts"].get(current_claim.id, 0)
            existing_queries = state["search_queries"].get(current_claim.id, [])
            
            if iteration > 0 and existing_queries:
                # Generate refined queries based on previous results
                evidence_gap = "Need more relevant evidence for verification"
                queries = self.search_architect.generate_refined_queries(
                    current_claim,
                    existing_queries,
                    evidence_gap
                )
            else:
                # Generate initial queries
                queries = self.search_architect.generate_queries(current_claim)
            
            # Update queries in state
            all_queries = existing_queries + queries
            search_queries = {**state["search_queries"], current_claim.id: all_queries}
            
            # Increment iteration count
            iteration_counts = {
                **state["iteration_counts"],
                current_claim.id: iteration + 1
            }
            
            logger.debug(f"Generated {len(queries)} queries for claim")
            
            return {
                **state,
                "search_queries": search_queries,
                "iteration_counts": iteration_counts
            }
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return state
    
    def _search_evidence_node(self, state: GraphState) -> GraphState:
        """Search for evidence using generated queries.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with evidence buffer
        """
        claims = state["claims"]
        current_index = state["current_claim_index"]
        current_claim = claims[current_index]
        
        try:
            # Get the most recent queries
            queries = state["search_queries"].get(current_claim.id, [])[-3:]
            
            if not queries:
                logger.warning("No queries available for search")
                return state
            
            # Execute search and filter
            evidence = self.scraper_agent.search_and_filter(
                current_claim,
                queries
            )
            
            # Merge with existing evidence
            existing_evidence = state["evidence_buffer"].get(current_claim.id, [])
            
            # Deduplicate by URL
            seen_urls = {e.url for e in existing_evidence}
            new_evidence = [e for e in evidence if e.url not in seen_urls]
            
            all_evidence = existing_evidence + new_evidence
            
            # Keep top evidence by relevance
            all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
            all_evidence = all_evidence[:settings.MAX_EVIDENCE_PER_CLAIM]
            
            evidence_buffer = {
                **state["evidence_buffer"],
                current_claim.id: all_evidence
            }
            
            logger.info(f"Collected {len(all_evidence)} evidence pieces")
            
            return {
                **state,
                "evidence_buffer": evidence_buffer
            }
            
        except Exception as e:
            logger.error(f"Evidence search failed: {e}")
            return state
    
    def _verify_claim_node(self, state: GraphState) -> GraphState:
        """Verify the current claim against collected evidence.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with verification result
        """
        claims = state["claims"]
        current_index = state["current_claim_index"]
        current_claim = claims[current_index]
        
        try:
            current_claim.status = ClaimStatus.VERIFYING
            
            # Get evidence for this claim
            evidence = state["evidence_buffer"].get(current_claim.id, [])
            iteration = state["iteration_counts"].get(current_claim.id, 1)
            
            # Verify the claim
            result = self.verifier_agent.verify_with_retry(
                current_claim,
                evidence,
                iteration
            )
            
            # Store result (we'll finalize in routing or aggregation)
            # Using a temporary key to check in routing
            return {
                **state,
                "_current_result": result
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            
            # Create a failure result
            result = VerificationResult(
                claim=current_claim,
                evidence_list=[],
                verdict=Verdict.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning=f"Verification error: {str(e)}"
            )
            
            return {
                **state,
                "_current_result": result
            }
    
    def _aggregate_results_node(self, state: GraphState) -> GraphState:
        """Aggregate all verification results.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with aggregated results
        """
        logger.info("Aggregating verification results...")
        
        # Results should already be collected
        results = state["verification_results"]
        
        logger.info(
            f"Aggregated {len(results)} verification results: "
            f"{sum(1 for r in results if r.verdict == Verdict.SUPPORTED)} supported, "
            f"{sum(1 for r in results if r.verdict == Verdict.REFUTED)} refuted, "
            f"{sum(1 for r in results if r.verdict == Verdict.NOT_ENOUGH_INFO)} inconclusive"
        )
        
        return state
    
    def _generate_report_node(self, state: GraphState) -> GraphState:
        """Generate the final verification report.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with final report
        """
        logger.info("Generating final report...")
        
        try:
            # Create the report
            report = FinalReport(
                original_text=state["original_text"],
                claims=state["claims"],
                verification_results=state["verification_results"],
                processing_time_seconds=time.time() - state["start_time"]
            )
            
            # Calculate trust score
            report.overall_trust_score = report.calculate_trust_score()
            
            # Generate summary
            report.summary = self._generate_summary(report)
            
            logger.info(f"Report generated. Trust score: {report.overall_trust_score:.2f}")
            
            return {
                **state,
                "final_report": report
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            
            # Create minimal report on failure
            report = FinalReport(
                original_text=state["original_text"],
                claims=state["claims"],
                verification_results=state["verification_results"],
                overall_trust_score=0.0,
                summary=f"Report generation failed: {str(e)}",
                processing_time_seconds=time.time() - state["start_time"]
            )
            
            return {
                **state,
                "final_report": report
            }
    
    # ==================== Routing Functions ====================
    
    def _route_after_decomposition(self, state: GraphState) -> str:
        """Route after claim decomposition.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name
        """
        if state.get("error"):
            return "generate_report"
        
        claims = state.get("claims", [])
        
        if not claims:
            logger.warning("No claims extracted, generating report")
            return "generate_report"
        
        return "prepare_claim"
    
    def _route_after_verification(self, state: GraphState) -> str:
        """Route after claim verification - decide next action.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name
        """
        claims = state["claims"]
        current_index = state["current_claim_index"]
        current_claim = claims[current_index]
        current_result = state.get("_current_result")
        
        if current_result is None:
            # Error case - move to next claim
            return self._finalize_and_route_next(state)
        
        # Check if we should continue searching for this claim
        iteration = state["iteration_counts"].get(current_claim.id, 1)
        
        should_continue = self.verifier_agent.should_continue_searching(
            current_result,
            iteration
        )
        
        if should_continue:
            logger.debug(f"Continuing search for claim (iteration {iteration + 1})")
            return "generate_queries"
        
        # Finalize this claim and decide next action
        return self._finalize_and_route_next(state)
    
    def _finalize_and_route_next(self, state: GraphState) -> str:
        """Finalize current claim result and route to next action.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node name
        """
        claims = state["claims"]
        current_index = state["current_claim_index"]
        current_result = state.get("_current_result")
        
        # Add result to verification_results
        if current_result:
            current_result.claim.status = ClaimStatus.COMPLETED
            state["verification_results"].append(current_result)
        
        # Clean up temporary state
        if "_current_result" in state:
            del state["_current_result"]
        
        # Move to next claim
        state["current_claim_index"] = current_index + 1
        
        # Check if more claims to process
        if state["current_claim_index"] < len(claims):
            return "prepare_claim"
        
        return "aggregate_results"
    
    # ==================== Helper Functions ====================
    
    def _generate_summary(self, report: FinalReport) -> str:
        """Generate a human-readable summary of the report.
        
        Args:
            report: The final report
            
        Returns:
            Summary string
        """
        total = len(report.verification_results)
        
        if total == 0:
            return "No verifiable claims were found in the text."
        
        supported = sum(
            1 for r in report.verification_results 
            if r.verdict == Verdict.SUPPORTED
        )
        refuted = sum(
            1 for r in report.verification_results 
            if r.verdict == Verdict.REFUTED
        )
        inconclusive = sum(
            1 for r in report.verification_results 
            if r.verdict == Verdict.NOT_ENOUGH_INFO
        )
        
        summary_parts = [
            f"Analyzed {total} claims from the text.",
            f"Results: {supported} supported, {refuted} refuted, {inconclusive} inconclusive.",
            f"Overall trust score: {report.overall_trust_score:.1%}"
        ]
        
        if refuted > 0:
            refuted_claims = [
                r.claim.text for r in report.verification_results
                if r.verdict == Verdict.REFUTED
            ]
            summary_parts.append(
                f"Refuted claims: {'; '.join(refuted_claims[:3])}"
            )
        
        return " ".join(summary_parts)
    
    # ==================== Public Interface ====================
    
    def run(self, text: str) -> FinalReport:
        """Run the verification pipeline synchronously.
        
        Args:
            text: Input text to verify
            
        Returns:
            Final verification report
        """
        logger.info(f"Starting verification for text: {text[:100]}...")
        
        initial_state = create_initial_state(text)
        
        # Run the graph
        final_state = self.compiled_graph.invoke(initial_state)
        
        report = final_state.get("final_report")
        
        if report is None:
            # Create error report
            report = FinalReport(
                original_text=text,
                claims=[],
                verification_results=[],
                overall_trust_score=0.0,
                summary=final_state.get("error", "Unknown error occurred"),
                processing_time_seconds=time.time() - initial_state["start_time"]
            )
        
        return report
    
    async def arun(self, text: str) -> FinalReport:
        """Run the verification pipeline asynchronously.
        
        Args:
            text: Input text to verify
            
        Returns:
            Final verification report
        """
        import asyncio
        return await asyncio.to_thread(self.run, text)
    
    def stream(self, text: str):
        """Stream verification progress.
        
        Args:
            text: Input text to verify
            
        Yields:
            State updates as they occur
        """
        logger.info(f"Starting streaming verification...")
        
        initial_state = create_initial_state(text)
        
        for state_update in self.compiled_graph.stream(initial_state):
            yield state_update


# ==================== Module-level convenience functions ====================

def create_graph(
    llm_service: Optional[LLMService] = None,
    **kwargs
) -> ClaimLensGraph:
    """Create a ClaimLens graph instance.
    
    Args:
        llm_service: Optional LLM service instance
        **kwargs: Additional arguments for ClaimLensGraph
        
    Returns:
        Configured ClaimLensGraph instance
    """
    return ClaimLensGraph(llm_service=llm_service, **kwargs)


def run_verification(text: str, **kwargs) -> FinalReport:
    """Run verification on text.
    
    Args:
        text: Input text to verify
        **kwargs: Arguments for ClaimLensGraph
        
    Returns:
        Final verification report
    """
    graph = create_graph(**kwargs)
    return graph.run(text)


async def run_verification_async(text: str, **kwargs) -> FinalReport:
    """Run verification asynchronously.
    
    Args:
        text: Input text to verify
        **kwargs: Arguments for ClaimLensGraph
        
    Returns:
        Final verification report
    """
    graph = create_graph(**kwargs)
    return await graph.arun(text)
