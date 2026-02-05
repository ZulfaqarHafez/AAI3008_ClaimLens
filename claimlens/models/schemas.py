"""Pydantic data models for ClaimLens pipeline."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ClaimStatus(str, Enum):
    """Status of a claim in the verification pipeline."""
    PENDING = "pending"
    SEARCHING = "searching"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(str, Enum):
    """Verification verdict for a claim."""
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class Claim(BaseModel):
    """Represents an atomic, verifiable claim extracted from text."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The atomic claim text")
    source_sentence: str = Field(..., description="Original sentence the claim was extracted from")
    status: ClaimStatus = Field(default=ClaimStatus.PENDING, description="Current verification status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "claim-001",
                "text": "The Eiffel Tower is 330 meters tall.",
                "source_sentence": "The Eiffel Tower, standing at 330 meters, is one of the most visited monuments.",
                "status": "pending"
            }
        }


class Evidence(BaseModel):
    """Represents a piece of evidence retrieved from web search."""
    
    url: str = Field(..., description="URL of the evidence source")
    title: str = Field(..., description="Title of the source page")
    snippet: str = Field(..., description="Relevant text snippet from the source")
    relevance_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Relevance score of the evidence to the claim"
    )
    source_quality: Optional[str] = Field(
        default=None, 
        description="Quality assessment of the source (high/medium/low)"
    )
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.toureiffel.paris/en",
                "title": "The Eiffel Tower - Official Website",
                "snippet": "The Eiffel Tower is 330 metres (1,083 feet) tall...",
                "relevance_score": 0.95,
                "source_quality": "high"
            }
        }


class VerificationResult(BaseModel):
    """Result of verifying a single claim against evidence."""
    
    claim: Claim = Field(..., description="The claim that was verified")
    evidence_list: List[Evidence] = Field(
        default_factory=list, 
        description="List of evidence used for verification"
    )
    verdict: Verdict = Field(..., description="Final verdict for the claim")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score of the verdict"
    )
    reasoning: str = Field(..., description="Explanation of the verdict reasoning")
    iterations_used: int = Field(
        default=1, 
        description="Number of search iterations used"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim": {
                    "id": "claim-001",
                    "text": "The Eiffel Tower is 330 meters tall.",
                    "source_sentence": "The Eiffel Tower, standing at 330 meters...",
                    "status": "completed"
                },
                "evidence_list": [],
                "verdict": "SUPPORTED",
                "confidence": 0.92,
                "reasoning": "Multiple reliable sources confirm the Eiffel Tower's height as 330 meters.",
                "iterations_used": 1
            }
        }


class FinalReport(BaseModel):
    """Complete verification report for the input text."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = Field(..., description="The original input text")
    claims: List[Claim] = Field(default_factory=list, description="List of extracted claims")
    verification_results: List[VerificationResult] = Field(
        default_factory=list, 
        description="Verification results for each claim"
    )
    overall_trust_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Overall trustworthiness score of the text"
    )
    summary: Optional[str] = Field(
        default=None, 
        description="Summary of the verification findings"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_seconds: Optional[float] = Field(
        default=None, 
        description="Total processing time in seconds"
    )
    
    def calculate_trust_score(self) -> float:
        """Calculate the overall trust score based on verification results."""
        if not self.verification_results:
            return 0.0
        
        # Weight factors
        SUPPORT_WEIGHT = 0.5
        CONFIDENCE_WEIGHT = 0.3
        EVIDENCE_QUALITY_WEIGHT = 0.2
        
        # Calculate claim support ratio
        supported_count = sum(
            1 for r in self.verification_results 
            if r.verdict == Verdict.SUPPORTED
        )
        refuted_count = sum(
            1 for r in self.verification_results 
            if r.verdict == Verdict.REFUTED
        )
        total_claims = len(self.verification_results)
        
        # Penalize refuted claims more heavily
        support_ratio = (supported_count - refuted_count * 0.5) / total_claims
        support_ratio = max(0.0, min(1.0, support_ratio))
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in self.verification_results) / total_claims
        
        # Calculate evidence quality score
        quality_scores = []
        for result in self.verification_results:
            for evidence in result.evidence_list:
                if evidence.source_quality == "high":
                    quality_scores.append(1.0)
                elif evidence.source_quality == "medium":
                    quality_scores.append(0.6)
                else:
                    quality_scores.append(0.3)
        
        evidence_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Weighted average
        trust_score = (
            support_ratio * SUPPORT_WEIGHT +
            avg_confidence * CONFIDENCE_WEIGHT +
            evidence_quality * EVIDENCE_QUALITY_WEIGHT
        )
        
        return round(trust_score, 3)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "report-001",
                "original_text": "The Eiffel Tower is 330 meters tall and was built in 1889.",
                "claims": [],
                "verification_results": [],
                "overall_trust_score": 0.85,
                "summary": "2 claims verified: 2 supported, 0 refuted."
            }
        }


class GraphState(BaseModel):
    """State schema for LangGraph orchestration."""
    
    original_text: str = Field(default="", description="Original input text")
    claims: List[Claim] = Field(default_factory=list, description="Extracted claims")
    current_claim_index: int = Field(default=0, description="Index of claim being processed")
    search_queries: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Search queries per claim ID"
    )
    evidence_buffer: Dict[str, List[Evidence]] = Field(
        default_factory=dict, 
        description="Evidence collected per claim ID"
    )
    verification_results: List[VerificationResult] = Field(
        default_factory=list, 
        description="Completed verification results"
    )
    iteration_counts: Dict[str, int] = Field(
        default_factory=dict, 
        description="Iteration count per claim ID"
    )
    final_report: Optional[FinalReport] = Field(
        default=None, 
        description="Final verification report"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")
    
    class Config:
        arbitrary_types_allowed = True


# API Request/Response Models

class VerifyRequest(BaseModel):
    """Request model for verification endpoint."""
    
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=10000,
        description="The text paragraph to verify"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The Eiffel Tower is 330 meters tall and was completed in 1889."
            }
        }


class JobStatus(str, Enum):
    """Status of an async verification job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VerifyResponse(BaseModel):
    """Response model for verification endpoint."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    report: Optional[FinalReport] = Field(
        default=None, 
        description="Verification report (when completed)"
    )
    message: Optional[str] = Field(default=None, description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-001",
                "status": "completed",
                "report": None,
                "message": "Verification completed successfully"
            }
        }


class StreamEvent(BaseModel):
    """Event model for SSE streaming."""
    
    event_type: str = Field(..., description="Type of event")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
