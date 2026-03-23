"""FastAPI main application for ClaimLens verification service."""

import asyncio
import json
import logging
import queue as queue_module
import uuid
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from collections import defaultdict
import time as time_module

from ..config import settings
from ..models.schemas import (
    VerifyRequest,
    VerifyResponse,
    FinalReport,
    JobStatus,
    StreamEvent,
    Verdict,
)
from ..graph.orchestrator import ClaimLensGraph, create_graph
from ..storage import (
    init_redis,
    init_postgres,
    create_tables,
    save_report,
    save_job,
    get_job,
    delete_job,
    check_rate_limit,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# In-memory job storage (use Redis or database in production)
jobs: Dict[str, Dict] = {}

# Rate limiting storage
rate_limit_storage: Dict[str, list] = defaultdict(list)

# Graph instance (singleton)
_graph_instance: Optional[ClaimLensGraph] = None

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_graph() -> ClaimLensGraph:
    """Get or create the graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_graph()
    return _graph_instance


def cleanup_old_jobs():
    """Remove old jobs if storage exceeds limit."""
    if len(jobs) > settings.MAX_JOBS_STORED:
        # Sort by created_at and remove oldest
        sorted_jobs = sorted(
            jobs.items(),
            key=lambda x: x[1].get("created_at", ""),
        )
        # Remove oldest 10% of jobs
        to_remove = len(jobs) - int(settings.MAX_JOBS_STORED * 0.9)
        for job_id, _ in sorted_jobs[:to_remove]:
            del jobs[job_id]
        logger.info(f"Cleaned up {to_remove} old jobs")


async def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Verify API key if authentication is enabled.
    
    Args:
        api_key: API key from request header
        
    Returns:
        True if valid or auth disabled
        
    Raises:
        HTTPException: If API key is invalid
    """
    # If no API key configured, allow all requests
    if not settings.API_KEY:
        return True
    
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API key required"}
        )
    return True


async def rate_limit_check(request: Request):
    client_ip = request.client.host if request.client else "unknown"

    try:
        allowed = check_rate_limit(
            client_ip,
            settings.RATE_LIMIT_REQUESTS,
            settings.RATE_LIMIT_WINDOW
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        return
    except HTTPException:
        raise
    except Exception:
        pass

    current_time = time_module.time()
    window_start = current_time - settings.RATE_LIMIT_WINDOW
    rate_limit_storage[client_ip] = [
        t for t in rate_limit_storage[client_ip] if t > window_start
    ]

    if len(rate_limit_storage[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    rate_limit_storage[client_ip].append(current_time)


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error message to prevent information leakage.
    
    Args:
        error: The exception that occurred
        
    Returns:
        Safe error message for clients
    """
    if settings.DEBUG_MODE:
        return str(error)
    
    # Generic error messages for production
    error_str = str(error).lower()
    
    if "api key" in error_str or "authentication" in error_str:
        return "Authentication error occurred"
    elif "timeout" in error_str:
        return "Request timed out"
    elif "connection" in error_str:
        return "Service temporarily unavailable"
    else:
        return "An internal error occurred"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ClaimLens API...")
    logger.info(f"Using LLM model: {settings.LLM_MODEL}")
    logger.info(f"Verifier type: {settings.VERIFIER_TYPE}")
    logger.info(f"Search provider: {settings.SEARCH_PROVIDER}")

    try:
        init_redis()
        init_postgres()
        create_tables()
        logger.info("Storage initialized")
    except Exception as e:
        logger.warning(f"Storage init failed, continuing without full persistence: {e}")

    yield

    logger.info("Shutting down ClaimLens API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="ClaimLens API",
        description="Agentic fact-checking pipeline for verifying claims in text",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware with configurable origins
    cors_origins = [
        origin.strip() 
        for origin in settings.CORS_ORIGINS.split(",") 
        if origin.strip()
    ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )
    
    return app


# Create app instance
app = create_app()


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0"
    }


@app.get("/config")
async def get_config(_: bool = Depends(verify_api_key)):
    """Get current configuration (non-sensitive)."""
    return {
        "llm_model": settings.LLM_MODEL,
        "verifier_type": settings.VERIFIER_TYPE,
        "search_provider": settings.SEARCH_PROVIDER,
        "max_verification_iterations": settings.MAX_VERIFICATION_ITERATIONS,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "max_evidence_per_claim": settings.MAX_EVIDENCE_PER_CLAIM,
    }


# ==================== Synchronous Verification ====================

@app.post("/verify", response_model=FinalReport)
async def verify_text(
    request: VerifyRequest,
    http_request: Request,
    _: bool = Depends(verify_api_key)
):
    """Verify claims in the provided text.
    
    This endpoint runs the full verification pipeline synchronously
    and returns the complete report when finished.
    
    Args:
        request: VerifyRequest with text to verify
        
    Returns:
        FinalReport with all verification results
    """
    # Rate limiting
    await rate_limit_check(http_request)
    
    # Input validation
    if len(request.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.MAX_TEXT_LENGTH} characters allowed."
        )
    
    logger.info(f"Received verification request: {request.text[:100]}...")
    
    try:
        graph = get_graph()
        report = await graph.arun(request.text)
        save_report(report)
        
        logger.info(
            f"Verification completed. Trust score: {report.overall_trust_score:.2f}"
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(e)
        )


# ==================== Async Verification with Job ID ====================

async def run_verification_job(job_id: str, text: str):
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        save_job(job_id, jobs[job_id])

        graph = get_graph()
        report = await graph.arun(text)

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["report"] = json.loads(report.model_dump_json())
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        save_job(job_id, jobs[job_id])
        save_report(report)

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = sanitize_error_message(e)
        save_job(job_id, jobs[job_id])


@app.post("/verify/async", response_model=VerifyResponse)
async def verify_text_async(
    request: VerifyRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key)
):
    """Start async verification and return job ID.
    
    Use GET /verify/{job_id} to check status and retrieve results.
    
    Args:
        request: VerifyRequest with text to verify
        
    Returns:
        VerifyResponse with job_id and initial status
    """
    # Rate limiting
    await rate_limit_check(http_request)
    
    # Input validation
    if len(request.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.MAX_TEXT_LENGTH} characters allowed."
        )
    
    # Cleanup old jobs
    cleanup_old_jobs()
    
    job_id = str(uuid.uuid4())
    
    # Initialize job (don't store raw text for security)
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "text_length": len(request.text),  # Store length instead of full text
        "report": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    save_job(job_id, jobs[job_id])
    
    # Start background task
    background_tasks.add_task(run_verification_job, job_id, request.text)
    
    logger.info(f"Created async job: {job_id}")
    
    return VerifyResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Verification job started"
    )


@app.get("/verify/{job_id}", response_model=VerifyResponse)
async def get_verification_status(job_id: str, _: bool = Depends(verify_api_key)):
    """Get the status of an async verification job.
    
    Args:
        job_id: The job identifier returned from POST /verify/async
        
    Returns:
        VerifyResponse with current status and report (if completed)
    """
    job = jobs.get(job_id)

    if job is None:
        job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )   
    
    message = None
    if job["status"] == JobStatus.PROCESSING:
        message = "Verification in progress..."
    elif job["status"] == JobStatus.COMPLETED:
        message = "Verification completed successfully"
    elif job["status"] == JobStatus.FAILED:
        message = f"Verification failed: {job.get('error', 'Unknown error')}"
    
    return VerifyResponse(
        job_id=job_id,
        status=job["status"],
        report=job.get("report"),
        message=message
    )


@app.delete("/verify/{job_id}")
async def delete_verification_job(job_id: str, _: bool = Depends(verify_api_key)):
    """Delete a verification job.
    
    Args:
        job_id: The job identifier to delete
        
    Returns:
        Confirmation message
    """
    if job_id in jobs:
        del jobs[job_id]

    delete_job(job_id)

    return {"message": "Job deleted successfully"}


# ==================== Streaming Verification ====================

async def generate_sse_events(text: str):
    """Generate Server-Sent Events for streaming verification.
    
    Runs the synchronous LangGraph pipeline in a thread to avoid blocking
    the async event loop, enabling real-time SSE flushing.
    
    Args:
        text: Text to verify
        
    Yields:
        SSE-formatted event strings
    """
    try:
        graph = get_graph()
        
        # Send start event (don't include text preview for security)
        start_event = StreamEvent(
            event_type="start",
            data={"message": "Verification started"}
        )
        yield f"event: start\ndata: {start_event.model_dump_json()}\n\n"
        
        # Bridge sync graph.stream() into async via thread + queue
        q: queue_module.Queue = queue_module.Queue()
        _SENTINEL = object()
        
        def _run_stream():
            try:
                for state_update in graph.stream(text):
                    q.put(state_update)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_SENTINEL)
        
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_stream)
        
        while True:
            # Non-blocking poll keeps the event loop free to flush SSE
            while q.empty():
                await asyncio.sleep(0.05)
            
            item = q.get_nowait()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            
            state_update = item
            
            # Determine event type from state keys
            node_name = list(state_update.keys())[0] if state_update else "unknown"
            state = state_update.get(node_name) or {}
            
            # Emit pipeline step event for every node
            step_event = StreamEvent(
                event_type="step",
                data={"node": node_name}
            )
            yield f"event: step\ndata: {step_event.model_dump_json()}\n\n"
            
            # Create appropriate event based on node
            if "claims" in state and state.get("claims"):
                claims_data = [
                    {"id": c.id, "text": c.text, "status": c.status.value}
                    for c in state["claims"]
                ]
                event = StreamEvent(
                    event_type="claims_extracted",
                    data={"claims": claims_data, "count": len(claims_data)}
                )
                yield f"event: claims_extracted\ndata: {event.model_dump_json()}\n\n"
            
            # Only emit claim_verified on finalize_claim (not verify_claim,
            # which can fire multiple times per claim during retries)
            if node_name == "finalize_claim" and "verification_results" in state:
                results_list = state["verification_results"]
                if results_list:
                    result = results_list[-1]  # latest finalized result
                    # Include evidence/sources so frontend can show them immediately
                    evidence_data = [
                        {
                            "url": e.url,
                            "title": e.title,
                            "snippet": e.snippet[:300],
                            "relevance_score": e.relevance_score,
                            "source_quality": e.source_quality,
                            "credibility_score": e.credibility_score,
                            "credibility_reasoning": e.credibility_reasoning,
                            "published_date": e.published_date,
                        }
                        for e in result.evidence_list
                    ]
                    event = StreamEvent(
                        event_type="claim_verified",
                        data={
                            "claim_id": result.claim.id,
                            "claim_text": result.claim.text,
                            "verdict": result.verdict.value,
                            "confidence": result.confidence,
                            "reasoning": result.reasoning,
                            "evidence": evidence_data,
                        }
                    )
                    yield f"event: claim_verified\ndata: {event.model_dump_json()}\n\n"
            
            if "final_report" in state and state.get("final_report"):
                report = state["final_report"]
                event = StreamEvent(
                    event_type="complete",
                    data={
                        "trust_score": report.overall_trust_score,
                        "summary": report.summary,
                        "total_claims": len(report.claims),
                        "report": json.loads(report.model_dump_json()),
                    }
                )
                yield f"event: complete\ndata: {event.model_dump_json()}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_event = StreamEvent(
            event_type="error",
            data={"error": sanitize_error_message(e)}
        )
        yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"


@app.post("/verify/stream")
async def verify_text_stream(
    request: VerifyRequest, 
    http_request: Request,
    _: bool = Depends(verify_api_key)
):
    """Stream verification progress using Server-Sent Events.
    
    This endpoint returns real-time updates as each claim is verified.
    
    Events:
    - start: Verification started
    - claims_extracted: Claims have been decomposed from text
    - claim_verified: A single claim has been verified
    - complete: Verification finished
    - error: An error occurred
    
    Args:
        request: VerifyRequest with text to verify
        
    Returns:
        StreamingResponse with SSE events
    """
    # Rate limiting
    await rate_limit_check(http_request)
    
    # Input validation
    if len(request.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.MAX_TEXT_LENGTH} characters allowed."
        )
    
    logger.info(f"Starting streaming verification (length: {len(request.text)})")
    
    return StreamingResponse(
        generate_sse_events(request.text),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ==================== Utility Endpoints ====================

class DecomposeRequest(BaseModel):
    """Request for claim decomposition only."""
    text: str


@app.post("/decompose")
async def decompose_text(
    request: DecomposeRequest,
    http_request: Request,
    _: bool = Depends(verify_api_key)
):
    """Decompose text into claims without verification.
    
    Useful for previewing claims before running full verification.
    
    Args:
        request: DecomposeRequest with text
        
    Returns:
        List of extracted claims
    """
    # Rate limiting
    await rate_limit_check(http_request)
    
    # Input validation
    if len(request.text) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.MAX_TEXT_LENGTH} characters allowed."
        )
    
    try:
        graph = get_graph()
        claims = graph.decomposition_agent.decompose(request.text)
        claims = graph.decomposition_agent.validate_claims(claims)
        
        return {
            "claims": [
                {
                    "id": c.id,
                    "text": c.text,
                    "source_sentence": c.source_sentence
                }
                for c in claims
            ],
            "count": len(claims)
        }
        
    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=sanitize_error_message(e)
        )


@app.get("/jobs")
async def list_jobs(_: bool = Depends(verify_api_key)):
    """List all verification jobs.
    
    Returns:
        List of jobs with their statuses (limited info for security)
    """
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"].value if isinstance(job["status"], JobStatus) else job["status"],
                "created_at": job["created_at"],
                "completed_at": job.get("completed_at"),
            }
            for job_id, job in jobs.items()
        ],
        "total": len(jobs)
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "claimlens.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
