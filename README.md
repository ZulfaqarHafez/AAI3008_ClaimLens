# ClaimLens

**An agentic fact-checking pipeline powered by LangGraph, fine-tuned DeBERTa-v3 NLI, and a real-time Next.js frontend.**

ClaimLens automatically decomposes any paragraph of text into atomic, independently verifiable claims, retrieves web evidence for each claim through targeted search queries, and verifies each against retrieved evidence using a custom-trained Natural Language Inference model — returning a structured trust report with confidence scores and source citations.

> **Project for AAI3008 Large Language Models — Singapore Institute of Technology**

---

## Team

| Name | LinkedIn |
|------|----------|
| Zulfaqar Hafez | [linkedin.com/in/zulfaqar-hafez](https://www.linkedin.com/in/zulfaqar-hafez/) |
| Genisa Lee | [linkedin.com/in/genisa-lee](https://www.linkedin.com/in/genisa-lee/) |
| Tay Wei Lin | [linkedin.com/in/tayweilin](https://www.linkedin.com/in/tayweilin/) |
| Gallant Teo | [linkedin.com/in/gallant-teo-2ab291186](https://www.linkedin.com/in/gallant-teo-2ab291186/) |
| Neo Jun Wei | [linkedin.com/in/neojunwei](https://www.linkedin.com/in/neojunwei/) |

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [NLI Verification Model](#nli-verification-model)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Models](#data-models)
- [Trust Score Calculation](#trust-score-calculation)
- [Tech Stack](#tech-stack)

---

## Overview

Modern information environments make it increasingly difficult to assess the accuracy of written claims. ClaimLens addresses this by providing an end-to-end automated pipeline that:

1. **Decomposes** unstructured text into discrete, atomic claims using a language model
2. **Enriches** each claim with contextual metadata — entity aliases, temporal cues, event frames — for more precise retrieval
3. **Retrieves** relevant web evidence using targeted, LLM-generated search queries executed in parallel
4. **Verifies** each claim against retrieved evidence using a fine-tuned DeBERTa-v3 NLI model with LLM cross-checking for uncertain cases
5. **Aggregates** individual verdicts into a final trust report with an overall confidence score

The system is designed as a **multi-agent workflow** orchestrated by LangGraph, enabling conditional branching, iterative search retries, and isolated failure handling per claim.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             ClaimLens Pipeline                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────┐     ┌──────────────┐     ┌──────────────┐                  │
│   │   Input    │────▶│ Decomposition│────▶│   Context    │                  │
│   │   Text     │     │    Agent     │     │    Agent     │                  │
│   └────────────┘     └──────────────┘     └──────────────┘                  │
│                             │                     │                          │
│                             ▼                     ▼                          │
│                      ┌──────────┐         ┌──────────────┐                  │
│                      │  Atomic  │         │  Enriched    │                  │
│                      │  Claims  │         │  Context +   │                  │
│                      └──────────┘         │  EventFrames │                  │
│                                           └──────────────┘                  │
│                                                  │                           │
│                                                  ▼                           │
│   ┌────────────────┐     ┌──────────────┐  ┌──────────────┐                 │
│   │   Credibility  │◀────│   Scraper    │◀─│    Search    │                 │
│   │     Agent      │     │    Agent     │  │  Architect   │                 │
│   └────────────────┘     └──────────────┘  └──────────────┘                 │
│           │                     │                                            │
│           ▼                     ▼                                            │
│   ┌────────────────┐     ┌──────────────┐                                   │
│   │  Scored        │     │  Filtered    │                                   │
│   │  Evidence      │     │  Evidence    │                                   │
│   └────────────────┘     └──────────────┘                                   │
│           │                     │                                            │
│           └──────────┬──────────┘                                            │
│                      ▼                                                       │
│             ┌─────────────────────────────────────┐                         │
│             │         Verifier Agent               │                         │
│             │   ClaimLens DeBERTa-v3 NLI Model     │                         │
│             │   + Multi-Layer Post-Verification    │                         │
│             │     Gatings + LLM Cross-Check        │                         │
│             └─────────────────────────────────────┘                         │
│                      │                                                       │
│          ┌───────────┴───────────┐                                           │
│          │ confidence < 0.7?     │                                           │
│          ▼                       ▼                                           │
│   ┌──────────────┐      ┌──────────────────┐                                │
│   │ Retry Search │      │  Finalize Claim  │                                │
│   │ (max 3 iter) │      └──────────────────┘                                │
│   └──────────────┘               │                                          │
│                                  ▼                                          │
│                        ┌──────────────────┐                                 │
│                        │  Final Report    │                                  │
│                        │  Trust Score     │                                  │
│                        │  Verdict Summary │                                  │
│                        └──────────────────┘                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### LangGraph State Flow

```
START
  │
  ▼
decompose_claims ──── (no claims extracted) ───▶ generate_report ───▶ END
  │
  ▼
prepare_claim
  │
  ▼
enrich_context          ← adds entity aliases, temporal/venue cues
  │
  ▼
frame_claim             ← extract {person, action, location, time, context}
  │
  ▼
generate_queries        ← 2–5 targeted search queries
  │
  ▼
search_evidence         ← parallel Tavily/SerpAPI queries
  │
  ▼
frame_evidence          ← align evidence event frames to claim frame
  │
  ▼
assess_credibility      ← score each source (expertise, recency, bias)
  │
  ▼
verify_claim            ← DeBERTa-v3 NLI + post-verification gatings
  │
  ├── (confidence < 0.7, iterations < 3) ───▶ generate_queries  [retry loop]
  │
  ▼
finalize_claim
  │
  ├── (more claims remaining) ───▶ prepare_claim
  │
  ▼
aggregate_results
  │
  ▼
generate_report
  │
  ▼
END
```

---

## Agent Pipeline

ClaimLens uses seven specialized agents, each with a clearly defined responsibility. Agents are modular and can accept injected dependencies, making them individually testable and swappable.

### 1. Decomposition Agent

**File**: `claimlens/agents/decomposition.py`

Breaks raw input text into atomic, independently verifiable claims using GPT-4o-mini with structured output. Each claim is a single factual statement that can be checked in isolation.

- Removes compound claims (e.g., "X happened AND Y is true")
- Filters subjective language ("I believe", "arguably")
- Validates claim length (10–500 characters)
- Preserves the original `source_sentence` for traceability
- Falls back gracefully if the LLM returns malformed output

```
Input:  "The Eiffel Tower is 330 meters tall and was built in 1889."

Output:
  Claim 1: "The Eiffel Tower is 330 meters tall"
  Claim 2: "The Eiffel Tower was built in 1889"
```

---

### 2. Context Enrichment Agent

**File**: `claimlens/agents/context.py`

Enriches each claim with structured metadata to improve downstream search and verification accuracy.

Outputs a `ClaimContext` object containing:

| Field | Description |
|-------|-------------|
| `normalized_claim` | Standardized claim text |
| `enriched_claim_text` | Self-contained version with contextual details |
| `context_summary` | Brief background on the claim topic |
| `temporal_context` | Dates or time periods referenced |
| `venue_context` | Institutions or locations referenced |
| `entity_aliases` | Alternative names (e.g., "DPM" → "Deputy Prime Minister") |
| `search_hints` | Related phrases and synonyms for search |
| `context_notes` | Annotated entity list with confidence scores |
| `event_frame` | Structured event representation (see Event Frame Agent) |

---

### 3. Event Frame Agent

**File**: `claimlens/agents/event_frame.py`

Extracts a structured event frame from both claims and evidence to enable precise matching. This prevents false positives where evidence discusses a related but distinct event.

```python
EventFrame:
  person:   "Gan Kim Yong"
  action:   "announced"
  location: "Parliament of Singapore"
  time:     "March 2025"
  context:  "Committee of Supply debate"
```

The agent compares claim and evidence frames across four dimensions (person, action, location, time) and returns a match verdict: `match` | `partial` | `contradict` | `insufficient`.

---

### 4. Search Architect Agent

**File**: `claimlens/agents/search_architect.py`

Generates 2–5 targeted search queries per claim, using the enriched context and event frame to improve retrieval precision.

- Produces diverse query formulations to increase recall
- Context-aware mode uses enriched claim text and entity aliases
- Retry mode generates different queries if initial evidence is insufficient
- Falls back to simple keyword queries if LLM output is malformed

```
For claim: "The Eiffel Tower is 330 meters tall"

Queries:
  1. "Eiffel Tower height 330 meters"
  2. "Eiffel Tower dimensions specifications"
  3. "Gustave Eiffel tower 330m official height"
```

---

### 5. Scraper Agent

**File**: `claimlens/agents/scraper.py`

Executes generated search queries in parallel using Tavily or SerpAPI, retrieves evidence snippets, and filters for relevance.

- Executes up to 5 queries concurrently via `ThreadPoolExecutor`
- Deduplicates evidence by URL across queries
- Uses GPT-4o-mini to score each evidence piece for relevance (0.0–1.0)
- Retains the top 5 most relevant evidence pieces (configurable)
- Preserves source URL and domain metadata for credibility assessment

---

### 6. Credibility Assessment Agent

**File**: `claimlens/agents/credibility.py`

Evaluates the trustworthiness of each evidence source across three dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Author Expertise | 40% | Domain authority of the source (expert publication → anonymous blog) |
| Recency | 20% | Publication freshness (< 6 months → > 5 years old) |
| Bias Score | 40% | Objectivity assessment (balanced journalism → propaganda) |

Returns a `credibility_score` (0.0–1.0) and `source_quality` label (`high`, `medium`, `low`) for each evidence piece, which is incorporated into the final NLI weighted voting.

---

### 7. Verifier Agent

**File**: `claimlens/agents/verifier.py`

The core verification component. Combines DeBERTa-v3 NLI scoring with multi-layer post-verification gatings and selective LLM cross-checking.

#### Verdict Types

| Verdict | Meaning |
|---------|---------|
| `SUPPORTED` | Evidence confirms the claim |
| `REFUTED` | Evidence contradicts the claim |
| `NOT_ENOUGH_INFO` | Evidence is inconclusive or insufficient |

#### Iterative Search Refinement

- If confidence < 0.7, the verifier requests new search queries (up to 3 iterations)
- `get_evidence_gap()` describes what evidence is missing for the retry
- Early stopping if verdict is clear (confidence > 0.5) and not `NOT_ENOUGH_INFO`

#### Post-Verification Gatings

Three independent checks applied after the NLI decision to filter false positives:

1. **Event Frame Gating** — Claim and evidence event frames must describe the same event across person, action, location, and time. Mismatches downgrade or block a `SUPPORTED` verdict.

2. **Direct Match Gating** — The evidence snippet must contain key claim keywords. Prevents the NLI model from inferring support based on topically related but factually irrelevant text.

3. **Cross-Source Agreement** — At least two distinct source domains must agree on the verdict. Prevents single-source reliance from generating a high-confidence result.

---

## NLI Verification Model

**Model**: [`Zulfhagez/claimlens-deberta-v3-nli`](https://huggingface.co/Zulfhagez/claimlens-deberta-v3-nli) (HuggingFace Hub)

A fine-tuned DeBERTa-v3 model trained specifically for fact-checking NLI, offering superior performance on factual claims compared to general-purpose zero-shot models like `facebook/bart-large-mnli`.

| Label ID | Verdict |
|----------|---------|
| 0 | SUPPORTED |
| 1 | REFUTED |
| 2 | NOT_ENOUGH_INFO |

### Weighted NLI Voting

Rather than a single inference pass, ClaimLens performs NLI over all evidence pieces and aggregates using a weighted vote:

```
weighted_score = NLI_confidence × evidence_relevance_score × credibility_score
final_verdict  = argmax(sum of weighted_scores per label)
```

### LLM Cross-Check Triggers

In cases where the NLI model is uncertain, GPT-4o-mini is called as a second opinion:

| Trigger | Condition | Action |
|---------|-----------|--------|
| A | REFUTED verdict + high relevance (≥ 0.65) | LLM second opinion |
| B | NOT_ENOUGH_INFO + high relevance (≥ 0.60) | LLM second opinion |
| C | Low NLI confidence (< 0.55) | LLM second opinion |

The LLM override is applied only if its confidence is ≥ 0.70. Critically, cross-checking always uses the **original claim text** (not the enriched version) to prevent context contamination.

### Alternative Verifier Backends

The verifier is pluggable via the `VERIFIER_TYPE` config variable:

| Backend | Model | Notes |
|---------|-------|-------|
| `claimlens` (default) | DeBERTa-v3 fine-tuned | Best performance; requires GPU recommended |
| `huggingface` | `facebook/bart-large-mnli` | Zero-shot; no fine-tuning |
| `openai` | GPT-4o-mini | No local model; pure LLM-based |

---

## API Reference

The backend exposes a FastAPI application with REST and streaming endpoints.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/verify` | Synchronous verification — returns full report |
| `POST` | `/verify/stream` | SSE streaming — real-time per-claim updates |
| `POST` | `/verify/async` | Async job submission — returns `job_id` |
| `GET` | `/verify/{job_id}` | Poll async job status and results |
| `DELETE` | `/verify/{job_id}` | Cancel or delete an async job |
| `POST` | `/decompose` | Extract claims only (no verification) |
| `GET` | `/health` | Backend health check |
| `GET` | `/config` | Active configuration (non-sensitive fields) |
| `GET` | `/jobs` | List all tracked jobs |

### Synchronous Verification

```bash
POST /verify
Content-Type: application/json

{
  "text": "The Great Wall of China is visible from space with the naked eye."
}
```

Returns a `FinalReport` with verdict, confidence, and evidence for each claim.

### SSE Streaming

```bash
POST /verify/stream
Content-Type: application/json

{ "text": "..." }
```

Returns Server-Sent Events as processing progresses:

```
event: start
data: {"event_type":"start","data":{"message":"Verification started"}}

event: claims_extracted
data: {"event_type":"claims_extracted","data":{"claims":[...],"count":2}}

event: claim_verified
data: {"event_type":"claim_verified","data":{"claim_id":"...","verdict":"SUPPORTED","confidence":0.94}}

event: complete
data: {"event_type":"complete","data":{"trust_score":0.92,"summary":"...","report":{...}}}
```

### Async Job

```bash
# Submit
POST /verify/async
{ "text": "..." }
→ { "job_id": "uuid", "status": "pending" }

# Poll
GET /verify/{job_id}
→ { "status": "completed", "report": {...} }
```

### Security

- **API Key Authentication**: Optional `X-API-Key` header, configurable per environment
- **Rate Limiting**: Per-client-IP throttling (Redis-backed or in-memory)
- **CORS**: Configurable allowed origins
- **Input Validation**: Max 10,000 characters; sanitized error messages in production

---

## Frontend

The frontend is a Next.js 16 application with React 19 and Tailwind CSS 4, providing a real-time verification interface.

### Pages

**Landing Page** (`/`)
- Hero section with project overview and key metrics
- Visual pipeline diagram showing the verification flow
- Feature highlights and how-it-works walkthrough
- Call-to-action to begin verification

**Verification Page** (`/verify`)
- Text input with live character counter (max 10,000)
- Real-time pipeline visualizer showing active graph node
- Per-claim progress tracking with status indicators
- Detailed results view with verdicts, confidence scores, and evidence sources

### Real-Time Streaming

The frontend uses the native `EventSource`/`fetch` SSE API to receive incremental updates:

- Pipeline stage transitions update the visual graph in real time
- Claims appear as they are extracted
- Each claim verdict updates independently as verification completes
- AbortController supports mid-stream cancellation

### State Management

A React Context (`VerificationContext`) manages global verification state:

```typescript
type Phase = "input" | "loading" | "results"

interface VerificationState {
  text: string
  phase: Phase
  claims: ExtractedClaim[]
  verified: VerifiedClaim[]
  report: FinalReport | null
  error: string | null
  progressMsg: string
  currentNode: PipelineNode | null
  completedNodes: PipelineNode[]
}
```

### API Proxy

`next.config.ts` rewrites `/api/*` to the backend URL via `API_BACKEND_URL`, avoiding CORS issues in development and enabling seamless deployment behind a single domain in production.

---

## Project Structure

```
AAI3008_ClaimLens/
├── claimlens/                          # Python backend
│   ├── config.py                       # Pydantic-settings configuration
│   ├── storage.py                      # Redis + PostgreSQL integration
│   ├── agents/
│   │   ├── decomposition.py            # Claim extraction (GPT-4o-mini)
│   │   ├── context.py                  # Context enrichment & entity aliasing
│   │   ├── event_frame.py              # Structured event frame extraction
│   │   ├── search_architect.py         # Search query generation
│   │   ├── scraper.py                  # Parallel evidence retrieval (Tavily)
│   │   ├── credibility.py              # Source credibility scoring
│   │   └── verifier.py                 # NLI verification + post-gatings
│   ├── models/
│   │   ├── schemas.py                  # Pydantic data models
│   │   └── nli_placeholder.py          # Verifier implementations (ClaimLens, HF, OpenAI)
│   ├── graph/
│   │   └── orchestrator.py             # LangGraph state machine
│   ├── services/
│   │   ├── llm_service.py              # OpenAI API wrapper
│   │   └── search_service.py           # Search API abstraction
│   └── api/
│       └── main.py                     # FastAPI app (REST + SSE)
│
├── claimlens-ui/                       # Next.js 16 frontend
│   └── src/
│       ├── app/
│       │   ├── page.tsx                # Landing page
│       │   ├── verify/page.tsx         # Verification interface
│       │   └── layout.tsx              # Root layout (Navbar + Footer)
│       ├── components/
│       │   ├── landing/                # Hero, Architecture, Features, HowItWorks, CTA
│       │   ├── verify/                 # VerifyPage, PipelineVisualizer, ProgressTracker, ResultsView
│       │   └── layout/                 # Navbar, Footer
│       ├── context/
│       │   └── VerificationContext.tsx # Global verification state
│       ├── hooks/
│       │   └── useVerification.ts      # Context hook
│       ├── lib/
│       │   └── api.ts                  # Fetch + SSE client
│       ├── types/
│       │   └── api.ts                  # TypeScript types (Claim, Evidence, FinalReport, etc.)
│       └── constants/
│           ├── verdicts.ts             # Verdict colors and labels
│           └── validation.ts           # Input limits
│
├── examples/
│   ├── run_verification.py             # Standalone CLI demo
│   └── custom_verifier.py              # Custom verifier implementation guide
│
├── tests/
│   └── test_pipeline.py                # Unit tests
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key
- Tavily API key

### Installation

```bash
# Clone the repository
git clone https://github.com/ZulfaqarHafez/AAI3008_ClaimLens.git
cd AAI3008_ClaimLens

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd claimlens-ui
npm install
cd ..
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
VERIFIER_TYPE=claimlens        # claimlens | huggingface | openai
```

### Running

```bash
# Terminal 1 — FastAPI backend
uvicorn claimlens.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Next.js frontend
cd claimlens-ui
npm run dev
```

- **Frontend**: http://localhost:3000
- **API docs (Swagger)**: http://localhost:8000/docs
- **API docs (ReDoc)**: http://localhost:8000/redoc

### Running Tests

```bash
pytest tests/ -v

# With coverage report
pytest tests/ --cov=claimlens --cov-report=html
```

---

## Configuration

All settings are managed via environment variables (`.env`) and Pydantic-settings.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `TAVILY_API_KEY` | — | Tavily search key (required if using Tavily) |
| `SERPAPI_KEY` | — | SerpAPI key (alternative to Tavily) |
| `LLM_MODEL` | `gpt-4o` | OpenAI model for agents |
| `LLM_TEMPERATURE` | `0.1` | Temperature for LLM responses |
| `VERIFIER_TYPE` | `claimlens` | NLI backend: `claimlens`, `huggingface`, `openai` |
| `HF_NLI_MODEL` | `facebook/bart-large-mnli` | HuggingFace model (if `VERIFIER_TYPE=huggingface`) |
| `SEARCH_PROVIDER` | `tavily` | Search backend: `tavily`, `serpapi` |
| `MAX_VERIFICATION_ITERATIONS` | `3` | Max search retry iterations per claim |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence before accepting verdict |
| `MAX_EVIDENCE_PER_CLAIM` | `5` | Maximum evidence pieces per claim |
| `SEARCH_RESULTS_PER_QUERY` | `5` | Results returned per search query |
| `API_HOST` | `0.0.0.0` | Backend host |
| `API_PORT` | `8000` | Backend port |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `API_KEY` | — | Optional API key for authentication |
| `RATE_LIMIT_REQUESTS` | — | Requests per minute per IP |

---

## Data Models

### Claim

```json
{
  "id": "uuid",
  "text": "The Eiffel Tower is 330 meters tall",
  "source_sentence": "The Eiffel Tower is 330 meters tall and was built in 1889.",
  "status": "completed",
  "context": { /* ClaimContext */ }
}
```

### Evidence

```json
{
  "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
  "title": "Eiffel Tower — Wikipedia",
  "snippet": "The Eiffel Tower is 330 metres (1,083 ft) tall...",
  "relevance_score": 0.96,
  "credibility_score": 0.91,
  "source_quality": "high",
  "event_frame": { /* EventFrame */ }
}
```

### VerificationResult

```json
{
  "claim": { /* Claim */ },
  "evidence_list": [ /* Evidence[] */ ],
  "verdict": "SUPPORTED",
  "confidence": 0.94,
  "reasoning": "Multiple authoritative sources confirm the 330-metre height.",
  "iterations_used": 1
}
```

### FinalReport

```json
{
  "id": "uuid",
  "original_text": "...",
  "claims": [ /* Claim[] */ ],
  "verification_results": [ /* VerificationResult[] */ ],
  "overall_trust_score": 0.92,
  "summary": "2 of 2 claims supported. High confidence in submitted information.",
  "processing_time_seconds": 14.3
}
```

---

## Trust Score Calculation

```
overall_trust_score = weighted average of:
  claim_support_ratio   (50%)  — percentage of claims with SUPPORTED verdict
                                 (REFUTED claims are penalised)
  average_confidence    (30%)  — mean confidence across all verdicts
  evidence_quality      (20%)  — derived from average credibility scores
```

### Confidence Levels

| Range | Level | Interpretation |
|-------|-------|----------------|
| 0.90 – 1.00 | Very High | Multiple strong, independent sources align |
| 0.70 – 0.90 | High | Primary authoritative sources support the verdict |
| 0.50 – 0.70 | Moderate | Some sources align; some ambiguity present |
| < 0.50 | Low | Contradictory evidence or insufficient information |

---

## Tech Stack

### Backend

| Category | Technology |
|----------|-----------|
| API framework | FastAPI 0.109+ |
| Agent orchestration | LangGraph 0.2+ |
| LLM integration | LangChain + OpenAI SDK |
| NLI model | DeBERTa-v3 (fine-tuned via HuggingFace Transformers) |
| ML inference | PyTorch 2.1+ |
| Web search | Tavily API / SerpAPI |
| Async HTTP | HTTPX, aiohttp |
| Job storage | Redis (TTL-based caching) |
| Report persistence | PostgreSQL (JSONB) |
| Configuration | Pydantic-settings |

### Frontend

| Category | Technology |
|----------|-----------|
| Framework | Next.js 16 (App Router) |
| UI library | React 19 (with React Compiler) |
| Styling | Tailwind CSS 4 |
| Language | TypeScript 5 |
| Icons | Lucide React |
| Streaming | Native SSE via Fetch API |

---

## Roadmap

- [x] Fine-tuned DeBERTa-v3 NLI model (`claimlens-deberta-v3-nli`)
- [x] Next.js frontend with real-time SSE streaming
- [x] Context enrichment and event frame extraction
- [x] Multi-layer post-verification gatings
- [x] Source credibility scoring
- [x] Iterative search refinement with retry logic
- [x] Async job management with Redis
- [ ] Redis-backed job persistence in production
- [ ] Claim deduplication across pipeline runs
- [ ] Multi-language support
- [ ] Batch text processing endpoint
- [ ] Claim provenance and source citation export
- [ ] User accounts and verification history dashboard

---

## License

MIT License — see [LICENSE](LICENSE) for details.
