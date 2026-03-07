# ClaimLens 🔍

An agentic fact-checking pipeline using LangGraph that decomposes user-provided paragraphs into atomic claims and verifies each claim against web evidence using a fine-tuned DeBERTa-v3 NLI model.

> **Project for AAI3008 Large Language Model module**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClaimLens Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │  Input   │───▶│ Decompose│───▶│  Search  │───▶│ Scraper  │     │
│  │  Text    │    │  Agent   │    │ Architect│    │  Agent   │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                        │               │               │           │
│                        ▼               ▼               ▼           │
│                  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│                  │  Claims  │    │ Queries  │    │ Evidence │     │
│                  └──────────┘    └──────────┘    └──────────┘     │
│                                                        │           │
│                                                        ▼           │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐     │
│  │  Final   │◀───│Aggregate │◀───│     Verifier Agent       │     │
│  │  Report  │    │ Results  │    │  (ClaimLens DeBERTa NLI) │     │
│  └──────────┘    └──────────┘    └──────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- OpenAI API key
- Tavily API key

### Installation

```bash
# Clone the repository
cd AAI3008_ClaimLens

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
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
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
# VERIFIER_TYPE=claimlens
```

### Running

```bash
# Terminal 1 — Start the FastAPI backend
uvicorn claimlens.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Start the Next.js frontend
cd claimlens-ui
npm run dev
```

- Backend API docs: http://localhost:8000/docs
- Frontend UI: http://localhost:3000

## 📁 Project Structure

```
AAI3008_ClaimLens/
├── claimlens/                        # Python backend
│   ├── __init__.py
│   ├── config.py                     # Settings (env vars, defaults)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── decomposition.py          # Breaks text into atomic claims (GPT-4o mini)
│   │   ├── search_architect.py       # Generates search queries per claim
│   │   ├── scraper.py                # Retrieves and filters web evidence (Tavily)
│   │   └── verifier.py               # Verifies claims against evidence
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                # Pydantic data models (Claim, Evidence, etc.)
│   │   └── nli_placeholder.py        # NLI verifier implementations (ClaimLens DeBERTa, HF, OpenAI)
│   ├── graph/
│   │   ├── __init__.py
│   │   └── orchestrator.py           # LangGraph state machine orchestrator
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py            # OpenAI API wrapper
│   │   └── search_service.py         # Web search API wrapper (Tavily/SerpAPI)
│   └── api/
│       ├── __init__.py
│       └── main.py                   # FastAPI endpoints (sync, async, SSE streaming)
│
├── claimlens-ui/                     # Next.js frontend (TypeScript + Tailwind)
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx            # Root layout with Navbar + Footer
│   │   │   ├── page.tsx              # Landing page
│   │   │   ├── error.tsx             # Global error boundary
│   │   │   ├── not-found.tsx         # 404 page
│   │   │   ├── loading.tsx           # Root loading skeleton
│   │   │   ├── globals.css           # Tailwind v4 theme
│   │   │   └── verify/
│   │   │       ├── page.tsx          # Verification page
│   │   │       └── loading.tsx       # Verify route loading
│   │   ├── components/
│   │   │   ├── landing/              # Landing page sections
│   │   │   │   ├── Hero.tsx
│   │   │   │   ├── HowItWorks.tsx
│   │   │   │   ├── Features.tsx
│   │   │   │   └── CTA.tsx
│   │   │   ├── layout/               # Layout components
│   │   │   │   ├── Navbar.tsx
│   │   │   │   └── Footer.tsx
│   │   │   └── verify/               # Verification components
│   │   │       ├── VerifyPage.tsx
│   │   │       ├── ProgressTracker.tsx
│   │   │       └── ResultsView.tsx
│   │   ├── constants/                # Shared constants
│   │   │   ├── verdicts.ts           # Verdict colors, icons, labels
│   │   │   └── validation.ts         # Input validation limits
│   │   ├── hooks/                    # Custom React hooks
│   │   │   └── useVerification.ts    # Verification state + streaming logic
│   │   ├── lib/
│   │   │   └── api.ts               # API client with SSE + AbortController
│   │   └── types/
│   │       ├── api.ts               # TypeScript types matching backend schemas
│   │       └── index.ts             # Barrel export
│   ├── .env.local                    # Frontend env vars (API_BACKEND_URL)
│   ├── next.config.ts                # API proxy (rewrites /api/* → backend)
│   ├── package.json
│   └── tsconfig.json
│
├── examples/
│   ├── run_verification.py           # Standalone demo script
│   └── custom_verifier.py            # Custom verifier implementation example
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # Unit tests for the pipeline
│
├── .env.example                      # Environment variable template
├── .gitignore
├── requirements.txt                  # Python dependencies
└── README.md
```

## 🔌 API Endpoints

### Synchronous Verification

```bash
POST /verify
Content-Type: application/json

{
  "text": "The Eiffel Tower is 330 meters tall and was built in 1889."
}
```

### Async Verification

```bash
# Start verification job
POST /verify/async
Content-Type: application/json

{
  "text": "Your text here..."
}

# Returns: { "job_id": "uuid", "status": "pending" }

# Check status
GET /verify/{job_id}
```

### Streaming Verification (SSE)

```bash
POST /verify/stream
Content-Type: application/json

{
  "text": "Your text here..."
}

# Returns Server-Sent Events with progress updates
```

### Utility Endpoints

```bash
# Health check
GET /health

# Get configuration
GET /config

# Decompose text into claims (without verification)
POST /decompose

# List all jobs
GET /jobs
```

## 🧪 Example Usage

### Python Client

```python
import requests

# Synchronous verification
response = requests.post(
    "http://localhost:8000/verify",
    json={"text": "The Great Wall of China is visible from space."}
)

report = response.json()
print(f"Trust Score: {report['overall_trust_score']}")
print(f"Summary: {report['summary']}")

for result in report['verification_results']:
    print(f"\nClaim: {result['claim']['text']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}")
```

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | gpt-4o-mini | OpenAI model to use |
| `LLM_TEMPERATURE` | 0.1 | Temperature for LLM responses |
| `MAX_VERIFICATION_ITERATIONS` | 3 | Max search iterations per claim |
| `CONFIDENCE_THRESHOLD` | 0.7 | Threshold to stop searching |
| `MAX_EVIDENCE_PER_CLAIM` | 5 | Max evidence pieces per claim |
| `SEARCH_RESULTS_PER_QUERY` | 5 | Results per search query |
| `VERIFIER_TYPE` | claimlens | Verifier backend (`claimlens` / `huggingface` / `openai`) |
| `SEARCH_PROVIDER` | tavily | Search API (`tavily` / `serpapi`) |

## 🤖 ClaimLens DeBERTa NLI Model

The default verifier uses a fine-tuned DeBERTa-v3 model (`Zulfhagez/claimlens-deberta-v3-nli`) for 3-label natural language inference:

| Label ID | Verdict |
|----------|---------|
| 0 | SUPPORTED |
| 1 | REFUTED |
| 2 | NOT_ENOUGH_INFO |

The model is lazy-loaded on first use and runs weighted voting across all evidence pieces for each claim, combining NLI confidence with evidence relevance scores.

## 📊 Data Models

### Claim
```json
{
  "id": "uuid",
  "text": "The atomic claim text",
  "source_sentence": "Original sentence",
  "status": "pending | searching | verifying | completed | failed"
}
```

### VerificationResult
```json
{
  "claim": "Claim",
  "evidence_list": ["Evidence"],
  "verdict": "SUPPORTED | REFUTED | NOT_ENOUGH_INFO",
  "confidence": 0.92,
  "reasoning": "Explanation text",
  "iterations_used": 1
}
```

### FinalReport
```json
{
  "id": "uuid",
  "original_text": "Input text",
  "claims": ["Claim"],
  "verification_results": ["VerificationResult"],
  "overall_trust_score": 0.87,
  "summary": "Human-readable summary",
  "processing_time_seconds": 91.2
}
```

## 🧮 Trust Score Calculation

```
overall_trust_score = weighted_average of:
  - claim_support_ratio (50%): % of claims supported (penalizes refuted)
  - average_confidence (30%): Mean confidence across all verdicts
  - evidence_quality_score (20%): Based on source reliability
```

## 🛣️ LangGraph State Flow

```
START
  │
  ▼
decompose_claims
  │
  ├─── (no claims) ───▶ generate_report ───▶ END
  │
  ▼
prepare_claim
  │
  ▼
generate_queries
  │
  ▼
search_evidence
  │
  ▼
verify_claim
  │
  ├─── (low confidence) ───▶ generate_queries (loop)
  │
  ▼
finalize_claim
  │
  ├─── (more claims) ───▶ prepare_claim
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

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=claimlens --cov-report=html
```

## 📝 License

MIT License - See LICENSE file for details.

## 🔮 Future Improvements

- [x] Integrate custom DeBERTa-v3 NLI model
- [x] Next.js frontend with streaming verification
- [ ] Add Redis for job persistence
- [ ] Add claim deduplication
- [ ] Support for multiple languages
- [ ] Batch processing endpoint
- [ ] Source credibility scoring
- [ ] Claim provenance tracking
