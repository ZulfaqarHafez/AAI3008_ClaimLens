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

## ☁️ AWS Deployment and Persistent Storage

ClaimLens is deployed on AWS with the following infrastructure:

- **Amazon EC2** for hosting both the frontend and FastAPI backend
- **Amazon RDS PostgreSQL** for persistent storage of verification reports
- **Amazon ElastiCache Redis** for asynchronous job and state storage
- **Elastic IP** for a stable public address

### Application Access

- **Frontend:** runs on port `3000`
- **Backend API:** runs on port `8000`

### Persistent Storage

Persistent storage was implemented and validated end-to-end.

- Verification results are stored durably in **PostgreSQL**
- Async verification job state is managed through the backend storage layer with **Redis**
- Successful verification runs can be confirmed in the `verification_reports` table in RDS

### Running the Frontend on EC2

```bash
cd claimlens-ui
npm run build
npm run start -- --hostname 0.0.0.0 --port 3000
```

### Running the Backend on EC2

```bash
uvicorn claimlens.api.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

## 🎯 How the App Works

### User Flow

ClaimLens is an **agentic fact-checking system** that uses multiple AI agents working together to verify claims automatically. Here's what happens when a user submits text:

#### Step 1: Landing Page
Users arrive at the landing page which explains the concept:
- Displays hero section with key metrics (trust score breakdown)
- Shows the architecture pipeline with visual flow of how claims are decomposed → searched → verified
- Lists key features: atomic claim extraction, multi-evidence gathering, credibility scoring
- Includes call-to-action to verify claims

#### Step 2: User Submits Text
User navigates to the Verify page and enters text they want fact-checked:

```
Input Example:
"The Eiffel Tower is 330 meters tall and was built in 1889. It attracts over 7 million visitors annually."
```

The app validates that:
- Text is not empty
- Text is under 10,000 characters
- No special filtering applied (all languages welcome)

#### Step 3: Decomposition Agent
The **Decomposition Agent** breaks the input into atomic, verifiable claims:

```python
# Agent: decomposition.py
# Model: GPT-4o mini
# Task: Extract atomic claims from unstructured text

Input: "The Eiffel Tower is 330 meters tall and was built in 1889."

Output Claims:
├─ Claim 1: "The Eiffel Tower is 330 meters tall"
├─ Claim 2: "The Eiffel Tower was built in 1889"
└─ Metadata: source_sentence, claim_id, status (pending)
```

The agents ensures:
- Each claim is independently verifiable
- No compound claims (e.g., "X is true AND Y happened")
- Claims retain original context via source_sentence reference

#### Step 4: Search Architect Agent
For each claim, the **Search Architect Agent** generates targeted search queries:

```python
# Agent: search_architect.py
# Model: GPT-4o mini
# Task: Generate search queries that will find relevant evidence

For Claim: "The Eiffel Tower is 330 meters tall"

Queries Generated:
├─ Query 1: "Eiffel Tower height meters"
├─ Query 2: "Eiffel Tower dimensions specifications"
└─ Query 3: "Gustave Eiffel tower 330m"

Reasoning: Multiple formulations increase chance of finding reliable sources
```

#### Step 5: Scraper Agent
The **Scraper Agent** executes generated queries and retrieves web evidence:

```python
# Agent: scraper.py
# Service: Tavily Web Search API
# Task: Find and extract relevant evidence from web sources

For Query: "Eiffel Tower height meters"

Evidence Retrieved:
├─ Source 1: Wikipedia - "The Eiffel Tower... 330 metres (1,083 ft) tall"
├─ Source 2: ArchitectureToday - "Height: 330m with antenna"
└─ Source 3: TouristGuide - "Standing at 330 meters, the Eiffel Tower..."

Processing:
- Removes duplicates
- Filters irrelevant results
- Keeps top 5 sources per query
- Preserves source URL and relevance metadata
```

#### Step 6: Verification Agent (NLI Model)
The **Verifier Agent** uses a fine-tuned NLI (Natural Language Inference) model to compare claims against evidence:

```python
# Model: ClaimLens DeBERTa-v3-NLI (fine-tuned)
# Task: Determine if evidence supports, refutes, or is neutral toward claim

Processing for Claim: "The Eiffel Tower is 330 meters tall"

Evidence Analysis:
├─ Evidence 1: "330 metres tall"
│   └─ NLI Output: SUPPORTED (confidence: 0.94)
│
├─ Evidence 2: "330m with antenna"
│   └─ NLI Output: SUPPORTED (confidence: 0.89)
│
└─ Evidence 3: "Standing at 330 meters"
    └─ NLI Output: SUPPORTED (confidence: 0.92)

Final Verdict Calculation:
├─ Verdict: SUPPORTED
├─ Confidence: 0.92 (weighted average)
├─ Reasoning: "Multiple reliable sources confirm the Eiffel Tower height"
└─ Iterations Used: 1 (high confidence found on first search)
```

#### Step 7: Real-time Progress Tracking
As the pipeline runs, the frontend shows real-time progress:

```
Progress Display:
├─ ⏳ Decomposing claims...     [ACTIVE]
├─ ⏳ Generating search queries... [QUEUED]
├─ ⏳ Gathering evidence...    [QUEUED]
├─ ⏳ Verifying claims...      [QUEUED]
└─ ⏳ Aggregating results...   [QUEUED]

Per-Claim Tracking:
├─ [✓] Claim 1: "The Eiffel Tower is 330 meters tall"
├─ [⏳] Claim 2: "The Eiffel Tower was built in 1889"
└─ [○] Claim 3: Not yet processed
```

#### Step 8: Final Report
After all claims are verified, the user sees comprehensive results:

```json
{
  "overall_trust_score": 0.88,
  "summary": "88% of claims are supported by evidence. High confidence in provided information.",
  "claims_breakdown": {
    "supported": 2,
    "refuted": 0,
    "insufficient_info": 0
  },
  "verification_results": [
    {
      "claim": "The Eiffel Tower is 330 meters tall",
      "verdict": "SUPPORTED",
      "confidence": 0.92,
      "evidence_count": 3,
      "reasoning": "Multiple reliable sources confirm..."
    },
    {
      "claim": "The Eiffel Tower was built in 1889",
      "verdict": "SUPPORTED",
      "confidence": 0.95,
      "evidence_count": 5,
      "reasoning": "All major historical sources confirm..."
    }
  ]
}
```

### What Makes it Agentic

ClaimLens isn't a simple API call—it's a **multi-agent system** orchestrated by LangGraph:

1. **Independent Agents**: Each agent (decomposer, searcher, scraper, verifier) can make decisions autonomously
2. **Stateful Orchestration**: LangGraph manages state transitions between agents, with conditional logic:
   - If confidence is low → Generate more search queries and try again
   - If max iterations reached → Return best confidence found
   - If no claims extracted → Return empty report
3. **Evidence-Based Reasoning**: The verifier doesn't just match keywords—it understands semantic relationships through NLI
4. **Iterative Refinement**: The system can loop back to search_architect if initial verification confidence is below threshold (default 0.7)

### Verdict Types

The system returns one of three verdicts per claim:

| Verdict | Meaning | Example |
|---------|---------|---------|
| **SUPPORTED** | Evidence confirms the claim | "Snow is white" + evidence → SUPPORTED |
| **REFUTED** | Evidence contradicts the claim | "Snow is black" + evidence → REFUTED |
| **NOT_ENOUGH_INFO** | Evidence is inconclusive | Ambiguous claim + vague evidence → NOT_ENOUGH_INFO |

### Confidence Scoring

Each verdict includes a confidence score (0-1):
- **0.9+**: Very high confidence (multiple strong sources align)
- **0.7-0.9**: High confidence (primary sources support)
- **0.5-0.7**: Moderate confidence (some sources align, some ambiguous)
- **<0.5**: Low confidence (contradictory evidence or unclear)

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
