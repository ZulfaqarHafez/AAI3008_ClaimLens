# ClaimLens 🔍

An agentic fact-checking pipeline using LangGraph that decomposes user-provided paragraphs into atomic claims and verifies each claim against web evidence.

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
│  │  Report  │    │ Results  │    │ (HF NLI / OpenAI / Custom)│     │
│  └──────────┘    └──────────┘    └──────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Tavily API key (or SerpAPI key)

### Installation

```bash
# Clone the repository
cd AAI3008_ClaimLens

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here
```

### Running the API

```bash
# Start the FastAPI server
uvicorn claimlens.api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for the interactive API documentation.

## 📁 Project Structure

```
claimlens/
├── agents/
│   ├── __init__.py
│   ├── decomposition.py     # Breaks text into atomic claims
│   ├── search_architect.py  # Generates search queries
│   ├── scraper.py           # Retrieves and filters evidence
│   └── verifier.py          # Verifies claims against evidence
├── models/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic data models
│   └── nli_placeholder.py   # NLI verifier implementations
├── graph/
│   ├── __init__.py
│   └── orchestrator.py      # LangGraph state machine
├── services/
│   ├── __init__.py
│   ├── llm_service.py       # OpenAI API wrapper
│   └── search_service.py    # Web search API wrapper
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI endpoints
├── __init__.py
└── config.py                # Configuration settings
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

### Streaming Client (JavaScript)

```javascript
const eventSource = new EventSource('/verify/stream', {
  method: 'POST',
  body: JSON.stringify({ text: 'Your text here...' })
});

eventSource.addEventListener('claims_extracted', (e) => {
  console.log('Claims:', JSON.parse(e.data));
});

eventSource.addEventListener('claim_verified', (e) => {
  console.log('Verified:', JSON.parse(e.data));
});

eventSource.addEventListener('complete', (e) => {
  console.log('Complete:', JSON.parse(e.data));
});
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
| `VERIFIER_TYPE` | openai | Verifier backend (openai/huggingface) |
| `SEARCH_PROVIDER` | tavily | Search API (tavily/serpapi) |

## 🔄 Swapping the Verifier

The verifier component is designed for easy swapping. To use your custom DeBERTa-v3 model:

```python
from claimlens.models.nli_placeholder import BaseVerifier
from claimlens.agents.verifier import VerifierAgent

class ClaimLensVerifier(BaseVerifier):
    def __init__(self, model_path: str):
        # Load your custom model
        self.model = load_your_model(model_path)
    
    def verify(self, claim, evidence):
        # Implement your verification logic
        pass

# Use custom verifier
custom_verifier = ClaimLensVerifier("/path/to/model")
verifier_agent = VerifierAgent(verifier=custom_verifier)
```

## 📊 Data Models

### Claim
```python
{
  "id": "uuid",
  "text": "The atomic claim text",
  "source_sentence": "Original sentence",
  "status": "pending|searching|verifying|completed|failed"
}
```

### VerificationResult
```python
{
  "claim": Claim,
  "evidence_list": [Evidence],
  "verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_INFO",
  "confidence": 0.0-1.0,
  "reasoning": "Explanation",
  "iterations_used": 1-3
}
```

### FinalReport
```python
{
  "id": "uuid",
  "original_text": "Input text",
  "claims": [Claim],
  "verification_results": [VerificationResult],
  "overall_trust_score": 0.0-1.0,
  "summary": "Human-readable summary",
  "processing_time_seconds": float
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 🔮 Future Improvements

- [ ] Integrate custom DeBERTa-v3 NLI model
- [ ] Add Redis for job persistence
- [ ] Implement rate limiting
- [ ] Add claim deduplication
- [ ] Support for multiple languages
- [ ] Batch processing endpoint
- [ ] WebSocket support for real-time updates
- [ ] Source credibility scoring
- [ ] Claim provenance tracking
