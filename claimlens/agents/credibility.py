"""Credibility Assessment Agent for evaluating source trustworthiness."""

import logging
from typing import List, Optional

from ..models.schemas import Evidence, Claim
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class CredibilityAgent:
    """Agent responsible for assessing the credibility of evidence sources.

    Evaluates sources on three dimensions:
    - Author expertise: Does the source demonstrate domain authority?
    - Publication recency: Is the information current and up-to-date?
    - Bias detection: Does the source show signs of bias or spin?

    Produces a credibility_score (0.0-1.0) and replaces the simple
    domain-based source_quality assessment.
    """

    CREDIBILITY_PROMPT = """You are an expert media literacy and source credibility analyst. Your task is to assess the credibility of web sources used for fact-checking.

For each evidence source, evaluate these three dimensions:

1. **Author Expertise** (0.0-1.0): Does the source demonstrate domain authority?
   - 0.9-1.0: Recognized expert, academic institution, official government source, peer-reviewed
   - 0.7-0.8: Reputable journalism, established news organization with editorial standards
   - 0.5-0.6: General news outlet, reasonably credible but not specialized
   - 0.3-0.4: Blog, opinion piece, user-generated content with some backing
   - 0.0-0.2: Anonymous source, no credentials, unverifiable authorship

2. **Recency** (0.0-1.0): Is the information current and relevant?
   - 0.9-1.0: Published within the last 6 months, highly current
   - 0.7-0.8: Published within the last 1-2 years, still relevant
   - 0.5-0.6: Published 2-5 years ago, may need updating
   - 0.3-0.4: Published 5-10 years ago, potentially outdated
   - 0.0-0.2: Very old or undated, likely outdated
   - Note: For timeless facts (geography, historical events), recency matters less — score higher

3. **Bias Detection** (0.0-1.0): How neutral and objective is the source?
   - 0.9-1.0: Highly objective, balanced coverage, cites multiple perspectives
   - 0.7-0.8: Mostly balanced, minor editorial slant but factual
   - 0.5-0.6: Noticeable bias but still contains factual information
   - 0.3-0.4: Strong editorial bias, selective facts, emotionally charged language
   - 0.0-0.2: Propaganda, misinformation, extreme bias, no factual basis

Provide:
- Scores for each dimension
- An overall credibility_score (weighted average: expertise 40%, recency 20%, bias 40%)
- A brief credibility assessment summary"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    def assess_credibility(
        self,
        claim: Claim,
        evidence_list: List[Evidence]
    ) -> List[Evidence]:
        """Assess credibility of each evidence source and update source_quality.

        Args:
            claim: The claim the evidence relates to
            evidence_list: List of evidence to assess

        Returns:
            Evidence list with updated credibility scores and source_quality
        """
        if not evidence_list:
            return []

        evidence_text = "\n\n".join([
            f"[Source {i+1}]\n"
            f"URL: {e.url}\n"
            f"Title: {e.title}\n"
            f"Content: {e.snippet[:400]}\n"
            f"Published Date: {e.published_date or 'Unknown'}"
            for i, e in enumerate(evidence_list)
        ])

        user_prompt = f"""Assess the credibility of these sources used to verify the following claim:

CLAIM: "{claim.text}"

SOURCES:
{evidence_text}

Evaluate each source on author expertise, recency, and bias detection."""

        try:
            response_schema = {
                "type": "object",
                "properties": {
                    "assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_index": {"type": "integer"},
                                "author_expertise": {"type": "number", "minimum": 0, "maximum": 1},
                                "recency": {"type": "number", "minimum": 0, "maximum": 1},
                                "bias_score": {"type": "number", "minimum": 0, "maximum": 1},
                                "credibility_score": {"type": "number", "minimum": 0, "maximum": 1},
                                "assessment": {"type": "string"}
                            },
                            "required": ["source_index", "author_expertise", "recency", "bias_score", "credibility_score", "assessment"]
                        }
                    }
                },
                "required": ["assessments"]
            }

            result = self.llm_service.generate_structured(
                system_prompt=self.CREDIBILITY_PROMPT,
                user_prompt=user_prompt,
                response_schema=response_schema,
                temperature=0.1
            )

            assessments = {
                a["source_index"]: a
                for a in result.get("assessments", [])
            }

            for i, evidence in enumerate(evidence_list):
                assessment = assessments.get(i + 1)  # 1-indexed in prompt

                if assessment:
                    score = assessment.get("credibility_score", 0.5)
                    evidence.credibility_score = score
                    evidence.credibility_reasoning = assessment.get("assessment", "")

                    # Map credibility score to source_quality label
                    if score >= 0.7:
                        evidence.source_quality = "high"
                    elif score >= 0.4:
                        evidence.source_quality = "medium"
                    else:
                        evidence.source_quality = "low"
                else:
                    # Fallback if assessment missing for this source
                    evidence.credibility_score = 0.5
                    evidence.credibility_reasoning = "Assessment unavailable"
                    evidence.source_quality = "medium"

            logger.info(
                f"Credibility assessed for {len(evidence_list)} sources. "
                f"Avg score: {sum(e.credibility_score for e in evidence_list) / len(evidence_list):.2f}"
            )

            return evidence_list

        except Exception as e:
            logger.error(f"Credibility assessment failed: {e}")
            # Fallback: keep existing source_quality, set default credibility
            for evidence in evidence_list:
                evidence.credibility_score = 0.5
                evidence.credibility_reasoning = "Assessment failed"
            return evidence_list
