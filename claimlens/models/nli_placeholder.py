"""NLI placeholder models for verification."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import logging

from ..models.schemas import Evidence, VerificationResult, Verdict, Claim

logger = logging.getLogger(__name__)


class BaseVerifier(ABC):
    """Abstract base class for claim verifiers."""

    @abstractmethod
    def verify(
        self,
        claim: Claim,
        evidence: List[Evidence],
        enriched_claim_text: str = "",
    ) -> VerificationResult:
        pass

    @abstractmethod
    def batch_verify(
        self,
        claims_evidence: List[Tuple[Claim, List[Evidence]]],
    ) -> List[VerificationResult]:
        pass


# ---------------------------------------------------------------------------
# HuggingFace zero-shot verifier
# ---------------------------------------------------------------------------

class HuggingFaceNLIVerifier(BaseVerifier):
    """Verifier using HuggingFace BART-large-MNLI zero-shot classification."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self._pipeline  = None

    def _load_model(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading NLI model: {self.model_name}")
                self._pipeline = pipeline("zero-shot-classification", model=self.model_name)
            except ImportError:
                raise ImportError("transformers required: pip install transformers torch")
        return self._pipeline

    def _classify_nli(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        pipeline = self._load_model()
        labels   = ["entailment", "contradiction", "neutral"]
        result   = pipeline(
            premise,
            candidate_labels=labels,
            hypothesis_template="This text {} the claim: {}".format("{}", hypothesis),
        )
        return result["labels"][0], result["scores"][0]

    def _aggregate(self, claim: str, evidence_list: List[Evidence]) -> Tuple[Verdict, float, str]:
        if not evidence_list:
            return Verdict.NOT_ENOUGH_INFO, 0.3, "No evidence available."

        e_s, c_s, n_s = [], [], []
        for ev in evidence_list:
            label, score = self._classify_nli(ev.snippet, claim)
            bucket = e_s if label == "entailment" else (c_s if label == "contradiction" else n_s)
            bucket.append(score * ev.relevance_score)

        avg_e = sum(e_s) / len(e_s) if e_s else 0
        avg_c = sum(c_s) / len(c_s) if c_s else 0
        avg_n = sum(n_s) / len(n_s) if n_s else 0
        mx    = max(avg_e, avg_c, avg_n)

        if mx < 0.4:
            return Verdict.NOT_ENOUGH_INFO, 0.3 + mx * 0.2, "Evidence inconclusive."
        if avg_e >= avg_c and avg_e >= avg_n:
            return Verdict.SUPPORTED,       min(0.95, 0.5 + avg_e * 0.5), f"{len(e_s)} supporting sources."
        if avg_c > avg_e  and avg_c > avg_n:
            return Verdict.REFUTED,         min(0.95, 0.5 + avg_c * 0.5), f"{len(c_s)} contradicting sources."
        return Verdict.NOT_ENOUGH_INFO, 0.4, "Evidence mixed or neutral."

    def verify(self, claim, evidence, enriched_claim_text=""):
        hypothesis = enriched_claim_text if enriched_claim_text else claim.text
        verdict, confidence, reasoning = self._aggregate(hypothesis, evidence)
        return VerificationResult(
            claim=claim, evidence_list=evidence,
            verdict=verdict, confidence=confidence, reasoning=reasoning,
        )

    def batch_verify(self, claims_evidence):
        return [self.verify(c, e) for c, e in claims_evidence]


# ---------------------------------------------------------------------------
# OpenAI LLM verifier
# ---------------------------------------------------------------------------

class OpenAIVerifier(BaseVerifier):
    """Verifier using OpenAI GPT with structured output."""

    def __init__(self, llm_service=None):
        self._llm_service = llm_service

    @property
    def llm_service(self):
        if self._llm_service is None:
            from ..services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    def verify(self, claim, evidence, enriched_claim_text=""):
        if not evidence:
            return VerificationResult(
                claim=claim, evidence_list=[],
                verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.3,
                reasoning="No evidence available.",
            )
        hypothesis    = enriched_claim_text if enriched_claim_text else claim.text
        evidence_text = "\n\n".join([
            f"Source {i+1} ({e.url}):\n{e.snippet}" for i, e in enumerate(evidence)
        ])

        system_prompt = """You are an expert fact-checker performing Natural Language Inference (NLI).
Your task is to determine if the provided evidence SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO for the given claim.

Rules:
1. Verify the FULL EVENT described in the claim. Do not split into independent facts.
2. The SAME EVENT must match across person, action, location/venue, time/date, and context.
3. Use contextual reasoning for equivalent or hierarchical concepts.
   Examples: "Committee of Supply debate" is part of Parliament; "DPM" means "Deputy Prime Minister";
   speaking during a parliamentary debate means speaking in Parliament.
4. If evidence confirms the event but uses different wording, treat it as SUPPORTING.
5. If evidence contradicts any key dimension (person, date, location, or event), REFUTE.
6. If evidence is incomplete or cannot confirm the full event, NOT_ENOUGH_INFO.
7. Prefer authoritative sources in this order: government, major news, academic/institutional, other web, social.

Respond with a JSON object containing:
- verdict: "SUPPORTED", "REFUTED", or "NOT_ENOUGH_INFO"
- confidence: A float between 0.0 and 1.0
- reasoning: A brief explanation (1-2 sentences)"""

        context_block = self._format_context_block(claim)

        user_prompt = f"""Claim to verify:
"{hypothesis}"
{context_block}

Evidence:
{evidence_text}

Analyze the evidence and provide your verdict."""
        try:
            result = self.llm_service.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema={
                    "type": "object",
                    "properties": {
                        "verdict":    {"type": "string", "enum": ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reasoning":  {"type": "string"},
                    },
                    "required": ["verdict", "confidence", "reasoning"],
                },
            )
            return VerificationResult(
                claim=claim, evidence_list=evidence,
                verdict=Verdict(result["verdict"]),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
            )
        except Exception as e:
            logger.error(f"OpenAI verification failed: {e}")
            return VerificationResult(
                claim=claim, evidence_list=evidence,
                verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.3,
                reasoning=f"Verification failed: {str(e)}",
            )

    def batch_verify(self, claims_evidence):
        return [self.verify(c, e) for c, e in claims_evidence]

    def _format_context_block(self, claim: Claim) -> str:
        context = getattr(claim, "context", None)
        if not context:
            return ""

        lines = []
        if context.normalized_claim and context.normalized_claim != claim.text:
            lines.append(f"Normalized claim: {context.normalized_claim}")
        if context.enriched_claim_text and context.enriched_claim_text != claim.text:
            lines.append(f"Enriched claim: {context.enriched_claim_text}")
        if context.context_summary:
            lines.append(f"Context summary: {context.context_summary}")
        if context.temporal_context:
            lines.append(f"Temporal context: {context.temporal_context}")
        if context.venue_context:
            lines.append(f"Venue context: {context.venue_context}")
        if context.entity_aliases:
            lines.append(f"Entity aliases: {', '.join(context.entity_aliases[:6])}")
        if context.context_notes:
            notes = "; ".join(f"{n.entity}: {n.note}" for n in context.context_notes[:6])
            lines.append(f"Context notes: {notes}")

        if not lines:
            return ""

        return "\nADDITIONAL CONTEXT:\n" + "\n".join(f"- {line}" for line in lines)


# ---------------------------------------------------------------------------
# ClaimLens fine-tuned DeBERTa-v3 verifier  (primary verifier)
# ---------------------------------------------------------------------------

class ClaimLensVerifier(BaseVerifier):
    """Verifier using the fine-tuned ClaimLens DeBERTa-v3 NLI model.

    Labels: SUPPORTED (0), REFUTED (1), NEI (2).
    Model: Zulfhagez/claimlens-deberta-v3-nli

    LLM Cross-check triggers
    ------------------------
    Trigger A — REFUTED + avg_relevance >= 0.65
        DeBERTa commonly misclassifies as REFUTED when evidence has
        numerical values near (but not identical to) those in the claim.

    Trigger B — NOT_ENOUGH_INFO + avg_relevance >= 0.60  (lowered from 0.70)
        When evidence is relevant but NLI returns NEI, the key supporting
        sentence is usually present but DeBERTa failed to extract it.
        Lowering to 0.60 catches more borderline cases (e.g. claims 4 & 5
        in the Iran/Trump test where PBS at 90% relevance was ignored).

    Trigger C — any verdict with confidence < 0.55
        Low NLI confidence → defer to LLM.

    Override rule: LLM overrides NLI only when it disagrees AND its
    own confidence is >= LLM_OVERRIDE_MIN_CONFIDENCE (0.70).

    Cross-check hygiene
    -------------------
    The hypothesis passed to _llm_cross_check is ALWAYS claim.text (the
    original atomic claim), NOT the enriched_claim_text.  This prevents
    context-agent rewrites from contaminating the cross-check prompt with
    irrelevant background that causes the LLM to evaluate the wrong thing.
    The enriched text is only used for the NLI model where it helps expand
    acronyms; the LLM cross-check is smart enough to understand the plain
    atomic claim.
    """

    LABEL_MAP = {
        0: Verdict.SUPPORTED,
        1: Verdict.REFUTED,
        2: Verdict.NOT_ENOUGH_INFO,
    }

    # Cross-check trigger thresholds
    CROSS_CHECK_REFUTED_MIN_RELEVANCE = 0.65   # Trigger A
    CROSS_CHECK_NEI_MIN_RELEVANCE     = 0.60   # Trigger B (lowered from 0.70)
    CROSS_CHECK_LOW_CONFIDENCE        = 0.55   # Trigger C

    # LLM must have this confidence to override NLI
    LLM_OVERRIDE_MIN_CONFIDENCE = 0.70

    def __init__(self, model_path: str = "Zulfhagez/claimlens-deberta-v3-nli"):
        self.model_path = model_path
        self._model     = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch  # noqa: F401
            except ImportError:
                raise ImportError(
                    "transformers and torch required: "
                    "pip install transformers torch sentencepiece accelerate"
                )
            logger.info(f"Loading ClaimLens NLI model: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model     = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()

    def _run_nli(self, hypothesis: str, evidence_snippet: str) -> Tuple[Verdict, float]:
        """Run a single NLI inference pass."""
        import torch
        self._load_model()

        inputs = self._tokenizer(
            evidence_snippet,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs      = torch.softmax(logits, dim=-1).squeeze()
        pred_idx   = int(probs.argmax())
        confidence = float(probs[pred_idx])
        label      = self.LABEL_MAP[pred_idx]
        return label, confidence

    def _llm_cross_check(
        self,
        original_claim_text: str,   # ← ALWAYS the raw atomic claim, never enriched
        evidence: List[Evidence],
        nli_verdict: Verdict,
        nli_confidence: float,
    ) -> Tuple[Verdict, float, str]:
        """Cross-check the NLI result with an OpenAI LLM.

        IMPORTANT: Uses original_claim_text (not the enriched hypothesis) to
        prevent context-agent rewrites from confusing the LLM cross-check.
        The LLM is smart enough to understand the plain atomic claim; enriched
        text often adds irrelevant background that causes it to evaluate the
        wrong thing.

        Uses ALL evidence pieces (up to 5), sorted by relevance, to maximise
        the chance of the LLM finding the key supporting sentence.
        """
        try:
            from ..services.llm_service import LLMService
            llm = LLMService()

            # Sort by relevance, use all (up to 5)
            sorted_ev     = sorted(evidence, key=lambda e: e.relevance_score, reverse=True)
            evidence_text = "\n\n".join([
                f"[Source {i+1}] {e.title}\n{e.snippet[:700]}"
                for i, e in enumerate(sorted_ev[:5])
            ])

            result = llm.generate_structured(
                system_prompt=(
                    "You are an expert fact-checker. Given a single atomic claim and "
                    "a set of evidence sources, determine whether the evidence "
                    "SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO for the claim.\n\n"
                    "Definitions:\n"
                    "- SUPPORTED: at least one source confirms the claim is true.\n"
                    "- REFUTED: at least one source directly contradicts the claim.\n"
                    "- NOT_ENOUGH_INFO: no source addresses the claim, or evidence "
                    "is irrelevant or inconclusive.\n\n"
                    "Key rules:\n"
                    "1. Read each source carefully and fully before deciding.\n"
                    "2. A source that contains the same fact, statistic, or quote as "
                    "the claim is SUPPORTING evidence — not refuting evidence.\n"
                    "3. Focus ONLY on the specific claim provided. Ignore surrounding "
                    "context from the article the claim may have come from.\n"
                    "4. Only mark as NOT_ENOUGH_INFO if none of the sources address "
                    "the specific claim at all.\n"
                    "Respond only with valid JSON."
                ),
                user_prompt=(
                    f'CLAIM: "{original_claim_text}"\n\n'
                    f"EVIDENCE SOURCES:\n{evidence_text}"
                ),
                response_schema={
                    "type": "object",
                    "properties": {
                        "verdict":    {"type": "string", "enum": ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reasoning":  {"type": "string"},
                    },
                    "required": ["verdict", "confidence", "reasoning"],
                },
                temperature=0.1,
            )

            return (
                Verdict(result["verdict"]),
                float(result["confidence"]),
                result.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"LLM cross-check failed: {e}")
            return nli_verdict, nli_confidence, ""

    def verify(
        self,
        claim: Claim,
        evidence: List[Evidence],
        enriched_claim_text: str = "",
    ) -> VerificationResult:
        """Verify a claim using weighted NLI voting + LLM cross-check.

        Args:
            claim:                The claim to verify.
            evidence:             Evidence list (already relevance-scored).
            enriched_claim_text:  Optional context-enriched hypothesis from
                                  the ContextAgent — used ONLY for NLI.
                                  The LLM cross-check always uses claim.text.
        """
        # 1. NLI hypothesis — enriched text expands acronyms for the model
        nli_hypothesis = enriched_claim_text if enriched_claim_text else claim.text

        # 2. No evidence → early exit
        if not evidence:
            return VerificationResult(
                claim=claim, evidence_list=[],
                verdict=Verdict.NOT_ENOUGH_INFO, confidence=0.3,
                reasoning="No evidence available for verification.",
            )

        # 3. Weighted NLI voting
        weighted_scores: dict[Verdict, float] = {
            Verdict.SUPPORTED:       0.0,
            Verdict.REFUTED:         0.0,
            Verdict.NOT_ENOUGH_INFO: 0.0,
        }
        for ev in evidence:
            label, conf = self._run_nli(nli_hypothesis, ev.snippet)
            weighted_scores[label] += conf * ev.relevance_score

        # 4. Derive verdict
        verdict      = max(weighted_scores, key=weighted_scores.get)
        total_weight = sum(weighted_scores.values())
        confidence   = (
            weighted_scores[verdict] / total_weight if total_weight > 0 else 0.0
        )
        confidence = min(confidence, 0.99)

        label_counts = {
            v.value: round(weighted_scores[v], 3)
            for v in Verdict if weighted_scores[v] > 0
        }
        reasoning = (
            f"Aggregated {len(evidence)} evidence piece(s) via weighted voting. "
            f"Weighted scores: {label_counts}."
        )

        # 5. LLM cross-check decision
        avg_relevance = sum(ev.relevance_score for ev in evidence) / len(evidence)

        trigger_a = verdict == Verdict.REFUTED         and avg_relevance >= self.CROSS_CHECK_REFUTED_MIN_RELEVANCE
        trigger_b = verdict == Verdict.NOT_ENOUGH_INFO and avg_relevance >= self.CROSS_CHECK_NEI_MIN_RELEVANCE
        trigger_c = confidence < self.CROSS_CHECK_LOW_CONFIDENCE

        if trigger_a or trigger_b or trigger_c:
            trigger_name = (
                "REFUTED+high_relevance"   if trigger_a else
                "NEI+high_relevance"       if trigger_b else
                "low_confidence"
            )
            logger.info(
                f"NLI: {verdict.value} ({confidence:.2f}), avg_rel={avg_relevance:.2f} "
                f"— LLM cross-check [{trigger_name}]"
            )

            # ↓ Pass claim.text (original), NOT nli_hypothesis (enriched)
            # This prevents context rewrites from contaminating the cross-check.
            llm_verdict, llm_conf, llm_reasoning = self._llm_cross_check(
                original_claim_text=claim.text,
                evidence=evidence,
                nli_verdict=verdict,
                nli_confidence=confidence,
            )

            if llm_verdict != verdict and llm_conf >= self.LLM_OVERRIDE_MIN_CONFIDENCE:
                logger.info(
                    f"LLM overrides NLI: {verdict.value} → {llm_verdict.value} "
                    f"(LLM conf={llm_conf:.2f})"
                )
                verdict    = llm_verdict
                confidence = llm_conf
                reasoning  = (
                    f"[LLM cross-check override] {llm_reasoning} "
                    f"| NLI scores: {label_counts}"
                )
            else:
                if llm_reasoning:
                    reasoning = f"{reasoning} | LLM agrees: {llm_reasoning}"

        # 6. Return result
        return VerificationResult(
            claim=claim,
            evidence_list=evidence,
            verdict=verdict,
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )

    def batch_verify(self, claims_evidence):
        return [self.verify(claim, ev) for claim, ev in claims_evidence]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_verifier(verifier_type: str = "openai", **kwargs) -> BaseVerifier:
    """Factory function to create the appropriate verifier."""
    verifiers = {
        "huggingface": HuggingFaceNLIVerifier,
        "openai":      OpenAIVerifier,
        "claimlens":   ClaimLensVerifier,
    }
    if verifier_type not in verifiers:
        raise ValueError(
            f"Unknown verifier type: {verifier_type}. "
            f"Choose from: {list(verifiers.keys())}"
        )
    return verifiers[verifier_type](**kwargs)
