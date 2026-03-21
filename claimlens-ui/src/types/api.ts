export enum Verdict {
  SUPPORTED = "SUPPORTED",
  REFUTED = "REFUTED",
  NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO",
}

export enum ClaimStatus {
  PENDING = "pending",
  SEARCHING = "searching",
  VERIFYING = "verifying",
  COMPLETED = "completed",
  FAILED = "failed",
}

export enum JobStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
}

export interface Claim {
  id: string;
  text: string;
  source_sentence: string;
  status: ClaimStatus;
}

export interface Evidence {
  url: string;
  title: string;
  snippet: string;
  relevance_score: number;
  source_quality?: string;
  credibility_score?: number;
  credibility_reasoning?: string;
  published_date?: string;
  retrieved_at?: string;
}

export interface VerificationResult {
  claim: Claim;
  evidence_list: Evidence[];
  verdict: Verdict;
  confidence: number;
  reasoning: string;
  iterations_used: number;
}

export interface FinalReport {
  id: string;
  original_text: string;
  claims: Claim[];
  verification_results: VerificationResult[];
  overall_trust_score: number;
  summary?: string;
  created_at: string;
  processing_time_seconds?: number;
}

export interface StreamEvent {
  event_type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface ClaimsExtractedData {
  claims: { id: string; text: string; status: string }[];
  count: number;
}

export interface ClaimVerifiedData {
  claim_id: string;
  claim_text: string;
  verdict: Verdict;
  confidence: number;
  reasoning?: string;
  evidence?: Evidence[];
}

export interface CompleteData {
  trust_score: number;
  summary: string;
  total_claims: number;
  report?: FinalReport;
}

export type PipelineNode =
  | "decompose_claims"
  | "prepare_claim"
  | "assess_context"
  | "generate_queries"
  | "search_evidence"
  | "assess_credibility"
  | "verify_claim"
  | "finalize_claim"
  | "aggregate_results"
  | "generate_report";

export interface StepData {
  node: PipelineNode;
}
