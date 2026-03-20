"use client";

import { useEffect, useState } from "react";
import { Loader2, CheckCircle2, XCircle, AlertCircle, Circle, Clock, ExternalLink, ChevronDown, ChevronUp, Shield } from "lucide-react";
import type { ExtractedClaim, VerifiedClaim } from "@/hooks/useVerification";
import type { PipelineNode } from "@/types/api";

const THINKING_MESSAGES: Record<string, string[]> = {
  decompose_claims: [
    "Reading through the text...",
    "Identifying individual claims...",
    "Separating facts from opinions...",
    "Extracting atomic statements...",
  ],
  prepare_claim: [
    "Preparing the next claim...",
    "Setting up verification context...",
  ],
  enrich_context: [
    "Adding helpful context...",
    "Linking venues and titles...",
  ],
  frame_claim: [
    "Structuring the claim event...",
    "Extracting key dimensions...",
  ],
  generate_queries: [
    "Crafting search queries...",
    "Thinking about the best angles to verify this...",
    "Formulating fact-check queries...",
  ],
  search_evidence: [
    "Scanning the web for evidence...",
    "Checking reliable sources...",
    "Gathering relevant articles...",
    "Retrieving and filtering results...",
  ],
  frame_evidence: [
    "Structuring evidence events...",
    "Aligning evidence to claim context...",
  ],
  assess_credibility: [
    "Evaluating source credibility...",
    "Checking author expertise...",
    "Analyzing publication recency...",
    "Detecting potential bias...",
  ],
  verify_claim: [
    "Running NLI inference...",
    "Comparing claim against evidence...",
    "Analyzing textual entailment...",
    "Computing confidence scores...",
  ],
  finalize_claim: [
    "Recording the verdict...",
    "Saving claim results...",
  ],
  aggregate_results: [
    "Combining all verdicts...",
    "Calculating trust score...",
    "Weighing evidence quality...",
  ],
  generate_report: [
    "Writing the summary...",
    "Generating the final report...",
    "Putting it all together...",
  ],
};

function useElapsedTime() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function useThinkingMessage(currentNode: PipelineNode | null) {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    setIndex(0);
  }, [currentNode]);
  useEffect(() => {
    const t = setInterval(() => setIndex((i) => i + 1), 3000);
    return () => clearInterval(t);
  }, [currentNode]);
  const messages = currentNode ? THINKING_MESSAGES[currentNode] || [] : [];
  if (messages.length === 0) return null;
  return messages[index % messages.length];
}

interface Props {
  message: string;
  claims: ExtractedClaim[];
  verified: VerifiedClaim[];
  currentNode?: PipelineNode | null;
}

export default function ProgressTracker({ message, claims, verified, currentNode = null }: Props) {
  const verifiedIds = new Set(verified.map((v) => v.claim_id));
  const elapsed = useElapsedTime();
  const thinkingMsg = useThinkingMessage(currentNode);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
      {/* Spinner + message + elapsed */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
          <p className="text-sm font-medium text-gray-700">{message}</p>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-gray-400">
          <Clock className="h-3.5 w-3.5" />
          <span className="tabular-nums">{elapsed}</span>
        </div>
      </div>

      {/* Thinking sub-message */}
      {thinkingMsg && (
        <p className="mb-4 ml-9 text-xs text-indigo-400 animate-pulse">{thinkingMsg}</p>
      )}

      {!thinkingMsg && <div className="mb-4" />}

      {/* Claim list */}
      {claims.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Claims ({verified.length}/{claims.length})
          </p>
          {claims.map((c) => {
            const done = verifiedIds.has(c.id);
            const match = verified.find((v) => v.claim_id === c.id);
            return (
              <ProgressClaimCard key={c.id} claim={c} done={done} match={match} />
            );
          })}
        </div>
      )}
    </div>
  );
}

function verdictStyle(verdict: string) {
  switch (verdict) {
    case "SUPPORTED":
      return { border: "border-emerald-200", bg: "bg-emerald-50/50", text: "text-emerald-600", badge: "bg-emerald-100 text-emerald-700", Icon: CheckCircle2 };
    case "REFUTED":
      return { border: "border-red-200", bg: "bg-red-50/50", text: "text-red-600", badge: "bg-red-100 text-red-700", Icon: XCircle };
    default:
      return { border: "border-amber-200", bg: "bg-amber-50/50", text: "text-amber-600", badge: "bg-amber-100 text-amber-700", Icon: AlertCircle };
  }
}

function ProgressClaimCard({ claim, done, match }: { claim: ExtractedClaim; done: boolean; match?: VerifiedClaim }) {
  const [expanded, setExpanded] = useState(false);
  const style = match ? verdictStyle(match.verdict) : null;
  const hasEvidence = match?.evidence && match.evidence.length > 0;

  return (
    <div
      className={`rounded-lg border text-sm transition ${
        done && style ? `${style.border} ${style.bg}` : "border-gray-100 bg-gray-50"
      }`}
    >
      <button
        onClick={() => done && hasEvidence && setExpanded(!expanded)}
        className={`flex w-full items-start gap-3 px-4 py-3 text-left ${done && hasEvidence ? "cursor-pointer" : "cursor-default"}`}
      >
        {done && style ? (
          <style.Icon className={`mt-0.5 h-4 w-4 shrink-0 ${style.text}`} />
        ) : (
          <Circle className="mt-0.5 h-4 w-4 shrink-0 text-gray-300" />
        )}
        <div className="min-w-0 flex-1">
          <p className={done ? "text-gray-700" : "text-gray-500"}>{claim.text}</p>
          {match && (
            <div className="mt-1.5 flex items-center gap-2">
              <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${style?.badge}`}>
                {match.verdict.replace(/_/g, " ")}
              </span>
              <span className="text-xs text-gray-400">
                {Math.round(match.confidence * 100)}% confidence
              </span>
              {hasEvidence && (
                <span className="flex items-center gap-1 text-xs text-gray-400">
                  <Shield className="h-3 w-3" />
                  {match.evidence!.length} source{match.evidence!.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          )}
        </div>
        {done && hasEvidence && (
          <div className="mt-1">
            {expanded ? (
              <ChevronUp className="h-4 w-4 text-gray-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-gray-400" />
            )}
          </div>
        )}
      </button>

      {/* Expanded evidence section */}
      {expanded && match?.evidence && (
        <div className="border-t border-gray-100 px-4 py-3 space-y-2">
          {match.reasoning && (
            <p className="text-xs text-gray-500 mb-2">
              <span className="font-semibold text-gray-700">Reasoning: </span>
              {match.reasoning}
            </p>
          )}
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Sources
          </p>
          {match.evidence.map((e, j) => (
            <div key={j} className="rounded-md border border-gray-100 bg-white p-2.5">
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs font-semibold text-gray-700 line-clamp-1">{e.title}</p>
                <a
                  href={e.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="shrink-0 text-indigo-500 hover:text-indigo-700"
                  onClick={(ev) => ev.stopPropagation()}
                >
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              <p className="mt-1 text-xs text-gray-400 line-clamp-2">{e.snippet}</p>
              <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-gray-400">
                {e.credibility_score != null && (
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    Credibility: {Math.round(e.credibility_score * 100)}%
                  </span>
                )}
                {e.source_quality && (
                  <span>Quality: {e.source_quality}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
