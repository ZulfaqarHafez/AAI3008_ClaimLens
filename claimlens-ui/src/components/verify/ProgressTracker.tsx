"use client";

import { useEffect, useState, useMemo } from "react";
import { Loader2, CheckCircle2, Circle, Clock } from "lucide-react";
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
              <div
                key={c.id}
                className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm transition ${
                  done ? "border-emerald-200 bg-emerald-50/50" : "border-gray-100 bg-gray-50"
                }`}
              >
                {done ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                ) : (
                  <Circle className="mt-0.5 h-4 w-4 shrink-0 text-gray-300" />
                )}
                <div className="min-w-0 flex-1">
                  <p className={done ? "text-gray-700" : "text-gray-500"}>{c.text}</p>
                  {match && (
                    <p className="mt-1 text-xs text-gray-400">
                      {match.verdict} — {Math.round(match.confidence * 100)}% confidence
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
