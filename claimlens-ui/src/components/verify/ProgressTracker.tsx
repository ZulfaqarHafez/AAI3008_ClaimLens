"use client";

import { useEffect, useState } from "react";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Circle,
  Clock,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Shield,
} from "lucide-react";
import type { ExtractedClaim, VerifiedClaim } from "@/hooks/useVerification";
import type { PipelineNode } from "@/types/api";

const THINKING_MESSAGES: Record<string, string[]> = {
  decompose_claims: [
    "Reading through the text...",
    "Identifying individual claims...",
    "Separating facts from opinions...",
    "Extracting atomic statements...",
  ],
  prepare_claim: ["Preparing the next claim...", "Setting up verification context..."],
  enrich_context: ["Adding helpful context...", "Linking venues and titles..."],
  frame_claim: ["Structuring the claim event...", "Extracting key dimensions..."],
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
  frame_evidence: ["Structuring evidence events...", "Aligning evidence to claim context..."],
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
  finalize_claim: ["Recording the verdict...", "Saving claim results..."],
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
    const timer = setInterval(() => setElapsed((seconds) => seconds + 1), 1000);
    return () => clearInterval(timer);
  }, []);
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function useThinkingMessage(currentNode: PipelineNode | null) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setIndex((i) => i + 1), 3000);
    return () => clearInterval(timer);
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
    <section className="panel rounded-[1.6rem] p-5 sm:p-6">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--brand)]/14 text-[var(--brand-strong)]">
            <Loader2 className="h-5 w-5 animate-spin" />
          </span>
          <div>
            <p className="text-sm font-semibold text-[#1c3a43]">{message}</p>
            {thinkingMsg && <p className="text-xs text-[#5d757d]">{thinkingMsg}</p>}
          </div>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-full border border-[#cfd8d1] bg-[#f0f6f4] px-3 py-1 text-xs font-semibold text-[#3f6068]">
          <Clock className="h-3.5 w-3.5" />
          <span className="tabular-nums">{elapsed}</span>
        </div>
      </div>

      {claims.length > 0 && (
        <div className="mt-5 space-y-3">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#5f777f]">
            Claims ({verified.length}/{claims.length})
          </p>
          {claims.map((claim) => {
            const done = verifiedIds.has(claim.id);
            const match = verified.find((v) => v.claim_id === claim.id);
            return (
              <ProgressClaimCard key={claim.id} claim={claim} done={done} match={match} />
            );
          })}
        </div>
      )}
    </section>
  );
}

function verdictStyle(verdict: string) {
  switch (verdict) {
    case "SUPPORTED":
      return {
        border: "border-emerald-200",
        bg: "bg-emerald-50/75",
        text: "text-emerald-700",
        badge: "bg-emerald-100 text-emerald-700",
        Icon: CheckCircle2,
      };
    case "REFUTED":
      return {
        border: "border-red-200",
        bg: "bg-red-50/75",
        text: "text-red-700",
        badge: "bg-red-100 text-red-700",
        Icon: XCircle,
      };
    default:
      return {
        border: "border-amber-200",
        bg: "bg-amber-50/75",
        text: "text-amber-700",
        badge: "bg-amber-100 text-amber-700",
        Icon: AlertCircle,
      };
  }
}

function ProgressClaimCard({
  claim,
  done,
  match,
}: {
  claim: ExtractedClaim;
  done: boolean;
  match?: VerifiedClaim;
}) {
  const [expanded, setExpanded] = useState(false);
  const style = match ? verdictStyle(match.verdict) : null;
  const hasEvidence = match?.evidence && match.evidence.length > 0;

  return (
    <article
      className={`rounded-xl border text-sm transition ${
        done && style ? `${style.border} ${style.bg}` : "border-[#d8d5c8] bg-[#f8f6ef]"
      }`}
    >
      <button
        onClick={() => done && hasEvidence && setExpanded(!expanded)}
        className={`flex w-full items-start gap-3 px-4 py-3 text-left ${
          done && hasEvidence ? "cursor-pointer" : "cursor-default"
        }`}
      >
        {done && style ? (
          <style.Icon className={`mt-0.5 h-4 w-4 shrink-0 ${style.text}`} />
        ) : (
          <Circle className="mt-0.5 h-4 w-4 shrink-0 text-[#9cb0b5]" />
        )}

        <div className="min-w-0 flex-1">
          <p className={done ? "text-[#28444e]" : "text-[#617a81]"}>{claim.text}</p>
          {match && (
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${style?.badge}`}>
                {match.verdict.replace(/_/g, " ")}
              </span>
              <span className="text-xs text-[#5e777f]">
                {Math.round(match.confidence * 100)}% confidence
              </span>
              {hasEvidence && (
                <span className="flex items-center gap-1 text-xs text-[#5e777f]">
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
              <ChevronUp className="h-4 w-4 text-[#6f858b]" />
            ) : (
              <ChevronDown className="h-4 w-4 text-[#6f858b]" />
            )}
          </div>
        )}
      </button>

      {expanded && match?.evidence && (
        <div className="space-y-2 border-t border-[#d8d5c8] px-4 py-3">
          {match.reasoning && (
            <p className="mb-1 text-xs text-[#4f666f]">
              <span className="font-semibold text-[#28444e]">Reasoning: </span>
              {match.reasoning}
            </p>
          )}
          <p className="text-xs font-semibold uppercase tracking-[0.13em] text-[#5f777f]">Sources</p>

          {match.evidence.map((evidence, j) => (
            <div key={j} className="rounded-lg border border-[#d8d5c8] bg-white/85 p-2.5">
              <div className="flex items-start justify-between gap-2">
                <p className="line-clamp-1 text-xs font-semibold text-[#26414b]">{evidence.title}</p>
                <a
                  href={evidence.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="shrink-0 text-[var(--brand)] hover:text-[var(--brand-strong)]"
                  onClick={(ev) => ev.stopPropagation()}
                >
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>

              <p className="mt-1 line-clamp-2 text-xs text-[#617a81]">{evidence.snippet}</p>
              <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-[#60797f]">
                {evidence.credibility_score != null && (
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3" />
                    Credibility: {Math.round(evidence.credibility_score * 100)}%
                  </span>
                )}
                {evidence.source_quality && <span>Quality: {evidence.source_quality}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}
