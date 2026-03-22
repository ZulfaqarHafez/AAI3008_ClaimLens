"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  RotateCcw,
  ExternalLink,
  Calendar,
  CheckCircle2,
  XCircle,
  AlertCircle,
  FileText,
} from "lucide-react";
import type { FinalReport, VerificationResult, Verdict } from "@/types/api";
import { VERDICT_CONFIG } from "@/constants/verdicts";

function formatDate(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return dateStr;
    return date.toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return dateStr;
  }
}

function trustColour(pct: number) {
  if (pct >= 75) {
    return {
      text: "text-emerald-700",
      ring: "stroke-emerald-600",
      panel: "from-emerald-50 to-emerald-100/80",
      tone: "text-emerald-700",
    };
  }
  if (pct >= 55) {
    return {
      text: "text-amber-700",
      ring: "stroke-amber-600",
      panel: "from-amber-50 to-amber-100/80",
      tone: "text-amber-700",
    };
  }
  return {
    text: "text-rose-700",
    ring: "stroke-rose-600",
    panel: "from-rose-50 to-rose-100/80",
    tone: "text-rose-700",
  };
}

function trustLabel(pct: number) {
  if (pct >= 80) return "Highly Trustworthy";
  if (pct >= 65) return "Mostly Trustworthy";
  if (pct >= 45) return "Mixed Reliability";
  if (pct >= 25) return "Low Trustworthiness";
  return "Very Unreliable";
}

function TrustRing({ pct, colours }: { pct: number; colours: ReturnType<typeof trustColour> }) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const dash = (pct / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center">
      <svg width="112" height="112" viewBox="0 0 112 112" className="-rotate-90">
        <circle cx="56" cy="56" r={radius} fill="none" stroke="#d7d6c8" strokeWidth="9" />
        <circle
          cx="56"
          cy="56"
          r={radius}
          fill="none"
          strokeWidth="9"
          strokeLinecap="round"
          className={`${colours.ring} transition-all duration-700`}
          strokeDasharray={`${dash} ${circumference}`}
        />
      </svg>
      <span className={`absolute text-3xl font-extrabold tabular-nums ${colours.text}`}>{pct}</span>
    </div>
  );
}

interface Props {
  report: FinalReport;
  onReset: () => void;
}

function computeEvidenceStrength(evidenceList: VerificationResult["evidence_list"]) {
  if (!evidenceList || evidenceList.length === 0) return 0.2;
  const avgRel =
    evidenceList.reduce((sum, e) => sum + (e.relevance_score ?? 0), 0) /
    evidenceList.length;
  const avgCred =
    evidenceList.reduce((sum, e) => sum + (e.credibility_score ?? 0.5), 0) /
    evidenceList.length;
  const coverage = Math.min(1, evidenceList.length / 3);
  const signal = 0.6 * avgRel + 0.4 * avgCred;
  return Math.max(0.2, Math.min(1, signal * coverage));
}

function computeClaimPct(result: VerificationResult) {
  const strength = computeEvidenceStrength(result.evidence_list);
  const score = 0.7 * result.confidence + 0.3 * strength;
  return Math.round(score * 100);
}

export default function ResultsView({ report, onReset }: Props) {
  const [showOriginal, setShowOriginal] = useState(false);

  const trustPct = Math.round(report.overall_trust_score * 100);
  const colours = trustColour(trustPct);

  const supported = report.verification_results.filter((r) => r.verdict === "SUPPORTED").length;
  const refuted = report.verification_results.filter((r) => r.verdict === "REFUTED").length;
  const nei = report.verification_results.filter((r) => r.verdict === "NOT_ENOUGH_INFO").length;
  const total = report.verification_results.length;

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-[1.8rem] border border-[#d8d5c8] bg-white p-6 shadow-[0_20px_45px_-35px_rgba(16,42,52,0.52)] sm:p-7">
        <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${colours.panel} opacity-70`} />
        <div className="relative flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-xl">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#5e767e]">Overall Trust Score</p>
            <p className={`mt-1 text-5xl font-extrabold ${colours.text}`}>{trustPct}%</p>
            <p className={`mt-1 text-sm font-semibold ${colours.tone}`}>{trustLabel(trustPct)}</p>
            {report.summary && <p className="mt-3 text-sm leading-relaxed text-[#45636c]">{report.summary}</p>}
            {report.processing_time_seconds != null && (
              <p className="mt-1.5 text-xs font-semibold text-[#547078]">
                Processed in {report.processing_time_seconds.toFixed(1)}s
              </p>
            )}
          </div>
          <TrustRing pct={trustPct} colours={colours} />
        </div>

        {total > 0 && (
          <div className="relative mt-6 border-t border-[#d8d5c8] pt-5">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-[#5e767e]">Verdict Breakdown</p>
            <div className="mb-3 flex h-3 overflow-hidden rounded-full bg-[#dbdbcf]">
              {supported > 0 && <div className="bg-emerald-500" style={{ width: `${(supported / total) * 100}%` }} />}
              {refuted > 0 && <div className="bg-rose-500" style={{ width: `${(refuted / total) * 100}%` }} />}
              {nei > 0 && <div className="bg-slate-500" style={{ width: `${(nei / total) * 100}%` }} />}
            </div>
            <div className="flex flex-wrap items-center gap-4 text-xs font-semibold">
              <span className="inline-flex items-center gap-1.5 text-emerald-700">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {supported} Supported
              </span>
              <span className="inline-flex items-center gap-1.5 text-rose-700">
                <XCircle className="h-3.5 w-3.5" />
                {refuted} Refuted
              </span>
              <span className="inline-flex items-center gap-1.5 text-slate-600">
                <AlertCircle className="h-3.5 w-3.5" />
                {nei} Not Enough Info
              </span>
            </div>
          </div>
        )}
      </section>

      {report.original_text && (
        <section className="panel rounded-[1.55rem] overflow-hidden">
          <button
            onClick={() => setShowOriginal((v) => !v)}
            className="surface-hover tap-soft flex w-full items-center justify-between px-5 py-4 text-left"
          >
            <div className="flex items-center gap-2 text-sm font-semibold text-[#25424c]">
              <FileText className="h-4 w-4 text-[var(--brand)]" />
              Original Text
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-[#5e767d]">
                {report.original_text.length} chars, {total} claim{total !== 1 ? "s" : ""}
              </span>
              {showOriginal ? (
                <ChevronUp className="h-4 w-4 text-[#5e767d]" />
              ) : (
                <ChevronDown className="h-4 w-4 text-[#5e767d]" />
              )}
            </div>
          </button>
          {showOriginal && (
            <div className="border-t border-[#d8d5c8] px-5 py-4">
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-[#395861]">{report.original_text}</p>
            </div>
          )}
        </section>
      )}

      {report.verification_results.map((result, i) => (
        <ResultCard
          key={result.claim.id ?? i}
          result={result}
          index={i}
          animationDelay={120 + i * 55}
        />
      ))}

      <div className="text-center">
        <button
          onClick={onReset}
          className="btn-primary tap-soft inline-flex items-center gap-2 rounded-full px-6 py-2.5 text-sm font-semibold"
        >
          <RotateCcw className="h-4 w-4" />
          Verify Another Text
        </button>
      </div>
    </div>
  );
}

function ResultCard({
  result,
  index,
  animationDelay,
}: {
  result: VerificationResult;
  index: number;
  animationDelay: number;
}) {
  const [open, setOpen] = useState(false);
  const verdictConfig = VERDICT_CONFIG[result.verdict as Verdict] ?? VERDICT_CONFIG.NOT_ENOUGH_INFO;
  const Icon = verdictConfig.icon;
  const claimPct = computeClaimPct(result);

  const barColour =
    claimPct >= 80 ? "bg-emerald-500" : claimPct >= 55 ? "bg-amber-500" : "bg-rose-500";

  return (
    <article
      className="panel animated-rise overflow-hidden rounded-[1.45rem]"
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      <button onClick={() => setOpen(!open)} className="surface-hover tap-soft flex w-full items-start gap-4 p-5 text-left">
        <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${verdictConfig.bg}`}>
          <Icon className={`h-4 w-4 ${verdictConfig.text}`} />
        </div>

        <div className="min-w-0 flex-1">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#5f767f]">Claim {index + 1}</p>
          <p className="mt-1 text-sm font-semibold text-[#17333d]">{result.claim.text}</p>

          <div className="mt-2 flex flex-wrap items-center gap-3">
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${verdictConfig.bg} ${verdictConfig.text}`}>
              {verdictConfig.label}
            </span>
            <div className="flex items-center gap-2">
              <div className="h-2 w-28 overflow-hidden rounded-full bg-[#d9dbcf]">
                <div className={`h-full rounded-full ${barColour}`} style={{ width: `${claimPct}%` }} />
              </div>
              <span className="text-xs font-semibold text-[#4f6770]">{claimPct}%</span>
              <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#647d84]">Claim Score</span>
            </div>
          </div>
        </div>

        <div className="mt-1 shrink-0">
          {open ? <ChevronUp className="h-5 w-5 text-[#5d767d]" /> : <ChevronDown className="h-5 w-5 text-[#5d767d]" />}
        </div>
      </button>

      {open && (
        <div className="border-t border-[#d8d5c8] px-5 py-4">
          {result.reasoning && (
            <p className="mb-4 text-sm leading-relaxed text-[#3f5c64]">
              <span className="font-semibold text-[#1f3b45]">Reasoning: </span>
              {result.reasoning}
            </p>
          )}

          {result.claim.context && (
            <div className="mb-4 rounded-xl border border-[#d8d5c8] bg-[#f8f6ef] p-3">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.13em] text-[#5f767f]">Context</p>
              <div className="space-y-1 text-xs text-[#405f68]">
                {result.claim.context.normalized_claim && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Normalized: </span>
                    {result.claim.context.normalized_claim}
                  </p>
                )}
                {result.claim.context.enriched_claim_text && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Enriched: </span>
                    {result.claim.context.enriched_claim_text}
                  </p>
                )}
                {result.claim.context.context_summary && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Summary: </span>
                    {result.claim.context.context_summary}
                  </p>
                )}
                {result.claim.context.temporal_context && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Time: </span>
                    {result.claim.context.temporal_context}
                  </p>
                )}
                {result.claim.context.venue_context && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Venue: </span>
                    {result.claim.context.venue_context}
                  </p>
                )}
                {result.claim.context.entity_aliases && result.claim.context.entity_aliases.length > 0 && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Aliases: </span>
                    {result.claim.context.entity_aliases.join(", ")}
                  </p>
                )}
                {result.claim.context.search_hints && result.claim.context.search_hints.length > 0 && (
                  <p>
                    <span className="font-semibold text-[#27444e]">Search hints: </span>
                    {result.claim.context.search_hints.join(", ")}
                  </p>
                )}
                {result.claim.context.context_notes && result.claim.context.context_notes.length > 0 && (
                  <div>
                    <span className="font-semibold text-[#27444e]">Context notes: </span>
                    <ul className="mt-1 list-disc pl-4 text-xs text-[#405f68]">
                      {result.claim.context.context_notes.slice(0, 6).map((note, idx) => (
                        <li key={idx}>
                          <span className="font-semibold text-[#27444e]">{note.entity}:</span> {note.note}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {result.evidence_list.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-[0.13em] text-[#5f767f]">
                Evidence ({result.evidence_list.length})
              </p>
              {result.evidence_list.map((evidence, j) => (
                <EvidenceCard key={j} evidence={evidence} />
              ))}
            </div>
          ) : (
            <p className="text-sm text-[#5d747d]">No evidence collected for this claim.</p>
          )}
        </div>
      )}
    </article>
  );
}

function EvidenceCard({ evidence }: { evidence: NonNullable<VerificationResult["evidence_list"]>[number] }) {
  const credPct = evidence.credibility_score != null ? Math.round(evidence.credibility_score * 100) : null;

  const credColour =
    credPct == null ? "text-[#5f787f]" : credPct >= 70 ? "text-emerald-700" : credPct >= 45 ? "text-amber-700" : "text-rose-700";

  return (
    <article className="rounded-xl border border-[#d8d5c8] bg-[#faf8f1] p-3">
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-semibold leading-snug text-[#20404a]">{evidence.title}</p>
        <a
          href={evidence.url}
          target="_blank"
          rel="noopener noreferrer"
          className="tap-soft shrink-0 text-[var(--brand)] hover:text-[var(--brand-strong)]"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </div>

      <p className="mt-1 text-xs leading-relaxed text-[#4f6971]">{evidence.snippet}</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[#5d757c]">
        <span>Relevance: {Math.round(evidence.relevance_score * 100)}%</span>
        {credPct != null && <span className={`font-semibold ${credColour}`}>Credibility: {credPct}%</span>}
        {evidence.source_quality && <span>Quality: {evidence.source_quality}</span>}
        {evidence.published_date && (
          <span className="inline-flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            {formatDate(evidence.published_date)}
          </span>
        )}
      </div>

      {evidence.credibility_reasoning && (
        <p className="mt-1 text-xs italic leading-relaxed text-[#6c8389]">{evidence.credibility_reasoning}</p>
      )}
    </article>
  );
}
