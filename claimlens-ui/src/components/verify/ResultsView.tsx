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
} from "lucide-react";
import type { FinalReport, VerificationResult, Verdict } from "@/types/api";
import { VERDICT_CONFIG } from "@/constants/verdicts";

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return dateStr;
  }
}

interface Props {
  report: FinalReport;
  onReset: () => void;
}

export default function ResultsView({ report, onReset }: Props) {
  const trustPct = Math.round(report.overall_trust_score * 100);
  const trustColor =
    trustPct >= 70
      ? "text-emerald-600"
      : trustPct >= 40
        ? "text-amber-600"
        : "text-red-600";
  const trustRing =
    trustPct >= 70
      ? "ring-emerald-200"
      : trustPct >= 40
        ? "ring-amber-200"
        : "ring-red-200";
  const trustBg =
    trustPct >= 70
      ? "bg-emerald-50"
      : trustPct >= 40
        ? "bg-amber-50"
        : "bg-red-50";

  // Verdict counts
  const supported = report.verification_results.filter(
    (r) => r.verdict === "SUPPORTED"
  ).length;
  const refuted = report.verification_results.filter(
    (r) => r.verdict === "REFUTED"
  ).length;
  const nei = report.verification_results.filter(
    (r) => r.verdict === "NOT_ENOUGH_INFO"
  ).length;
  const total = report.verification_results.length;

  return (
    <div className="space-y-6">
      {/* Trust score banner */}
      <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wider text-gray-400">
              Overall Trust Score
            </p>
            <p className={`mt-1 text-4xl font-extrabold ${trustColor}`}>
              {trustPct}%
            </p>
            {report.summary && (
              <p className="mt-2 max-w-md text-sm text-gray-500">
                {report.summary}
              </p>
            )}
            {report.processing_time_seconds != null && (
              <p className="mt-1 text-xs text-gray-400">
                Processed in {report.processing_time_seconds.toFixed(1)}s
              </p>
            )}
          </div>
          <div
            className={`flex h-20 w-20 items-center justify-center rounded-full ${trustBg} ring-4 ${trustRing}`}
          >
            <span className={`text-2xl font-bold ${trustColor}`}>
              {trustPct}
            </span>
          </div>
        </div>

        {/* Verdict summary bar */}
        {total > 0 && (
          <div className="mt-6 border-t border-gray-100 pt-5">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-400">
              Verdict Breakdown
            </p>
            {/* Stacked bar */}
            <div className="mb-3 flex h-3 overflow-hidden rounded-full bg-gray-100">
              {supported > 0 && (
                <div
                  className="bg-emerald-500 transition-all"
                  style={{ width: `${(supported / total) * 100}%` }}
                />
              )}
              {refuted > 0 && (
                <div
                  className="bg-red-500 transition-all"
                  style={{ width: `${(refuted / total) * 100}%` }}
                />
              )}
              {nei > 0 && (
                <div
                  className="bg-gray-400 transition-all"
                  style={{ width: `${(nei / total) * 100}%` }}
                />
              )}
            </div>
            <div className="flex items-center gap-6 text-xs">
              <span className="flex items-center gap-1.5 text-emerald-600">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {supported} Supported
              </span>
              <span className="flex items-center gap-1.5 text-red-600">
                <XCircle className="h-3.5 w-3.5" />
                {refuted} Refuted
              </span>
              <span className="flex items-center gap-1.5 text-gray-500">
                <AlertCircle className="h-3.5 w-3.5" />
                {nei} Not Enough Info
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Result cards */}
      {report.verification_results.map((r, i) => (
        <ResultCard key={r.claim.id ?? i} result={r} index={i} />
      ))}

      {/* Reset */}
      <div className="text-center">
        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 rounded-full border border-gray-300 px-6 py-2.5 text-sm font-semibold text-gray-700 transition hover:bg-gray-100"
        >
          <RotateCcw className="h-4 w-4" />
          Verify Another Text
        </button>
      </div>
    </div>
  );
}

function ResultCard({ result, index }: { result: VerificationResult; index: number }) {
  const [open, setOpen] = useState(false);
  const v = VERDICT_CONFIG[result.verdict as Verdict] ?? VERDICT_CONFIG.NOT_ENOUGH_INFO;
  const Icon = v.icon;
  const confPct = Math.round(result.confidence * 100);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm">
      {/* Header */}
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-start gap-4 p-5 text-left"
      >
        <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${v.bg}`}>
          <Icon className={`h-4 w-4 ${v.text}`} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold text-gray-900">Claim {index + 1}</p>
          <p className="mt-0.5 text-sm text-gray-600">{result.claim.text}</p>
          <div className="mt-2 flex items-center gap-3">
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${v.bg} ${v.text}`}>
              {v.label}
            </span>
            {/* Confidence bar */}
            <div className="h-2 w-24 overflow-hidden rounded-full bg-gray-100">
              <div className={`h-full rounded-full ${v.bar}`} style={{ width: `${confPct}%` }} />
            </div>
            <span className="text-xs font-semibold text-gray-500">{confPct}%</span>
          </div>
        </div>
        <div className="mt-1">
          {open ? (
            <ChevronUp className="h-5 w-5 text-gray-400" />
          ) : (
            <ChevronDown className="h-5 w-5 text-gray-400" />
          )}
        </div>
      </button>

      {/* Expandable evidence */}
      {open && (
        <div className="border-t border-gray-100 px-5 py-4">
          {result.reasoning && (
            <p className="mb-4 text-sm leading-relaxed text-gray-600">
              <span className="font-semibold text-gray-800">Reasoning: </span>
              {result.reasoning}
            </p>
          )}
          {result.evidence_list.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                Evidence ({result.evidence_list.length})
              </p>
              {result.evidence_list.map((e, j) => (
                <div key={j} className="rounded-lg border border-gray-100 bg-gray-50 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs font-semibold text-gray-800">{e.title}</p>
                    <a
                      href={e.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 text-indigo-500 hover:text-indigo-700"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-gray-500">{e.snippet}</p>
                  <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-gray-400">
                    <span>Relevance: {Math.round(e.relevance_score * 100)}%</span>
                    {e.credibility_score != null && (
                      <span className={`font-semibold ${
                        e.credibility_score >= 0.7 ? "text-emerald-500" : e.credibility_score >= 0.4 ? "text-amber-500" : "text-red-500"
                      }`}>
                        Credibility: {Math.round(e.credibility_score * 100)}%
                      </span>
                    )}
                    {e.source_quality && <span>Quality: {e.source_quality}</span>}
                    {e.published_date && (
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {formatDate(e.published_date)}
                      </span>
                    )}
                  </div>
                  {e.credibility_reasoning && (
                    <p className="mt-1 text-xs italic text-gray-400">{e.credibility_reasoning}</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-400">No evidence collected for this claim.</p>
          )}
        </div>
      )}
    </div>
  );
}
