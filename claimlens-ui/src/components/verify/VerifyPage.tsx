"use client";

import { useState, useCallback } from "react";
import { Search, ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";
import { verifyTextStream, verifyText, type StreamCallbacks } from "@/lib/api";
import type { FinalReport, Verdict } from "@/types/api";
import ProgressTracker from "@/components/verify/ProgressTracker";
import ResultsView from "@/components/verify/ResultsView";

type Phase = "input" | "loading" | "results";

interface ExtractedClaim {
  id: string;
  text: string;
  status: string;
}

interface VerifiedClaim {
  claim_id: string;
  claim_text: string;
  verdict: Verdict;
  confidence: number;
}

export default function VerifyPage() {
  const [text, setText] = useState("");
  const [phase, setPhase] = useState<Phase>("input");
  const [claims, setClaims] = useState<ExtractedClaim[]>([]);
  const [verified, setVerified] = useState<VerifiedClaim[]>([]);
  const [report, setReport] = useState<FinalReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState("Starting verification...");

  const handleVerify = useCallback(async () => {
    if (!text.trim() || text.trim().length < 10) return;
    setPhase("loading");
    setError(null);
    setClaims([]);
    setVerified([]);
    setReport(null);
    setProgressMsg("Starting verification...");

    const callbacks: StreamCallbacks = {
      onStart: () => setProgressMsg("Decomposing claims..."),
      onClaimsExtracted: (d) => {
        setClaims(d.claims);
        setProgressMsg(`Extracted ${d.count} claim${d.count !== 1 ? "s" : ""}. Verifying...`);
      },
      onClaimVerified: (d) => {
        setVerified((prev) => [...prev, d]);
        setProgressMsg(`Verified: "${d.claim_text.slice(0, 60)}..."`);
      },
      onComplete: () => setProgressMsg("Finalizing report..."),
      onError: (err) => {
        setError(err);
        setPhase("input");
      },
    };

    try {
      // Try streaming first, fall back to sync
      let result: FinalReport | null = null;
      try {
        result = await verifyTextStream(text, callbacks);
      } catch {
        // Streaming failed, try sync
        result = await verifyText(text);
      }

      if (result) {
        setReport(result);
        setPhase("results");
      } else if (!error) {
        setError("Could not retrieve verification results.");
        setPhase("input");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Verification failed");
      setPhase("input");
    }
  }, [text, error]);

  const handleReset = () => {
    setPhase("input");
    setText("");
    setClaims([]);
    setVerified([]);
    setReport(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-24 pb-16">
      <div className="mx-auto max-w-3xl px-6">
        {/* Header */}
        <div className="mb-8 flex items-center gap-4">
          <Link href="/" className="rounded-lg border border-gray-200 p-2 transition hover:bg-gray-100">
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Verify Claims</h1>
            <p className="text-sm text-gray-500">Paste text below and let AI analyze it</p>
          </div>
        </div>

        {/* Input phase */}
        {phase === "input" && (
          <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
            {error && (
              <div className="mb-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}
            <textarea
              className="w-full resize-none rounded-xl border border-gray-200 bg-gray-50 p-4 text-sm leading-relaxed text-gray-800 placeholder:text-gray-400 focus:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              rows={8}
              placeholder="Paste an article, social media post, or any text you want to fact-check..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="mt-4 flex items-center justify-between">
              <span className="text-xs text-gray-400">{text.length} / 10,000 characters</span>
              <button
                onClick={handleVerify}
                disabled={text.trim().length < 10}
                className="flex items-center gap-2 rounded-full bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Search className="h-4 w-4" />
                Verify Claims
              </button>
            </div>
          </div>
        )}

        {/* Loading phase */}
        {phase === "loading" && (
          <ProgressTracker message={progressMsg} claims={claims} verified={verified} />
        )}

        {/* Results phase */}
        {phase === "results" && report && (
          <ResultsView report={report} onReset={handleReset} />
        )}
      </div>
    </div>
  );
}
