"use client";

import { Search, ArrowLeft, X as XIcon } from "lucide-react";
import Link from "next/link";
import { useVerification } from "@/hooks/useVerification";
import { MIN_TEXT_LENGTH, MAX_TEXT_LENGTH } from "@/constants/validation";
import PipelineVisualizer from "@/components/verify/PipelineVisualizer";
import ProgressTracker from "@/components/verify/ProgressTracker";
import ResultsView from "@/components/verify/ResultsView";

export default function VerifyPage() {
  const {
    text,
    setText,
    phase,
    claims,
    verified,
    report,
    error,
    progressMsg,
    currentNode,
    completedNodes,
    handleVerify,
    handleCancel,
    handleReset,
  } = useVerification();

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
              maxLength={MAX_TEXT_LENGTH}
              placeholder="Paste an article, social media post, or any text you want to fact-check..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="mt-4 flex items-center justify-between">
              <span className="text-xs text-gray-400">{text.length} / {MAX_TEXT_LENGTH.toLocaleString()} characters</span>
              <button
                onClick={handleVerify}
                disabled={text.trim().length < MIN_TEXT_LENGTH}
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
          <div className="space-y-4">
            <PipelineVisualizer currentNode={currentNode} completedNodes={completedNodes} />
            <ProgressTracker message={progressMsg} claims={claims} verified={verified} />
            <div className="text-center">
              <button
                onClick={handleCancel}
                className="inline-flex items-center gap-2 rounded-full border border-gray-300 px-5 py-2 text-sm font-semibold text-gray-600 transition hover:bg-gray-100"
              >
                <XIcon className="h-4 w-4" />
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Results phase */}
        {phase === "results" && report && (
          <ResultsView report={report} onReset={handleReset} />
        )}
      </div>
    </div>
  );
}
