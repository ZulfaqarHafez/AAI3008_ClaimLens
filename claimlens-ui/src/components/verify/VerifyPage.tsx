"use client";

import { Search, ArrowLeft, X as XIcon, FileText, Lightbulb } from "lucide-react";
import Link from "next/link";
import { useVerification } from "@/hooks/useVerification";
import { MIN_TEXT_LENGTH, MAX_TEXT_LENGTH } from "@/constants/validation";
import PipelineVisualizer from "@/components/verify/PipelineVisualizer";
import ProgressTracker from "@/components/verify/ProgressTracker";
import ResultsView from "@/components/verify/ResultsView";

const EXAMPLES = [
  "The Great Wall of China is visible from space with the naked eye. It was built over 2,000 years ago and stretches for exactly 10,000 miles.",
  "Python is the most popular programming language in the world. It was created by Guido van Rossum and first released in 1991.",
  "The human body has 206 bones. Babies are born with approximately 270 bones, which fuse together as they grow.",
];

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
          <Link
            href="/"
            className="rounded-lg border border-gray-200 p-2 transition hover:bg-gray-100"
          >
            <ArrowLeft className="h-5 w-5 text-gray-600" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Verify Claims
            </h1>
            <p className="text-sm text-gray-500">
              Paste text below and let AI analyze it
            </p>
          </div>
        </div>

        {/* Input phase */}
        {phase === "input" && (
          <div className="space-y-5">
            {/* Main input card */}
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
                <span className="text-xs text-gray-400">
                  {text.length} / {MAX_TEXT_LENGTH.toLocaleString()} characters
                </span>
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

            {/* Example texts */}
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex items-center gap-2">
                <FileText className="h-4 w-4 text-indigo-600" />
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                  Try an example
                </p>
              </div>
              <div className="space-y-2.5">
                {EXAMPLES.map((ex, i) => (
                  <button
                    key={i}
                    onClick={() => setText(ex)}
                    className="block w-full rounded-lg border border-gray-100 bg-gray-50 px-4 py-3 text-left text-xs leading-relaxed text-gray-600 transition hover:border-indigo-200 hover:bg-indigo-50/50"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            {/* Tips */}
            <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
              <div className="mb-3 flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-amber-500" />
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
                  Tips for best results
                </p>
              </div>
              <ul className="space-y-2 text-xs leading-relaxed text-gray-500">
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
                  Include specific factual claims — dates, numbers, names, and
                  measurable statements work best.
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
                  Longer paragraphs with multiple claims give a more comprehensive
                  trust score.
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
                  Opinions and subjective statements will be flagged as &ldquo;Not
                  Enough Info&rdquo; since they cannot be verified.
                </li>
              </ul>
            </div>
          </div>
        )}

        {/* Loading phase */}
        {phase === "loading" && (
          <div className="space-y-4">
            <PipelineVisualizer
              currentNode={currentNode}
              completedNodes={completedNodes}
            />
            <ProgressTracker
              message={progressMsg}
              claims={claims}
              verified={verified}
              currentNode={currentNode}
            />
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
