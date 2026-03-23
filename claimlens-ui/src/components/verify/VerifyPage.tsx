"use client";

import { Search, ArrowLeft, X as XIcon, FileText, Lightbulb, Sparkles } from "lucide-react";
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
    <div className="site-section min-h-screen pb-16 pt-26 sm:pt-31">
      <div className="pointer-events-none absolute left-0 top-12 h-56 w-56 rounded-full bg-[var(--brand)]/10 blur-3xl" />
      <div className="pointer-events-none absolute bottom-16 right-0 h-64 w-64 rounded-full bg-[var(--accent)]/14 blur-3xl" />
      <div className="site-shell max-w-4xl">
        <div className="animated-rise mb-8 flex items-center gap-3 sm:mb-10">
          <Link
            href="/"
            className="tap-soft inline-flex h-10 w-10 items-center justify-center rounded-xl border border-[var(--line)] bg-white/70 text-[#3b565f] transition hover:bg-white"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <p className="section-kicker">Verification Studio</p>
            <h1 className="mt-1 text-3xl font-bold text-[#132f39] sm:text-[2.25rem]">Evidence-first claim analysis</h1>
          </div>
        </div>

        {phase === "input" && (
          <div className="space-y-5">
            <section className="glass-card grain animated-rise stagger-1 rounded-[1.8rem] p-4.5 sm:p-7">
              {error && (
                <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              )}

              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#5c757d]">
                  Paste text to verify
                </p>
                <span className="inline-flex items-center gap-1.5 rounded-full border border-[#c8d5d3] bg-white/80 px-3 py-1 text-[11px] font-semibold text-[#34545d]">
                  <Sparkles className="h-3.5 w-3.5 text-[var(--brand)]" />
                  Live evidence + NLI scoring
                </span>
              </div>

              <textarea
                className="w-full resize-none rounded-2xl border border-[#d8d5c8] bg-[#faf8f2] p-4 text-sm leading-relaxed text-[#233d46] placeholder:text-[#7c8f95]"
                rows={9}
                maxLength={MAX_TEXT_LENGTH}
                placeholder="Paste an article, social post, or paragraph and ClaimLens will break it into verifiable claims..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />

              <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <span className="text-xs font-semibold text-[#637a81]">
                  {text.length} / {MAX_TEXT_LENGTH.toLocaleString()} characters
                </span>
                <button
                  onClick={handleVerify}
                  disabled={text.trim().length < MIN_TEXT_LENGTH}
                  className="btn-primary tap-soft inline-flex items-center justify-center gap-2 rounded-full px-6 py-2.5 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-45"
                >
                  <Search className="h-4 w-4" />
                  Verify Claims
                </button>
              </div>
            </section>

            <section className="panel animated-rise stagger-2 rounded-[1.65rem] p-4.5 sm:p-6">
              <div className="mb-4 flex items-center gap-2">
                <FileText className="h-4 w-4 text-[var(--brand)]" />
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#5c757d]">Try an example</p>
              </div>
              <div className="space-y-2.5">
                {EXAMPLES.map((example, i) => (
                  <button
                    key={i}
                    onClick={() => setText(example)}
                    className="surface-hover tap-soft block w-full rounded-xl border border-[#d8d5c8] bg-[#f8f5eb] px-4 py-3 text-left text-xs leading-relaxed text-[#3f5961] hover:border-[#96b0b1] hover:bg-[#f3f8f7]"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </section>

            <section className="panel animated-rise stagger-3 rounded-[1.65rem] p-4.5 sm:p-6">
              <div className="mb-3 flex items-center gap-2">
                <Lightbulb className="h-4 w-4 text-[var(--accent)]" />
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#5c757d]">Tips for stronger results</p>
              </div>
              <ul className="space-y-2.5 text-xs leading-relaxed text-[#52686f] sm:text-[0.82rem]">
                <li className="flex items-start gap-2">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--brand)]" />
                  Include concrete facts such as names, dates, locations, and measurable values.
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--brand)]" />
                  Add enough context so claims are understandable without external assumptions.
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--brand)]" />
                  Opinion-heavy statements tend to resolve as Not Enough Info.
                </li>
              </ul>
            </section>
          </div>
        )}

        {phase === "loading" && (
          <div className="animated-rise space-y-4">
            <PipelineVisualizer currentNode={currentNode} completedNodes={completedNodes} />
            <ProgressTracker
              message={progressMsg}
              claims={claims}
              verified={verified}
              currentNode={currentNode}
            />
            <div className="text-center">
              <button
                onClick={handleCancel}
                className="btn-outline tap-soft inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm font-semibold"
              >
                <XIcon className="h-4 w-4" />
                Cancel
              </button>
            </div>
          </div>
        )}

        {phase === "results" && report && (
          <div className="animated-rise">
            <ResultsView report={report} onReset={handleReset} />
          </div>
        )}
      </div>
    </div>
  );
}
