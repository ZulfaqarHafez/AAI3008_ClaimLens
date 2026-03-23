import type { ReactNode } from "react";
import {
  FileText,
  Scissors,
  Search,
  Globe,
  ShieldCheck,
  Brain,
  RotateCcw,
  BarChart3,
  FileCheck,
  ArrowDown,
  CheckCircle2,
  XCircle,
  HelpCircle,
} from "lucide-react";

function FlowArrow() {
  return (
    <div className="flex justify-center py-1">
      <ArrowDown className="h-4.5 w-4.5 text-[#5d7b7f]" />
    </div>
  );
}

function TinyTag({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-[#cfd4c8] bg-[#f5f2ea] px-2.5 py-0.5 text-[11px] font-semibold text-[#496168]">
      {children}
    </span>
  );
}

function StepCard({
  icon,
  title,
  subtitle,
  children,
}: {
  icon: ReactNode;
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <article className="panel rounded-2xl p-5 sm:p-6">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--brand)]/14 text-[var(--brand-strong)]">
          {icon}
        </div>
        <div className="min-w-0">
          <h3 className="text-sm font-bold text-[#17323b] sm:text-[0.95rem]">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs font-medium text-[#5d747b]">{subtitle}</p>}
          <div className="mt-2 text-xs leading-relaxed text-[#4b626a] sm:text-[0.82rem]">{children}</div>
        </div>
      </div>
    </article>
  );
}

export default function Architecture() {
  return (
    <section id="architecture" className="site-section py-24 sm:py-28">
      <div className="site-shell">
        <div className="mx-auto max-w-3xl text-center">
          <p className="section-kicker">Architecture</p>
          <h2 className="section-title mt-3">Inside the verification engine.</h2>
          <p className="section-lead mt-4">
            A staged LangGraph pipeline orchestrates retrieval, credibility checks, and
            NLI reasoning to produce inspectable verdicts for each claim.
          </p>
        </div>

        <div className="mx-auto mt-14 flex max-w-4xl flex-col gap-1">
          <StepCard icon={<FileText className="h-5 w-5" />} title="Input Text" subtitle="FastAPI endpoint + streaming updates">
            User submits a paragraph, post, or article to verify.
          </StepCard>

          <FlowArrow />

          <StepCard icon={<Scissors className="h-5 w-5" />} title="Decomposition Agent" subtitle="GPT-4o mini, temperature 0.1">
            Splits input into atomic, independently verifiable statements and removes ambiguity.
            <div className="mt-2 flex flex-wrap gap-1.5">
              <TinyTag>Claim 1</TinyTag>
              <TinyTag>Claim 2</TinyTag>
              <TinyTag>Claim N</TinyTag>
            </div>
          </StepCard>

          <FlowArrow />

          <div className="rounded-[1.7rem] border border-dashed border-[#8ea5a6] bg-[rgba(255,255,255,0.62)] p-4 sm:p-5">
            <p className="mb-3 text-center text-[11px] font-bold uppercase tracking-[0.16em] text-[#446068]">
              Per-claim loop (sequential)
            </p>

            <div className="space-y-1">
              <StepCard icon={<Search className="h-5 w-5" />} title="Search Architect" subtitle="GPT-4o, temperature 0.3">
                Produces diverse supporting, opposing, and contextual query variants. Retry passes refine
                query wording based on evidence gaps.
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <TinyTag>3-5 queries / claim</TinyTag>
                </div>
              </StepCard>

              <FlowArrow />

              <StepCard icon={<Globe className="h-5 w-5" />} title="Scraper Agent" subtitle="Tavily parallel retrieval + relevance filtering">
                Executes searches in parallel, then scores and retains top evidence snippets by relevance.
                <div className="mt-2 flex flex-wrap gap-1.5">
                  <TinyTag>Parallel search</TinyTag>
                  <TinyTag>Top 5 evidence</TinyTag>
                </div>
              </StepCard>

              <FlowArrow />

              <StepCard icon={<ShieldCheck className="h-5 w-5" />} title="Credibility Agent" subtitle="Source trust assessment">
                Evaluates source trustworthiness using author expertise, bias signals, and recency weighting.
                <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-lg border border-[#d4dbcf] bg-[#f6f2e7] p-2">
                    <p className="text-[11px] font-semibold text-[#37545c]">Author</p>
                    <p className="text-[10px] text-[#5b737a]">40%</p>
                  </div>
                  <div className="rounded-lg border border-[#d4dbcf] bg-[#f6f2e7] p-2">
                    <p className="text-[11px] font-semibold text-[#37545c]">Bias</p>
                    <p className="text-[10px] text-[#5b737a]">40%</p>
                  </div>
                  <div className="rounded-lg border border-[#d4dbcf] bg-[#f6f2e7] p-2">
                    <p className="text-[11px] font-semibold text-[#37545c]">Recency</p>
                    <p className="text-[10px] text-[#5b737a]">20%</p>
                  </div>
                </div>
              </StepCard>

              <FlowArrow />

              <StepCard icon={<Brain className="h-5 w-5" />} title="ClaimLens Verifier" subtitle="Fine-tuned DeBERTa-v3 NLI">
                Scores each claim-evidence pair and performs weighted voting using NLI confidence and relevance.
                <div className="mt-3 grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-lg bg-emerald-50 p-2">
                    <CheckCircle2 className="mx-auto h-4 w-4 text-emerald-600" />
                    <p className="mt-1 text-[11px] font-semibold text-emerald-700">Supported</p>
                  </div>
                  <div className="rounded-lg bg-rose-50 p-2">
                    <XCircle className="mx-auto h-4 w-4 text-rose-600" />
                    <p className="mt-1 text-[11px] font-semibold text-rose-700">Refuted</p>
                  </div>
                  <div className="rounded-lg bg-slate-100 p-2">
                    <HelpCircle className="mx-auto h-4 w-4 text-slate-500" />
                    <p className="mt-1 text-[11px] font-semibold text-slate-600">NEI</p>
                  </div>
                </div>
                <p className="mt-2 text-[11px] text-[#5d737a]">
                  LLM cross-check can override specific uncertain patterns when confidence and conflict rules are met.
                </p>
              </StepCard>

              <FlowArrow />

              <div className="mx-auto inline-flex items-center gap-2 rounded-xl border border-amber-300 bg-amber-50 px-3 py-1.5 text-[11px] font-semibold text-amber-700">
                <RotateCcw className="h-3.5 w-3.5" />
                Confidence below threshold triggers a bounded retry loop (max 3 iterations)
              </div>
            </div>
          </div>

          <FlowArrow />

          <StepCard icon={<BarChart3 className="h-5 w-5" />} title="Trust Score Aggregation" subtitle="Weighted summary">
            Blends claim support ratio, average confidence, and evidence quality into a single trust metric.
            <div className="mt-2 rounded-lg border border-[#ced8d0] bg-[#eef4f0] px-3 py-2 font-mono text-[11px] text-[#26474a]">
              Trust = 0.5 support_ratio + 0.3 avg_confidence + 0.2 evidence_quality
            </div>
          </StepCard>

          <FlowArrow />

          <StepCard icon={<FileCheck className="h-5 w-5" />} title="Final Report" subtitle="SSE streamed response">
            Returns per-claim verdicts, source citations, confidence metrics, and a human-readable summary.
          </StepCard>
        </div>
      </div>
    </section>
  );
}
