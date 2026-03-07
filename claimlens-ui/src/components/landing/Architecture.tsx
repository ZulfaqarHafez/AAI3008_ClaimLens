import {
  FileText,
  Scissors,
  Search,
  Globe,
  Filter,
  Brain,
  RotateCcw,
  BarChart3,
  FileCheck,
  ArrowDown,
  CheckCircle2,
  XCircle,
  HelpCircle,
} from "lucide-react";

/* ── tiny helpers ───────────────────────────────────────── */

function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <ArrowDown className="h-5 w-5 text-indigo-400" />
    </div>
  );
}

function RetryArrow() {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700">
      <RotateCcw className="h-3.5 w-3.5" />
      Confidence &lt; 0.7 → Retry with refined queries (max 3 iterations)
    </div>
  );
}

function Tag({ children, color = "indigo" }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    indigo: "bg-indigo-100 text-indigo-700",
    emerald: "bg-emerald-100 text-emerald-700",
    amber: "bg-amber-100 text-amber-700",
    rose: "bg-rose-100 text-rose-700",
    slate: "bg-slate-100 text-slate-700",
    sky: "bg-sky-100 text-sky-700",
  };
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${colors[color] ?? colors.indigo}`}>
      {children}
    </span>
  );
}

/* ── pipeline step card ─────────────────────────────────── */

interface StepProps {
  icon: React.ReactNode;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  accent?: string;
}

function Step({ icon, title, subtitle, children, accent = "indigo" }: StepProps) {
  const ring: Record<string, string> = {
    indigo: "border-indigo-200 bg-white",
    emerald: "border-emerald-200 bg-emerald-50/40",
    amber: "border-amber-200 bg-amber-50/40",
    sky: "border-sky-200 bg-sky-50/40",
  };
  const iconBg: Record<string, string> = {
    indigo: "bg-indigo-100 text-indigo-600",
    emerald: "bg-emerald-100 text-emerald-600",
    amber: "bg-amber-100 text-amber-600",
    sky: "bg-sky-100 text-sky-600",
  };

  return (
    <div className={`rounded-2xl border ${ring[accent] ?? ring.indigo} p-5 shadow-sm`}>
      <div className="flex items-start gap-3">
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${iconBg[accent] ?? iconBg.indigo}`}>
          {icon}
        </div>
        <div className="min-w-0">
          <h4 className="text-sm font-bold text-gray-900">{title}</h4>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
          <div className="mt-2 text-xs leading-relaxed text-gray-600">{children}</div>
        </div>
      </div>
    </div>
  );
}

/* ── main component ─────────────────────────────────────── */

export default function Architecture() {
  return (
    <section id="architecture" className="bg-white py-24">
      <div className="mx-auto max-w-4xl px-6">
        {/* heading */}
        <div className="text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-indigo-600">
            System Architecture
          </p>
          <h2 className="mt-2 text-3xl font-bold text-gray-900 sm:text-4xl">
            How ClaimLens Works Under&nbsp;the&nbsp;Hood
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-gray-500">
            An agentic pipeline powered by LangGraph orchestrates four specialised agents
            to decompose, search, filter, and verify every claim.
          </p>
        </div>

        {/* pipeline flow */}
        <div className="mt-14 flex flex-col items-center gap-0">
          {/* 1 – Input */}
          <Step icon={<FileText className="h-5 w-5" />} title="Input Text" accent="sky">
            <p>User submits a paragraph or article to be fact-checked.</p>
            <Tag color="sky">FastAPI + SSE Streaming</Tag>
          </Step>

          <Arrow />

          {/* 2 – Decomposition */}
          <Step
            icon={<Scissors className="h-5 w-5" />}
            title="Decomposition Agent"
            subtitle="GPT-4o-mini  ·  Temperature 0.1"
            accent="indigo"
          >
            <p>
              Breaks the input into <strong>atomic, independently verifiable claims</strong>.
              Each claim is a single factual statement with no pronouns or ambiguity.
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <Tag>Claim 1</Tag>
              <Tag>Claim 2</Tag>
              <Tag>Claim N</Tag>
            </div>
          </Step>

          <Arrow />

          {/* 3 – Per-claim loop label */}
          <div className="w-full rounded-xl border-2 border-dashed border-indigo-300 bg-indigo-50/50 p-5">
            <p className="mb-3 text-center text-xs font-bold uppercase tracking-wider text-indigo-600">
              For each claim (sequential processing)
            </p>

            {/* 3a – Search Architect */}
            <Step
              icon={<Search className="h-5 w-5" />}
              title="Search Architect Agent"
              subtitle="GPT-4o-mini  ·  Temperature 0.3"
              accent="indigo"
            >
              <p>
                Generates <strong>diverse search queries</strong> from multiple angles —
                supporting, opposing, and contextual — to maximise evidence coverage.
                On retries, produces <strong>refined queries</strong> informed by
                existing evidence gaps.
              </p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <Tag color="slate">3–5 queries per claim</Tag>
              </div>
            </Step>

            <Arrow />

            {/* 3b – Scraper */}
            <Step
              icon={<Globe className="h-5 w-5" />}
              title="Scraper Agent"
              subtitle="Tavily Search API  ·  Parallel Execution"
              accent="emerald"
            >
              <p>
                Executes queries in <strong>parallel</strong> via the Tavily search API
                (advanced depth). Then applies a <strong>GPT-4o-mini relevance filter</strong>{" "}
                that scores each result&apos;s relevance and source quality (high / medium / low).
              </p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <Tag color="emerald">Parallel Search</Tag>
                <Tag color="emerald">LLM Relevance Filter</Tag>
                <Tag color="slate">Top-5 evidence kept</Tag>
              </div>
            </Step>

            <Arrow />

            {/* 3c – Verifier */}
            <Step
              icon={<Brain className="h-5 w-5" />}
              title="ClaimLens Verifier"
              subtitle="Fine-tuned DeBERTa-v3 NLI Model"
              accent="amber"
            >
              <p>
                Runs each <em>(claim, evidence)</em> pair through{" "}
                <strong>Zulfhagez/claimlens-deberta-v3-nli</strong>, a fine-tuned
                DeBERTa-v3-base model for 3-label natural language inference.
              </p>
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
                  <p className="mt-1 text-[11px] font-semibold text-slate-600">Not Enough Info</p>
                </div>
              </div>
              <p className="mt-3 text-[11px] text-gray-500">
                <strong>Weighted voting:</strong> each evidence piece&apos;s NLI confidence is
                weighted by its relevance score. The label with the highest weighted sum wins.
              </p>
            </Step>

            <Arrow />

            {/* 3d – Retry */}
            <div className="flex justify-center">
              <RetryArrow />
            </div>
          </div>

          <Arrow />

          {/* 4 – Aggregation */}
          <Step
            icon={<BarChart3 className="h-5 w-5" />}
            title="Trust Score Aggregation"
            accent="indigo"
          >
            <p>
              Computes an overall <strong>trust score</strong> for the original text using a
              weighted formula:
            </p>
            <div className="mt-2 overflow-x-auto rounded-lg bg-gray-50 px-3 py-2 font-mono text-[11px] text-gray-700">
              Trust = 0.5 × support_ratio + 0.3 × avg_confidence + 0.2 × evidence_quality
            </div>
            <p className="mt-2">
              Refuted claims are penalised 0.5× in the support ratio. Evidence quality is
              derived from source-quality labels (high&nbsp;= 1.0, medium&nbsp;= 0.6, low&nbsp;= 0.3).
            </p>
          </Step>

          <Arrow />

          {/* 5 – Report */}
          <Step
            icon={<FileCheck className="h-5 w-5" />}
            title="Final Report"
            subtitle="LLM-generated summary"
            accent="sky"
          >
            <p>
              A structured report containing per-claim verdicts, evidence citations,
              confidence scores, and a human-readable summary — streamed to the
              frontend in real time via <strong>Server-Sent Events</strong>.
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <Tag color="sky">SSE Streaming</Tag>
              <Tag color="sky">Real-time UI Updates</Tag>
            </div>
          </Step>
        </div>

        {/* tech stack summary */}
        <div className="mt-14 rounded-2xl border border-gray-200 bg-gray-50 p-6">
          <h3 className="text-center text-sm font-bold uppercase tracking-wider text-gray-600">
            Technology Stack
          </h3>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: "Orchestration", value: "LangGraph State Machine" },
              { label: "LLM", value: "GPT-4o-mini (OpenAI)" },
              { label: "NLI Model", value: "DeBERTa-v3 (Fine-tuned)" },
              { label: "Search", value: "Tavily API" },
              { label: "Backend", value: "FastAPI + SSE" },
              { label: "Frontend", value: "Next.js + Tailwind CSS" },
              { label: "ML Framework", value: "PyTorch + HuggingFace" },
              { label: "Streaming", value: "Server-Sent Events" },
            ].map((t) => (
              <div key={t.label} className="text-center">
                <p className="text-[11px] font-semibold uppercase text-gray-400">{t.label}</p>
                <p className="mt-0.5 text-sm font-medium text-gray-800">{t.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
