import Link from "next/link";
import { ShieldCheck, Sparkles, Earth, ArrowUpRight } from "lucide-react";

const METRICS = [
  { label: "Atomic Claims", value: "Per sentence" },
  { label: "Evidence Streams", value: "Live web sources" },
  { label: "Verdict Labels", value: "Supported / Refuted / NEI" },
];

export default function Hero() {
  return (
    <section className="site-section overflow-hidden pb-24 pt-34 sm:pt-38">
      <div className="pointer-events-none absolute -left-20 top-26 h-52 w-52 rounded-full bg-[var(--brand)]/14 blur-3xl" />
      <div className="pointer-events-none absolute -right-8 top-20 h-60 w-60 rounded-full bg-[var(--accent)]/22 blur-3xl" />

      <div className="site-shell grid items-center gap-14 lg:grid-cols-[1.03fr_0.97fr]">
        <div className="animated-rise">
          <span className="brand-pill pulse-soft inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.15em]">
            <Sparkles className="h-3.5 w-3.5" />
            Agentic Fact Verification
          </span>

          <h1 className="mt-5 max-w-3xl text-4xl font-bold leading-[1.02] text-[var(--foreground)] sm:text-5xl lg:text-[4rem]">
            Fact-check fast with
            <span className="text-gradient"> credible evidence</span>, not vibes.
          </h1>

          <p className="section-lead mt-6 max-w-xl">
            ClaimLens decomposes text into verifiable claims, hunts for supporting and
            opposing sources, then runs NLI verification to produce a confidence-grounded
            report in one stream.
          </p>

          <div className="mt-9 flex flex-wrap gap-3">
            <Link
              href="/verify"
              className="btn-primary inline-flex items-center gap-2 rounded-full px-7 py-3 text-sm font-semibold"
            >
              Start Verifying
              <ArrowUpRight className="h-4 w-4" />
            </Link>
            <Link
              href="#architecture"
              className="btn-outline inline-flex items-center gap-2 rounded-full px-7 py-3 text-sm font-semibold"
            >
              Explore Architecture
            </Link>
          </div>

          <div className="mt-10 grid max-w-2xl gap-3 sm:grid-cols-3">
            {METRICS.map((metric, i) => (
              <div
                key={metric.label}
                className="panel animated-rise rounded-2xl p-4"
                style={{ animationDelay: `${120 + i * 80}ms` }}
              >
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#5f747a]">
                  {metric.label}
                </p>
                <p className="mt-1 text-sm font-semibold text-[#173039]">{metric.value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="grain animated-rise stagger-2">
          <div className="glass-card relative rounded-[2rem] p-6 sm:p-7">
            <div className="mb-5 flex items-center justify-between text-xs font-semibold text-[#5d747a]">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--surface-muted)] px-3 py-1">
                <Earth className="h-3.5 w-3.5 text-[var(--brand)]" />
                Live Evidence Trace
              </span>
              <span>Pipeline Snapshot</span>
            </div>

            <div className="space-y-3">
              <TraceRow step="Extract claim" detail="The Eiffel Tower is 330 meters tall." state="done" />
              <TraceRow step="Search evidence" detail="5 sources retrieved with high relevance." state="active" />
              <TraceRow step="Run verifier" detail="DeBERTa NLI infers SUPPORT with 0.96 confidence." state="done" />
            </div>

            <div className="float-soft mt-5 rounded-2xl border border-[#d7d3c8] bg-white/80 p-4">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[#5f747a]">Final verdict</p>
                <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-700">
                  Supported
                </span>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-[#29444d]">
                Evidence from Britannica and official tower sources confirms both the
                height and completion year.
              </p>
              <div className="mt-3 inline-flex items-center gap-2 text-xs font-semibold text-[var(--brand-strong)]">
                <ShieldCheck className="h-4 w-4" />
                Confidence 96%
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function TraceRow({
  step,
  detail,
  state,
}: {
  step: string;
  detail: string;
  state: "done" | "active";
}) {
  return (
    <div className="rounded-2xl border border-[#d7d3c8] bg-white/85 p-4 transition duration-200 hover:-translate-y-0.5 hover:bg-white">
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-semibold text-[#19313a]">{step}</p>
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.12em] ${
            state === "done"
              ? "bg-emerald-100 text-emerald-700"
              : "bg-[var(--brand)]/15 text-[var(--brand-strong)]"
          }`}
        >
          {state}
        </span>
      </div>
      <p className="mt-1 text-xs leading-relaxed text-[#50656c]">{detail}</p>
    </div>
  );
}
