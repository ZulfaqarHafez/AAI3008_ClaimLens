import { FileText, Search, ShieldCheck } from "lucide-react";

const STEPS = [
  {
    icon: FileText,
    step: "01",
    title: "Decompose Into Atomic Claims",
    description:
      "GPT-4o mini separates factual statements from opinion language and rewrites each claim into a directly testable unit.",
    detail: "No hidden pronouns, no bundled assumptions.",
  },
  {
    icon: Search,
    step: "02",
    title: "Search Across Multiple Angles",
    description:
      "ClaimLens generates diverse queries, fetches sources in parallel, and keeps only evidence with meaningful relevance.",
    detail: "Supports and contradictions are captured side-by-side.",
  },
  {
    icon: ShieldCheck,
    step: "03",
    title: "Verify And Score Confidence",
    description:
      "The verifier evaluates each claim-evidence pair, assigns verdict labels, and combines confidence with source quality.",
    detail: "You get claim-level reasoning and an aggregate trust score.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="site-section py-24 sm:py-28">
      <div className="site-shell">
        <div className="mx-auto max-w-2xl text-center">
          <p className="section-kicker">How It Works</p>
          <h2 className="section-title mt-3">Three deliberate steps, one clear verdict.</h2>
          <p className="section-lead mt-4">
            Each stage is purpose-built so verification stays transparent, traceable,
            and easy to inspect.
          </p>
        </div>

        <div className="mt-14 grid gap-6 lg:mt-16">
          {STEPS.map((step, i) => {
            const Icon = step.icon;
            const reversed = i % 2 === 1;

            return (
              <article
                key={step.step}
                className={`panel animated-rise rounded-[1.75rem] p-5 sm:p-7 ${
                  reversed ? "lg:ml-18" : "lg:mr-18"
                }`}
              >
                <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
                  <div className="flex items-start gap-4">
                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-[var(--brand)]/12 text-[var(--brand-strong)]">
                      <Icon className="h-7 w-7" />
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#557078]">
                        Step {step.step}
                      </p>
                      <h3 className="mt-1 text-xl font-bold text-[#15313a] sm:text-2xl">{step.title}</h3>
                      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-[#4f656d] sm:text-base">
                        {step.description}
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-[#d7d3c8] bg-[var(--surface-muted)]/65 px-4 py-3 text-sm font-semibold text-[#2d4a53] sm:min-w-64">
                    {step.detail}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
