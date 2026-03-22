import { Brain, Globe, Zap, Lock, BarChart3, RefreshCw } from "lucide-react";

const FEATURES = [
  {
    icon: Brain,
    title: "Fine-tuned DeBERTa NLI",
    desc: "Custom verifier for 3-label natural language inference on claim-checking data.",
  },
  {
    icon: Globe,
    title: "Real-time Web Search",
    desc: "Tavily-powered retrieval gathers fresh evidence from multiple publication angles.",
  },
  {
    icon: Zap,
    title: "LangGraph Orchestration",
    desc: "A deterministic state graph coordinates decomposition, retrieval, and verification.",
  },
  {
    icon: Lock,
    title: "Secure By Design",
    desc: "Sanitized errors and robust API boundaries keep sensitive user input protected.",
  },
  {
    icon: BarChart3,
    title: "Trust Score Summary",
    desc: "An aggregate trust metric combines verdict ratio, confidence, and evidence quality.",
  },
  {
    icon: RefreshCw,
    title: "Iterative Search Loop",
    desc: "Low-confidence claims trigger another evidence pass with refined search queries.",
  },
];

export default function Features() {
  return (
    <section id="features" className="site-section py-24 sm:py-28">
      <div className="site-shell">
        <div className="mx-auto max-w-2xl text-center">
          <p className="section-kicker">Features</p>
          <h2 className="section-title mt-3">Engineered for trust, not just speed.</h2>
          <p className="section-lead mt-4">
            Every module is optimized for explainability and signal quality so results
            stay auditable from claim to conclusion.
          </p>
        </div>

        <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((feature, i) => {
            const Icon = feature.icon;

            return (
              <article
                key={feature.title}
                className="panel animated-rise group relative overflow-hidden rounded-[1.6rem] p-6"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <div className="pointer-events-none absolute -right-12 -top-12 h-28 w-28 rounded-full bg-[var(--brand)]/8 blur-2xl transition group-hover:bg-[var(--accent)]/14" />
                <div className="relative">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-[var(--brand)]/12 text-[var(--brand-strong)]">
                    <Icon className="h-6 w-6" />
                  </div>
                  <h3 className="text-lg font-bold text-[#12303a]">{feature.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-[#4f656d]">{feature.desc}</p>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
