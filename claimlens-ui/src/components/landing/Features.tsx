import { Brain, Globe, Zap, Lock, BarChart3, RefreshCw } from "lucide-react";

const features = [
  {
    icon: <Brain className="h-6 w-6 text-indigo-600" />,
    title: "Fine-tuned DeBERTa NLI",
    desc: "Custom model trained for 3-label natural language inference on fact-checking data.",
  },
  {
    icon: <Globe className="h-6 w-6 text-indigo-600" />,
    title: "Real-time Web Search",
    desc: "Tavily-powered web searches retrieve fresh, relevant evidence for every claim.",
  },
  {
    icon: <Zap className="h-6 w-6 text-indigo-600" />,
    title: "LangGraph Orchestration",
    desc: "Agentic state machine coordinates decomposition, search, and verification.",
  },
  {
    icon: <Lock className="h-6 w-6 text-indigo-600" />,
    title: "Secure & Private",
    desc: "API-key auth, rate limiting, and sanitized errors keep your data safe.",
  },
  {
    icon: <BarChart3 className="h-6 w-6 text-indigo-600" />,
    title: "Trust Score",
    desc: "Weighted aggregate score reflecting overall text trustworthiness.",
  },
  {
    icon: <RefreshCw className="h-6 w-6 text-indigo-600" />,
    title: "Iterative Verification",
    desc: "Claims are re-searched if confidence is low, improving accuracy.",
  },
];

export default function Features() {
  return (
    <section id="features" className="bg-gray-50 py-24">
      <div className="mx-auto max-w-7xl px-6 text-center">
        <p className="text-sm font-semibold uppercase tracking-wider text-indigo-600">Features</p>
        <h2 className="mt-2 text-3xl font-bold text-gray-900 sm:text-4xl">Built for reliable fact-checking</h2>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-gray-500">
          Every piece of the pipeline is designed for accuracy, speed, and transparency.
        </p>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <div key={i} className="rounded-2xl border border-gray-200 bg-white p-6 text-left shadow-sm transition hover:shadow-md">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-50">{f.icon}</div>
              <h3 className="text-base font-bold text-gray-900">{f.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-gray-500">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
