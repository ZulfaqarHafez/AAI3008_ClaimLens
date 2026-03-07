import Link from "next/link";
import { CheckCircle, Zap, Globe } from "lucide-react";

export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-gradient-to-b from-indigo-50 via-white to-white pt-32 pb-20">
      {/* Decorative blobs */}
      <div className="pointer-events-none absolute -top-40 -right-40 h-[600px] w-[600px] rounded-full bg-indigo-100/50 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-20 -left-40 h-[400px] w-[400px] rounded-full bg-purple-100/40 blur-3xl" />

      <div className="relative mx-auto grid max-w-7xl gap-12 px-6 lg:grid-cols-2 lg:items-center">
        {/* Text */}
        <div>
          <div className="mb-6 inline-flex items-center gap-2 rounded-full bg-indigo-100 px-4 py-1.5 text-sm font-medium text-indigo-700">
            🤖 Powered by DeBERTa NLI + LangGraph
          </div>
          <h1 className="text-4xl font-extrabold leading-tight tracking-tight text-gray-900 sm:text-5xl lg:text-6xl">
            Verify any claim with{" "}
            <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
              AI-powered
            </span>{" "}
            fact-checking
          </h1>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-gray-600">
            Paste any text and ClaimLens will decompose it into atomic claims, search the web for evidence,
            and verify each claim using our fine-tuned NLI model.
          </p>

          <div className="mt-8 flex flex-wrap gap-4">
            <Link
              href="/verify"
              className="rounded-full bg-indigo-600 px-8 py-3 text-base font-semibold text-white shadow-lg shadow-indigo-200 transition hover:bg-indigo-700"
            >
              Start Verifying →
            </Link>
            <Link
              href="#how-it-works"
              className="rounded-full border border-gray-300 px-8 py-3 text-base font-semibold text-gray-700 transition hover:bg-gray-50"
            >
              Learn More
            </Link>
          </div>

          {/* Stat badges */}
          <div className="mt-10 flex flex-wrap gap-4">
            <StatBadge icon={<CheckCircle className="h-5 w-5 text-emerald-500" />} title="3-Label NLI" sub="Support · Refute · NEI" />
            <StatBadge icon={<Zap className="h-5 w-5 text-blue-500" />} title="Real-time" sub="Streaming results" />
            <StatBadge icon={<Globe className="h-5 w-5 text-amber-500" />} title="Multi-source" sub="Web evidence" />
          </div>
        </div>

        {/* Visual card */}
        <div className="hidden lg:block">
          <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-2xl shadow-indigo-100">
            <div className="mb-4 flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-red-400" />
              <span className="h-3 w-3 rounded-full bg-yellow-400" />
              <span className="h-3 w-3 rounded-full bg-green-400" />
              <span className="ml-3 text-sm font-medium text-gray-500">ClaimLens Verification</span>
            </div>
            <div className="mb-5 rounded-lg bg-gray-50 p-4 text-sm italic text-gray-700">
              &ldquo;The Eiffel Tower is 330 meters tall and was completed in 1889.&rdquo;
            </div>
            <ResultRow verdict="Supported" text="Height confirmed at 330m" confidence={98} color="emerald" />
            <ResultRow verdict="Supported" text="Completed March 1889" confidence={95} color="emerald" />
            <ResultRow verdict="Refuted" text="Great Wall not visible from space" confidence={91} color="red" />
          </div>
        </div>
      </div>
    </section>
  );
}

function StatBadge({ icon, title, sub }: { icon: React.ReactNode; title: string; sub: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-gray-200 bg-white px-4 py-2.5 shadow-sm">
      {icon}
      <div>
        <p className="text-sm font-semibold text-gray-900">{title}</p>
        <p className="text-xs text-gray-500">{sub}</p>
      </div>
    </div>
  );
}

function ResultRow({ verdict, text, confidence, color }: { verdict: string; text: string; confidence: number; color: string }) {
  const bg = color === "emerald" ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700";
  return (
    <div className="mb-2 flex items-center justify-between rounded-lg bg-gray-50 px-4 py-2.5 last:mb-0">
      <div className="flex items-center gap-3">
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${bg}`}>{verdict}</span>
        <span className="text-sm text-gray-700">{text}</span>
      </div>
      <span className="text-sm font-semibold text-gray-900">{confidence}%</span>
    </div>
  );
}
