import { FileText, Search, ShieldCheck } from "lucide-react";

const steps = [
  {
    icon: <FileText className="h-7 w-7 text-indigo-600" />,
    title: "Decompose",
    description: "GPT-4o mini breaks your text into individual, verifiable atomic claims.",
  },
  {
    icon: <Search className="h-7 w-7 text-indigo-600" />,
    title: "Search & Gather",
    description: "Tavily web searches retrieve real-time evidence for each claim.",
  },
  {
    icon: <ShieldCheck className="h-7 w-7 text-indigo-600" />,
    title: "Verify",
    description: "Our fine-tuned DeBERTa NLI model cross-checks every claim against evidence.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="bg-white py-24">
      <div className="mx-auto max-w-7xl px-6 text-center">
        <p className="text-sm font-semibold uppercase tracking-wider text-indigo-600">How It Works</p>
        <h2 className="mt-2 text-3xl font-bold text-gray-900 sm:text-4xl">Three simple steps</h2>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-gray-500">
          ClaimLens uses an agentic AI pipeline to verify any piece of text in seconds.
        </p>

        <div className="mt-16 grid gap-8 sm:grid-cols-3">
          {steps.map((s, i) => (
            <div
              key={i}
              className="relative rounded-2xl border border-gray-200 bg-white p-8 text-left shadow-sm transition hover:shadow-md"
            >
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-xl bg-indigo-50">
                {s.icon}
              </div>
              <span className="absolute top-6 right-6 text-4xl font-extrabold text-gray-100">{i + 1}</span>
              <h3 className="text-lg font-bold text-gray-900">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-gray-500">{s.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
