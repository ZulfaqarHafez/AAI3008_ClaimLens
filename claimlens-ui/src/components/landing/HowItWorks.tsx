import { FileText, Search, ShieldCheck } from "lucide-react";

const steps = [
  {
    icon: FileText,
    step: "01",
    title: "Decompose",
    description:
      "GPT-4o mini breaks your text into individual, verifiable atomic claims — each one a single factual statement.",
  },
  {
    icon: Search,
    step: "02",
    title: "Search & Gather",
    description:
      "Tavily web searches retrieve real-time evidence for each claim from multiple angles and sources.",
  },
  {
    icon: ShieldCheck,
    step: "03",
    title: "Verify",
    description:
      "Our fine-tuned DeBERTa NLI model cross-checks every claim against evidence and assigns a verdict.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="bg-white py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-indigo-600">
            How It Works
          </p>
          <h2 className="mt-2 text-3xl font-bold text-gray-900 sm:text-4xl">
            Three simple steps
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-gray-500">
            ClaimLens uses an agentic AI pipeline to verify any piece of text in
            seconds.
          </p>
        </div>

        <div className="mt-20 space-y-16 lg:space-y-20">
          {steps.map((s, i) => {
            const Icon = s.icon;
            const reversed = i % 2 !== 0;

            return (
              <div
                key={i}
                className={`flex flex-col items-center gap-10 lg:flex-row lg:gap-16 ${
                  reversed ? "lg:flex-row-reverse" : ""
                }`}
              >
                {/* Icon block */}
                <div className="flex shrink-0 items-center justify-center">
                  <div className="flex h-40 w-40 items-center justify-center rounded-3xl bg-indigo-50 shadow-sm sm:h-48 sm:w-48">
                    <Icon className="h-16 w-16 text-indigo-600 sm:h-20 sm:w-20" />
                  </div>
                </div>

                {/* Text */}
                <div className={`max-w-lg ${reversed ? "lg:text-right" : ""}`}>
                  <span className="text-xs font-bold uppercase tracking-widest text-indigo-600">
                    Step {s.step}
                  </span>
                  <h3 className="mt-2 text-2xl font-bold text-gray-900">
                    {s.title}
                  </h3>
                  <p className="mt-3 text-base leading-relaxed text-gray-500">
                    {s.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
