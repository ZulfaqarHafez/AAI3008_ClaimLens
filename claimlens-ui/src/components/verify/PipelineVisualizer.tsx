"use client";

import {
  FileText,
  Search,
  Globe,
  ShieldCheck,
  Brain,
  CheckCircle2,
  BarChart3,
  FileBarChart,
  Loader2,
  ArrowRight,
  BookOpen,
} from "lucide-react";
import type { PipelineNode } from "@/types/api";

interface Props {
  currentNode: PipelineNode | null;
  completedNodes: PipelineNode[];
}

const PIPELINE_STEPS: {
  node: PipelineNode;
  label: string;
  icon: typeof FileText;
  description: string;
}[] = [
  {
    node: "decompose_claims",
    label: "Decompose",
    icon: FileText,
    description: "Breaking text into claims",
  },
  {
    node: "enrich_context",
    label: "Context",
    icon: BookOpen,
    description: "Gathering background context",
  },
  {
    node: "generate_queries",
    label: "Search",
    icon: Search,
    description: "Generating search queries",
  },
  {
    node: "search_evidence",
    label: "Gather",
    icon: Globe,
    description: "Collecting web evidence",
  },
  {
    node: "assess_credibility",
    label: "Credibility",
    icon: ShieldCheck,
    description: "Assessing source credibility",
  },
  {
    node: "verify_claim",
    label: "Verify",
    icon: Brain,
    description: "Running NLI model",
  },
  {
    node: "aggregate_results",
    label: "Aggregate",
    icon: BarChart3,
    description: "Combining results",
  },
  {
    node: "generate_report",
    label: "Report",
    icon: FileBarChart,
    description: "Generating final report",
  },
];

const NODE_TO_STEP: Record<string, PipelineNode> = {
  decompose_claims: "decompose_claims",
  prepare_claim: "enrich_context",
  enrich_context: "enrich_context",
  frame_claim: "enrich_context",
  generate_queries: "generate_queries",
  search_evidence: "search_evidence",
  frame_evidence: "search_evidence",
  assess_credibility: "assess_credibility",
  verify_claim: "verify_claim",
  finalize_claim: "verify_claim",
  aggregate_results: "aggregate_results",
  generate_report: "generate_report",
};

function getStepStatus(
  step: PipelineNode,
  currentNode: PipelineNode | null,
  completedNodes: PipelineNode[]
): "idle" | "active" | "done" {
  const mappedCurrent = currentNode ? NODE_TO_STEP[currentNode] : null;
  const mappedCompleted = completedNodes.map((n) => NODE_TO_STEP[n]);

  if (mappedCompleted.includes(step)) return "done";
  if (mappedCurrent === step) return "active";

  return "idle";
}

export default function PipelineVisualizer({ currentNode, completedNodes }: Props) {
  return (
    <section className="panel rounded-[1.6rem] p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#5e757c]">Pipeline Progress</p>
        <span className="rounded-full border border-[#cad8d5] bg-[#eef8f6] px-3 py-1 text-[11px] font-semibold text-[#2f5559]">
          Live status
        </span>
      </div>

      <div className="overflow-x-auto">
        <div className="flex min-w-[740px] items-center justify-between gap-1 pb-1">
          {PIPELINE_STEPS.map((step, i) => {
            const status = getStepStatus(step.node, currentNode, completedNodes);
            const Icon = step.icon;

            return (
              <div key={step.node} className="flex items-center gap-1">
                <div className="flex flex-col items-center gap-1.5 text-center">
                  <div
                    className={`flex h-10 w-10 items-center justify-center rounded-xl border transition-all duration-300 ${
                      status === "active"
                        ? "border-[var(--brand)]/40 bg-[#dbf0ec]"
                        : status === "done"
                          ? "border-emerald-300 bg-emerald-100"
                          : "border-[#d6d8cd] bg-[#f4f2e9]"
                    }`}
                  >
                    {status === "active" ? (
                      <Loader2 className="h-4.5 w-4.5 animate-spin text-[var(--brand-strong)]" />
                    ) : status === "done" ? (
                      <CheckCircle2 className="h-4.5 w-4.5 text-emerald-700" />
                    ) : (
                      <Icon className="h-4.5 w-4.5 text-[#70858a]" />
                    )}
                  </div>
                  <p
                    className={`text-[10px] font-bold uppercase tracking-[0.1em] ${
                      status === "active"
                        ? "text-[var(--brand-strong)]"
                        : status === "done"
                          ? "text-emerald-700"
                          : "text-[#6f8187]"
                    }`}
                  >
                    {step.label}
                  </p>
                  <p className="max-w-[76px] text-[9px] leading-tight text-[#6f858b]">{step.description}</p>
                </div>

                {i < PIPELINE_STEPS.length - 1 && (
                  <ArrowRight
                    className={`mx-0.5 h-3.5 w-3.5 shrink-0 ${
                      status === "done" ? "text-emerald-400" : "text-[#b4c0bf]"
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
