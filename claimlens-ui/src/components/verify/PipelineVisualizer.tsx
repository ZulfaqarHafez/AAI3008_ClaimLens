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
  Circle,
  ArrowRight,
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

// Map intermediate nodes to the visible step they belong to
const NODE_TO_STEP: Record<string, PipelineNode> = {
  decompose_claims: "decompose_claims",
  prepare_claim: "generate_queries",
  generate_queries: "generate_queries",
  search_evidence: "search_evidence",
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

  // Check if this step's node (or any node mapped to it) is completed
  const mappedCompleted = completedNodes.map((n) => NODE_TO_STEP[n]);
  if (mappedCompleted.includes(step)) return "done";

  if (mappedCurrent === step) return "active";

  return "idle";
}

export default function PipelineVisualizer({ currentNode, completedNodes }: Props) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
      <p className="mb-4 text-xs font-semibold uppercase tracking-wider text-gray-400">
        Pipeline Progress
      </p>
      <div className="flex items-center justify-between gap-1">
        {PIPELINE_STEPS.map((step, i) => {
          const status = getStepStatus(step.node, currentNode, completedNodes);
          const Icon = step.icon;

          return (
            <div key={step.node} className="flex items-center gap-1">
              {/* Step */}
              <div className="flex flex-col items-center gap-1.5">
                <div
                  className={`flex h-10 w-10 items-center justify-center rounded-xl transition-all duration-300 ${
                    status === "active"
                      ? "bg-indigo-100 ring-2 ring-indigo-400 ring-offset-1"
                      : status === "done"
                        ? "bg-emerald-100"
                        : "bg-gray-100"
                  }`}
                >
                  {status === "active" ? (
                    <Loader2 className="h-4.5 w-4.5 animate-spin text-indigo-600" />
                  ) : status === "done" ? (
                    <CheckCircle2 className="h-4.5 w-4.5 text-emerald-600" />
                  ) : (
                    <Icon
                      className={`h-4.5 w-4.5 ${
                        status === "idle" ? "text-gray-400" : "text-gray-600"
                      }`}
                    />
                  )}
                </div>
                <span
                  className={`text-[10px] font-semibold leading-none ${
                    status === "active"
                      ? "text-indigo-600"
                      : status === "done"
                        ? "text-emerald-600"
                        : "text-gray-400"
                  }`}
                >
                  {step.label}
                </span>
                {status === "active" && (
                  <span className="mt-0.5 text-[9px] text-indigo-400 animate-pulse max-w-[80px] text-center leading-tight">
                    {step.description}
                  </span>
                )}
              </div>

              {/* Arrow connector */}
              {i < PIPELINE_STEPS.length - 1 && (
                <ArrowRight
                  className={`mx-0.5 h-3.5 w-3.5 shrink-0 ${
                    status === "done" ? "text-emerald-300" : "text-gray-200"
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
