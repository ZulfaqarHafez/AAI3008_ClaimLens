import { Loader2, CheckCircle2, Circle } from "lucide-react";
import type { Verdict } from "@/types/api";

interface ExtractedClaim {
  id: string;
  text: string;
  status: string;
}

interface VerifiedClaim {
  claim_id: string;
  claim_text: string;
  verdict: Verdict;
  confidence: number;
}

interface Props {
  message: string;
  claims: ExtractedClaim[];
  verified: VerifiedClaim[];
}

export default function ProgressTracker({ message, claims, verified }: Props) {
  const verifiedIds = new Set(verified.map((v) => v.claim_id));

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
      {/* Spinner + message */}
      <div className="mb-6 flex items-center gap-3">
        <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
        <p className="text-sm font-medium text-gray-700">{message}</p>
      </div>

      {/* Claim list */}
      {claims.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Claims ({verified.length}/{claims.length})
          </p>
          {claims.map((c) => {
            const done = verifiedIds.has(c.id);
            const match = verified.find((v) => v.claim_id === c.id);
            return (
              <div
                key={c.id}
                className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm transition ${
                  done ? "border-emerald-200 bg-emerald-50/50" : "border-gray-100 bg-gray-50"
                }`}
              >
                {done ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-500" />
                ) : (
                  <Circle className="mt-0.5 h-4 w-4 shrink-0 text-gray-300" />
                )}
                <div className="min-w-0 flex-1">
                  <p className={done ? "text-gray-700" : "text-gray-500"}>{c.text}</p>
                  {match && (
                    <p className="mt-1 text-xs text-gray-400">
                      {match.verdict} — {Math.round(match.confidence * 100)}% confidence
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
