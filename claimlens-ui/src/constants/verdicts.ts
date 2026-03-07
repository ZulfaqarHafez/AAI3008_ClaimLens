import { CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import { Verdict } from "@/types/api";

export const VERDICT_CONFIG: Record<
  Verdict,
  {
    icon: typeof CheckCircle2;
    label: string;
    bg: string;
    text: string;
    bar: string;
  }
> = {
  [Verdict.SUPPORTED]: {
    icon: CheckCircle2,
    label: "Supported",
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    bar: "bg-emerald-500",
  },
  [Verdict.REFUTED]: {
    icon: XCircle,
    label: "Refuted",
    bg: "bg-red-50",
    text: "text-red-700",
    bar: "bg-red-500",
  },
  [Verdict.NOT_ENOUGH_INFO]: {
    icon: AlertCircle,
    label: "Not Enough Info",
    bg: "bg-amber-50",
    text: "text-amber-700",
    bar: "bg-amber-500",
  },
};
