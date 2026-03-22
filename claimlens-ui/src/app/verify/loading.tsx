import { Loader2 } from "lucide-react";

export default function VerifyLoading() {
  return (
    <div className="site-shell flex min-h-[60vh] items-center justify-center">
      <div className="panel rounded-2xl px-8 py-7 text-center">
        <Loader2 className="mx-auto h-8 w-8 animate-spin text-[var(--brand)]" />
        <p className="mt-3 text-sm font-medium text-[#4e666e]">Loading verification workspace...</p>
      </div>
    </div>
  );
}
