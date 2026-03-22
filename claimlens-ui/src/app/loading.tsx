import { Loader2 } from "lucide-react";

export default function Loading() {
  return (
    <div className="site-shell flex min-h-[60vh] items-center justify-center">
      <div className="panel rounded-2xl p-6">
        <Loader2 className="h-8 w-8 animate-spin text-[var(--brand)]" />
      </div>
    </div>
  );
}
