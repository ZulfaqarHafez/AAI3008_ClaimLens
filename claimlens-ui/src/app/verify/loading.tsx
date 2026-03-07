import { Loader2 } from "lucide-react";

export default function VerifyLoading() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="text-center">
        <Loader2 className="mx-auto h-8 w-8 animate-spin text-indigo-600" />
        <p className="mt-3 text-sm text-gray-500">Loading verification page...</p>
      </div>
    </div>
  );
}
