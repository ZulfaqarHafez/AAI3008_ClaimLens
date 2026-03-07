"use client";

import { AlertCircle } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-6 text-center">
      <AlertCircle className="mb-4 h-12 w-12 text-red-400" />
      <h2 className="text-2xl font-bold text-gray-900">Something went wrong</h2>
      <p className="mt-2 max-w-md text-sm text-gray-500">
        {error.message || "An unexpected error occurred. Please try again."}
      </p>
      <button
        onClick={reset}
        className="mt-6 rounded-full bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700"
      >
        Try Again
      </button>
    </div>
  );
}
