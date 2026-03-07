import { Search } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-gray-200 bg-gray-50">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 py-10 sm:flex-row">
        <div className="flex items-center gap-2 text-lg font-bold text-gray-900">
          <Search className="h-5 w-5 text-indigo-600" />
          ClaimLens
        </div>
        <p className="text-sm text-gray-500">
          AAI3008 — AI-Powered Fact-Checking Pipeline
        </p>
      </div>
    </footer>
  );
}
