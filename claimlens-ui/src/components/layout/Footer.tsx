import Link from "next/link";
import { Search } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-gray-200 bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 py-14">
        <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2 text-lg font-bold text-gray-900">
              <Search className="h-5 w-5 text-indigo-600" />
              ClaimLens
            </div>
            <p className="mt-3 max-w-xs text-sm leading-relaxed text-gray-500">
              An agentic fact-checking pipeline that decomposes text into atomic
              claims and verifies each one against web evidence.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
              Quick Links
            </p>
            <ul className="mt-4 space-y-2.5">
              <li>
                <Link href="/" className="text-sm text-gray-600 transition hover:text-gray-900">
                  Home
                </Link>
              </li>
              <li>
                <Link href="/#how-it-works" className="text-sm text-gray-600 transition hover:text-gray-900">
                  How It Works
                </Link>
              </li>
              <li>
                <Link href="/#architecture" className="text-sm text-gray-600 transition hover:text-gray-900">
                  Architecture
                </Link>
              </li>
              <li>
                <Link href="/#features" className="text-sm text-gray-600 transition hover:text-gray-900">
                  Features
                </Link>
              </li>
            </ul>
          </div>

          {/* Project */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
              Project
            </p>
            <ul className="mt-4 space-y-2.5">
              <li>
                <Link href="/verify" className="text-sm text-gray-600 transition hover:text-gray-900">
                  Start Verifying
                </Link>
              </li>
              <li>
                <span className="text-sm text-gray-600">API Docs</span>
              </li>
              <li>
                <span className="text-sm text-gray-600">AAI3008 LLM Module</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-12 border-t border-gray-200 pt-6 text-center text-xs text-gray-400">
          AAI3008 — AI-Powered Fact-Checking Pipeline
        </div>
      </div>
    </footer>
  );
}
