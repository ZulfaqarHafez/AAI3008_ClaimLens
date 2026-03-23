import Link from "next/link";
import { Search, ShieldCheck, Radar } from "lucide-react";

export default function Footer() {
  return (
    <footer className="site-section border-t border-[var(--line)] bg-[var(--surface-muted)]/70">
      <div className="site-shell py-16">
        <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2 text-lg font-bold text-[var(--foreground)]">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--brand)]/14 text-[var(--brand-strong)]">
                <Search className="h-4.5 w-4.5" />
              </span>
              ClaimLens
            </div>
            <p className="mt-3 max-w-sm text-sm leading-relaxed text-[var(--ink-soft)]">
              An agentic fact-checking pipeline that decomposes text into atomic
              claims and verifies each one against web evidence.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <span className="brand-pill inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold">
                <ShieldCheck className="h-3.5 w-3.5" />
                Evidence-grounded
              </span>
              <span className="brand-pill inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold">
                <Radar className="h-3.5 w-3.5" />
                Multi-source verification
              </span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#557078]">
              Quick Links
            </p>
            <ul className="mt-4 space-y-2.5">
              <li>
                <Link href="/" className="text-sm text-[#395059] transition hover:text-[var(--brand-strong)]">
                  Home
                </Link>
              </li>
              <li>
                <Link href="/#how-it-works" className="text-sm text-[#395059] transition hover:text-[var(--brand-strong)]">
                  How It Works
                </Link>
              </li>
              <li>
                <Link href="/#architecture" className="text-sm text-[#395059] transition hover:text-[var(--brand-strong)]">
                  Architecture
                </Link>
              </li>
              <li>
                <Link href="/#features" className="text-sm text-[#395059] transition hover:text-[var(--brand-strong)]">
                  Features
                </Link>
              </li>
            </ul>
          </div>

          {/* Project */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[#557078]">
              Project
            </p>
            <ul className="mt-4 space-y-2.5">
              <li>
                <Link href="/verify" className="text-sm text-[#395059] transition hover:text-[var(--brand-strong)]">
                  Start Verifying
                </Link>
              </li>
              <li>
                <span className="text-sm text-[#395059]">API Docs</span>
              </li>
              <li>
                <span className="text-sm text-[#395059]">AAI3008 LLM Module</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-12 border-t border-[var(--line)] pt-6 text-center text-xs text-[#587177]">
          AAI3008 — AI-Powered Fact-Checking Pipeline
        </div>
      </div>
    </footer>
  );
}
