"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X, Search, Sparkles } from "lucide-react";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="fixed left-0 right-0 top-0 z-50 border-b border-[var(--line)]/70 bg-[rgba(246,244,236,0.84)] backdrop-blur-xl">
      <div className="site-shell flex h-18 items-center justify-between">
        <Link href="/" className="group flex items-center gap-2.5 text-[1.1rem] font-bold text-[var(--foreground)]">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--brand)]/12 text-[var(--brand-strong)] transition group-hover:bg-[var(--brand)]/18">
            <Search className="h-4.5 w-4.5" />
          </span>
          <span className="font-[var(--font-display)]">ClaimLens</span>
        </Link>

        {/* Desktop */}
        <div className="hidden items-center gap-8 md:flex">
          <Link href="/#how-it-works" className="text-sm font-semibold text-[#395059] transition hover:text-[var(--brand-strong)]">
            How It Works
          </Link>
          <Link href="/#architecture" className="text-sm font-semibold text-[#395059] transition hover:text-[var(--brand-strong)]">
            Architecture
          </Link>
          <Link href="/#features" className="text-sm font-semibold text-[#395059] transition hover:text-[var(--brand-strong)]">
            Features
          </Link>
          <Link
            href="/verify"
            className="btn-primary tap-soft inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm font-semibold"
          >
            <Sparkles className="h-4 w-4" />
            Start Verifying
          </Link>
        </div>

        {/* Mobile toggle */}
        <button
          className="tap-soft rounded-xl border border-[var(--line)] bg-white/60 p-2 text-[#2f4850] md:hidden"
          onClick={() => setOpen(!open)}
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="site-shell border-t border-[var(--line)] bg-[var(--surface)] pb-4 md:hidden">
          <Link href="/#how-it-works" className="block py-2.5 text-sm font-semibold text-[#395059]" onClick={() => setOpen(false)}>
            How It Works
          </Link>
          <Link href="/#architecture" className="block py-2.5 text-sm font-semibold text-[#395059]" onClick={() => setOpen(false)}>
            Architecture
          </Link>
          <Link href="/#features" className="block py-2.5 text-sm font-semibold text-[#395059]" onClick={() => setOpen(false)}>
            Features
          </Link>
          <Link
            href="/verify"
            className="btn-primary tap-soft mt-3 inline-flex w-full items-center justify-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold"
            onClick={() => setOpen(false)}
          >
            <Sparkles className="h-4 w-4" />
            Start Verifying
          </Link>
        </div>
      )}
    </nav>
  );
}
