"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X, Search } from "lucide-react";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-gray-200/60 bg-white/80 backdrop-blur-lg">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2 text-xl font-bold text-gray-900">
          <Search className="h-6 w-6 text-indigo-600" />
          ClaimLens
        </Link>

        {/* Desktop */}
        <div className="hidden items-center gap-8 md:flex">
          <Link href="/#how-it-works" className="text-sm font-medium text-gray-600 transition hover:text-gray-900">
            How It Works
          </Link>
          <Link href="/#features" className="text-sm font-medium text-gray-600 transition hover:text-gray-900">
            Features
          </Link>
          <Link
            href="/verify"
            className="rounded-full bg-indigo-600 px-5 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700"
          >
            Start Verifying →
          </Link>
        </div>

        {/* Mobile toggle */}
        <button className="md:hidden" onClick={() => setOpen(!open)} aria-label="Toggle menu">
          {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="border-b border-gray-200 bg-white px-6 pb-4 md:hidden">
          <Link href="/#how-it-works" className="block py-2 text-sm font-medium text-gray-600" onClick={() => setOpen(false)}>
            How It Works
          </Link>
          <Link href="/#features" className="block py-2 text-sm font-medium text-gray-600" onClick={() => setOpen(false)}>
            Features
          </Link>
          <Link
            href="/verify"
            className="mt-2 block rounded-full bg-indigo-600 px-5 py-2 text-center text-sm font-semibold text-white"
            onClick={() => setOpen(false)}
          >
            Start Verifying →
          </Link>
        </div>
      )}
    </nav>
  );
}
