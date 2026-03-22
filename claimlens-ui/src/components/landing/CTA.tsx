import Link from "next/link";
import { ArrowUpRight } from "lucide-react";

export default function CTA() {
  return (
    <section className="site-section pb-24 pt-8 sm:pb-30">
      <div className="site-shell">
        <div className="grain relative overflow-hidden rounded-[2rem] border border-[#1f4546]/18 bg-[linear-gradient(140deg,#0d5f5b,#0a4744_45%,#153f43)] px-6 py-14 text-center shadow-[0_30px_70px_-38px_rgba(8,40,39,0.95)] sm:px-10">
          <div className="pointer-events-none absolute -left-12 top-6 h-48 w-48 rounded-full bg-[rgba(219,122,47,0.26)] blur-3xl" />
          <div className="pointer-events-none absolute -right-10 bottom-0 h-52 w-52 rounded-full bg-[rgba(255,255,255,0.15)] blur-3xl" />

          <div className="relative">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[#d4ece9]">Ready To Verify</p>
            <h2 className="mt-3 text-3xl font-bold text-[#edf8f7] sm:text-[2.7rem]">
              Bring your next claim, we will bring receipts.
            </h2>
            <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-[#c4dfdc] sm:text-lg">
              Drop in an article, social post, or paragraph and receive an evidence-backed
              breakdown with confidence and traceable sources.
            </p>
            <Link
              href="/verify"
              className="mt-9 inline-flex items-center gap-2 rounded-full bg-[#edf8f7] px-8 py-3 text-sm font-bold text-[#0f4645] transition hover:bg-white"
            >
              Try ClaimLens Now
              <ArrowUpRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
