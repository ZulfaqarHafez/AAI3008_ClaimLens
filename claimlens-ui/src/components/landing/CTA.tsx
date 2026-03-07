import Link from "next/link";

export default function CTA() {
  return (
    <section className="bg-white py-24">
      <div className="mx-auto max-w-4xl px-6 text-center">
        <div className="rounded-3xl bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-16 shadow-xl">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">Ready to fact-check?</h2>
          <p className="mx-auto mt-4 max-w-xl text-lg text-indigo-100">
            Paste any article, social media post, or paragraph and get an instant AI-powered verification report.
          </p>
          <Link
            href="/verify"
            className="mt-8 inline-block rounded-full bg-white px-10 py-3.5 text-base font-bold text-indigo-600 shadow-lg transition hover:bg-indigo-50"
          >
            Try ClaimLens Now →
          </Link>
        </div>
      </div>
    </section>
  );
}
