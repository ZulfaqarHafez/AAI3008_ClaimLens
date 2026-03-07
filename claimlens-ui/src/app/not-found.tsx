import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-6 text-center">
      <p className="text-6xl font-extrabold text-indigo-600">404</p>
      <h2 className="mt-4 text-2xl font-bold text-gray-900">Page not found</h2>
      <p className="mt-2 max-w-md text-sm text-gray-500">
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>
      <Link
        href="/"
        className="mt-6 rounded-full bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700"
      >
        Back to Home
      </Link>
    </div>
  );
}
