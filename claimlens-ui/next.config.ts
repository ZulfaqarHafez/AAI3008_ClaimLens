import type { NextConfig } from "next";

const BACKEND_URL = process.env.API_BACKEND_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactCompiler: true,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
  // Required for SSE (Server-Sent Events) to stream through without buffering
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
