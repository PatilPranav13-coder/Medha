import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  devIndicators: false,
  output: process.env.MEDHA_STANDALONE === "true" ? "standalone" : undefined,
};

export default nextConfig;
