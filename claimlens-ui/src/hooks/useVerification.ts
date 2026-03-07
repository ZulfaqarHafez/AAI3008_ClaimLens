"use client";

// Re-export types and hook from the global VerificationContext
// so that existing imports continue to work.
export type { Phase, ExtractedClaim, VerifiedClaim } from "@/context/VerificationContext";
export { useVerificationContext as useVerification } from "@/context/VerificationContext";

