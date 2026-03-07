import type { FinalReport, ClaimsExtractedData, ClaimVerifiedData, CompleteData } from "@/types/api";

const API_BASE = "/api";

export async function verifyText(text: string): Promise<FinalReport> {
  const res = await fetch(`${API_BASE}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Verification failed");
  }
  return res.json();
}

export interface StreamCallbacks {
  onStart?: () => void;
  onClaimsExtracted?: (data: ClaimsExtractedData) => void;
  onClaimVerified?: (data: ClaimVerifiedData) => void;
  onComplete?: (data: CompleteData) => void;
  onError?: (error: string) => void;
}

export async function verifyTextStream(
  text: string,
  callbacks: StreamCallbacks,
): Promise<FinalReport | null> {
  const res = await fetch(`${API_BASE}/verify/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    callbacks.onError?.(err.detail || "Verification failed");
    return null;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError?.("Streaming not supported");
    return null;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let finalReport: FinalReport | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentEventType = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const raw = line.slice(6);
        try {
          const parsed = JSON.parse(raw);
          const data = parsed.data ?? parsed;

          switch (currentEventType) {
            case "start":
              callbacks.onStart?.();
              break;
            case "claims_extracted":
              callbacks.onClaimsExtracted?.(data as ClaimsExtractedData);
              break;
            case "claim_verified":
              callbacks.onClaimVerified?.(data as ClaimVerifiedData);
              break;
            case "complete":
              callbacks.onComplete?.(data as CompleteData);
              break;
            case "error":
              callbacks.onError?.(data.error ?? "Unknown error");
              break;
          }
        } catch {
          // ignore malformed JSON
        }
        currentEventType = "";
      }
    }
  }

  // After stream ends, fetch the full sync report for complete data
  try {
    finalReport = await verifyText(text);
  } catch {
    // streaming was the primary path; final report is optional
  }

  return finalReport;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
