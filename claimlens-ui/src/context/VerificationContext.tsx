"use client";

import { createContext, useContext, useState, useCallback, useRef } from "react";
import { verifyTextStream, verifyText, type StreamCallbacks } from "@/lib/api";
import type { FinalReport, ClaimsExtractedData, ClaimVerifiedData, PipelineNode } from "@/types/api";
import { MIN_TEXT_LENGTH } from "@/constants/validation";

export type Phase = "input" | "loading" | "results";

export interface ExtractedClaim {
  id: string;
  text: string;
  status: string;
}

export interface VerifiedClaim {
  claim_id: string;
  claim_text: string;
  verdict: string;
  confidence: number;
}

interface VerificationState {
  text: string;
  setText: (t: string) => void;
  phase: Phase;
  claims: ExtractedClaim[];
  verified: VerifiedClaim[];
  report: FinalReport | null;
  error: string | null;
  progressMsg: string;
  currentNode: PipelineNode | null;
  completedNodes: PipelineNode[];
  handleVerify: () => void;
  handleCancel: () => void;
  handleReset: () => void;
}

const VerificationContext = createContext<VerificationState | null>(null);

export function VerificationProvider({ children }: { children: React.ReactNode }) {
  const [text, setText] = useState("");
  const [phase, setPhase] = useState<Phase>("input");
  const [claims, setClaims] = useState<ExtractedClaim[]>([]);
  const [verified, setVerified] = useState<VerifiedClaim[]>([]);
  const [report, setReport] = useState<FinalReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState("Starting verification...");
  const [currentNode, setCurrentNode] = useState<PipelineNode | null>(null);
  const [completedNodes, setCompletedNodes] = useState<PipelineNode[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const claimsRef = useRef<ExtractedClaim[]>([]);
  const verifiedRef = useRef<VerifiedClaim[]>([]);

  const handleVerify = useCallback(async () => {
    if (!text.trim() || text.trim().length < MIN_TEXT_LENGTH) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("loading");
    setError(null);
    setClaims([]);
    setVerified([]);
    claimsRef.current = [];
    verifiedRef.current = [];
    setReport(null);
    setProgressMsg("Starting verification...");
    setCurrentNode(null);
    setCompletedNodes([]);

    const callbacks: StreamCallbacks = {
      onStart: () => setProgressMsg("Decomposing claims..."),
      onStep: (d) => {
        setCurrentNode((prev) => {
          if (prev && prev !== d.node) {
            setCompletedNodes((list) =>
              list.includes(prev) ? list : [...list, prev]
            );
          }
          return d.node;
        });

        const claimCount = claimsRef.current.length;
        const verifiedCount = verifiedRef.current.length;
        const claimProgress = claimCount > 0 ? ` (claim ${verifiedCount + 1} of ${claimCount})` : "";

        const labels: Record<string, string> = {
          decompose_claims: "Decomposing text into claims...",
          prepare_claim: `Preparing claims in parallel${claimProgress}...`,
          generate_queries: `Generating search queries${claimProgress}...`,
          search_evidence: `Searching for evidence${claimProgress}...`,
          verify_claim: `Running NLI verification${claimProgress}...`,
          finalize_claim: `Finalizing claim result${claimProgress}...`,
          aggregate_results: "Aggregating all results...",
          generate_report: "Generating final report...",
        };
        setProgressMsg(labels[d.node] || "Processing...");
      },
      onClaimsExtracted: (d: ClaimsExtractedData) => {
        setClaims(d.claims);
        claimsRef.current = d.claims;
        setProgressMsg(`Extracted ${d.count} claim${d.count !== 1 ? "s" : ""}. Verifying in parallel...`);
      },
      onClaimVerified: (d: ClaimVerifiedData) => {
        setVerified((prev) => {
          // Deduplicate by claim_id (replace if already exists from a retry)
          const filtered = prev.filter((v) => v.claim_id !== d.claim_id);
          const next = [...filtered, d];
          verifiedRef.current = next;
          return next;
        });
        setProgressMsg(`Verified: "${d.claim_text.slice(0, 60)}..."`);
      },
      onComplete: () => {
        setProgressMsg("Finalizing report...");
        setCurrentNode((prev) => {
          if (prev) setCompletedNodes((list) => list.includes(prev) ? list : [...list, prev]);
          return null;
        });
      },
      onError: (err: string) => {
        setError(err);
        setPhase("input");
      },
    };

    try {
      let result: FinalReport | null = null;
      try {
        result = await verifyTextStream(text, callbacks, controller.signal);
      } catch {
        result = await verifyText(text, controller.signal);
      }

      if (controller.signal.aborted) return;

      if (result) {
        setReport(result);
        setPhase("results");
      } else {
        setError("Could not retrieve verification results.");
        setPhase("input");
      }
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : "Verification failed");
      setPhase("input");
    }
  }, [text]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setPhase("input");
    setProgressMsg("Starting verification...");
  }, []);

  const handleReset = useCallback(() => {
    abortRef.current?.abort();
    setPhase("input");
    setText("");
    setClaims([]);
    setVerified([]);
    setReport(null);
    setError(null);
  }, []);

  return (
    <VerificationContext.Provider
      value={{
        text, setText, phase, claims, verified, report, error,
        progressMsg, currentNode, completedNodes,
        handleVerify, handleCancel, handleReset,
      }}
    >
      {children}
    </VerificationContext.Provider>
  );
}

export function useVerificationContext() {
  const ctx = useContext(VerificationContext);
  if (!ctx) throw new Error("useVerificationContext must be used within VerificationProvider");
  return ctx;
}
