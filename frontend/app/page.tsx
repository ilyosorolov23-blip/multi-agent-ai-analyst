"use client";

import { useState, useRef } from "react";
import AgentTrace from "@/components/AgentTrace";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLES = [
  "How many customers churned, and what reasons are given in the docs?",
  "What is the average order value?",
  "What is the refund policy?",
];

type Result = {
  answer: string;
  steps: string[];
  sources: string[];
  critic_ok?: boolean;
  revisions?: number;
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result>({ answer: "", steps: [], sources: [] });
  const abortRef = useRef<AbortController | null>(null);

  async function ask(q: string) {
    if (!q.trim() || loading) return;
    setLoading(true);
    setResult({ answer: "", steps: [], sources: [] });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
        signal: controller.signal,
      });
      if (!res.body) throw new Error("No response body — is the backend running?");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const evt of events) {
          const dataLine = evt.split("\n").find((l) => l.startsWith("data: "));
          if (!dataLine) continue;
          const payload = JSON.parse(dataLine.slice(6));
          if (payload.error) throw new Error(payload.error);

          setResult((prev) => ({
            answer: payload.answer ?? prev.answer,
            steps: payload.steps ?? prev.steps,
            sources: prev.sources,
            critic_ok: payload.critic_ok ?? prev.critic_ok,
            revisions: prev.revisions,
          }));
        }
      }
    } catch (e: any) {
      setResult((prev) => ({ ...prev, answer: `Error: ${e.message}` }));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <div className="header">
        <div className="brand">
          <span className="dot" />
          <div>
            <h1>Multi-Agent AI Analyst</h1>
            <div className="sub">supervisor · retriever · web · data · code · critic</div>
          </div>
        </div>
        <span className="status-pill">{API_URL.replace(/^https?:\/\//, "")}</span>
      </div>

      <div className="layout">
        <div className="panel">
          <h2>Ask</h2>
          <form
            className="ask-form"
            onSubmit={(e) => {
              e.preventDefault();
              ask(question);
            }}
          >
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question that needs data, docs, or math…"
            />
            <button type="submit" disabled={loading}>
              {loading ? "Working…" : "Ask"}
            </button>
          </form>
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="example-chip" onClick={() => { setQuestion(ex); ask(ex); }}>
                {ex}
              </button>
            ))}
          </div>

          {result.answer && (
            <div className="answer-box">
              {result.answer}
              {result.critic_ok !== undefined && (
                <div className={`critic-badge ${result.critic_ok ? "ok" : "revise"}`}>
                  {result.critic_ok ? "✓ verified by critic" : "↻ critic requested a revision"}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="panel">
          <h2>Live agent trace</h2>
          <AgentTrace steps={result.steps} />
        </div>
      </div>

      <div className="footer-note">
        F13 — streams live from /ask/stream · trace mirrors the supervisor graph in the project guide
      </div>
    </div>
  );
}
