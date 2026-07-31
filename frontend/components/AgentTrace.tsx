"use client";

const LABELS: Record<string, string> = {
  memory: "Memory — recalling past turns",
  supervisor: "Supervisor — routing",
  retriever: "Retriever — searching documents",
  web: "Web — searching the internet",
  data: "Data — running SQL",
  code: "Code — executing Python",
  generate: "Generate — drafting answer",
  critic: "Critic — verifying answer",
  remember: "Memory — storing this turn",
};

function nodeClass(node: string) {
  const base = node.split("→")[0].split("(")[0];
  return `trace-node n-${base} active`;
}

function label(node: string) {
  const base = node.split("→")[0].split("(")[0];
  const extra = node.includes("→") ? ` → ${node.split("→")[1]}` : "";
  return (LABELS[base] || base) + extra;
}

export default function AgentTrace({ steps }: { steps: string[] }) {
  if (steps.length === 0) {
    return <div className="empty-state">No trace yet — ask a question to watch the agents work.</div>;
  }
  return (
    <div className="trace">
      {steps.map((s, i) => (
        <div key={i} className={nodeClass(s)}>
          <span className="node-dot" />
          {label(s)}
        </div>
      ))}
    </div>
  );
}
