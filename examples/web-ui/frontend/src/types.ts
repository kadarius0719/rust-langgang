// Mirrors ai-core's provider-neutral message shapes (serde JSON form).

export type ContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; args: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }
  | { type: "thinking"; text: string; signature?: string };

export type Message = {
  role: "user" | "assistant" | "tool";
  content: ContentBlock[];
};

export type TraceEvent = {
  event: string;
  trace_id: number;
  [k: string]: unknown;
};
