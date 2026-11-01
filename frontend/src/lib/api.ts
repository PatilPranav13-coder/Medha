export type Mode = "code" | "research" | "architecture" | "engineering";
export type BuildTarget = "auto" | "apple" | "android" | "windows" | "cross_platform";

export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  instructions: string;
  created_at: string;
  updated_at: string;
}

export interface Conversation {
  id: string;
  project_id: string | null;
  title: string;
  mode: Mode;
  model_name: string;
  created_at: string;
  updated_at: string;
}

export interface Source {
  title: string;
  url: string;
  snippet?: string;
  kind?: string;
  attribution?: string;
  authors?: string;
  year?: string;
  venue?: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  model_name: string;
  input_tokens: number | null;
  output_tokens: number | null;
  sources: Source[];
  created_at: string;
}

export interface DocumentFile {
  id: string;
  filename: string;
  extension: string;
  size_bytes: number;
  created_at: string;
  chunk_count: number;
}

export interface Memory {
  id: string;
  project_id: string | null;
  label: string;
  content: string;
  created_at: string;
}

export interface ModelCatalog {
  provider: string;
  configured: boolean;
  models: { id: string; name: string }[];
}

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    credentials: "include",
    cache: "no-store",
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(data.detail || `Request failed (${response.status})`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function streamPost(
  path: string,
  body: unknown,
  handlers: { onDelta: (text: string) => void; onDone: (data: Record<string, unknown>) => void; onError: (message: string) => void },
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(data.detail || `Request failed (${response.status})`, response.status);
  }
  if (!response.body) throw new Error("The server did not open a response stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const consume = (frame: string) => {
    const line = frame.split(/\r?\n/).find((item) => item.startsWith("data:"));
    if (!line) return;
    try {
      const event = JSON.parse(line.slice(5).trim()) as Record<string, unknown>;
      if (typeof event.delta === "string") handlers.onDelta(event.delta);
      if (event.done) handlers.onDone(event);
      if (typeof event.error === "string") handlers.onError(event.error);
    } catch {
      handlers.onError("Medha received an invalid stream event.");
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() || "";
    frames.forEach(consume);
    if (done) break;
  }
  if (buffer.trim()) consume(buffer);
}

export const apiBase = API_BASE;
