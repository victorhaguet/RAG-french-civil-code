/**
 * Parser for the Streamed Answer of `POST /query/stream` (see ADR 0005):
 * NDJSON, one typed JSON message per line. Pure: it takes bytes, as they
 * arrive, and returns typed events, so it knows nothing of `fetch`.
 */

/** A Retrieved Article, as the stream cites it. */
export interface ArticleRef {
  ref: string;
  sectionParentTitre: string;
}

/** The messages the API sends, one per line. */
export type StreamMessage =
  | { type: "articles"; articles: ArticleRef[] }
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

/**
 * What the parser reports: a message, a line that isn't one (`malformed`), or
 * the stream ending without `done` or `error` (`interrupted`).
 */
export type ParsedEvent =
  | StreamMessage
  | { type: "malformed"; line: string }
  | { type: "interrupted" };

/**
 * Turns the stream's bytes into events. A line or a multi-byte UTF-8
 * character may be split across pieces: the parser keeps the unfinished end
 * until the next piece completes it.
 */
export class StreamParser {
  private readonly decoder = new TextDecoder();
  private buffer = "";
  private finished = false;

  /** Parse the next piece of the stream; returns the events of the lines it completes. */
  push(bytes: Uint8Array): ParsedEvent[] {
    this.buffer += this.decoder.decode(bytes, { stream: true });
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop() ?? "";
    return this.parseLines(lines);
  }

  /** The stream ended: parse what's left, and report an interruption if it never finished. */
  end(): ParsedEvent[] {
    this.buffer += this.decoder.decode();
    const events = this.parseLines([this.buffer]);
    this.buffer = "";
    if (!this.finished) {
      events.push({ type: "interrupted" });
    }
    return events;
  }

  private parseLines(lines: string[]): ParsedEvent[] {
    const events: ParsedEvent[] = [];
    for (const line of lines) {
      if (line.trim() === "") {
        continue;
      }
      const event = parseLine(line);
      if (event.type === "done" || event.type === "error") {
        this.finished = true;
      }
      events.push(event);
    }
    return events;
  }
}

function parseLine(line: string): ParsedEvent {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch {
    return { type: "malformed", line };
  }
  return toMessage(value) ?? { type: "malformed", line };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isArticleRef(value: unknown): value is ArticleRef {
  return (
    isRecord(value) &&
    typeof value.ref === "string" &&
    typeof value.sectionParentTitre === "string"
  );
}

/** The message `value` is, or `null` if it doesn't have a known message's shape. */
function toMessage(value: unknown): StreamMessage | null {
  if (!isRecord(value)) {
    return null;
  }
  switch (value.type) {
    case "articles":
      return Array.isArray(value.articles) && value.articles.every(isArticleRef)
        ? { type: "articles", articles: value.articles }
        : null;
    case "delta":
      return typeof value.text === "string" ? { type: "delta", text: value.text } : null;
    case "done":
      return { type: "done" };
    case "error":
      return typeof value.message === "string"
        ? { type: "error", message: value.message }
        : null;
    default:
      return null;
  }
}
