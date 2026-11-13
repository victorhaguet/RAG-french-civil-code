/**
 * Ask `POST /query/stream` a question and read its Streamed Answer as it
 * arrives, through the stream parser. Reports what went wrong as a `Failure`,
 * which `errorMessages` turns into French.
 */

import { QUERY_STREAM_URL } from "./config";
import type { Failure } from "./errorMessages";
import { StreamParser, type ArticleRef, type ParsedEvent } from "./streamParser";

export interface StreamAnswerOptions {
  question: string;
  accessToken: string;
  /** Called once, before any delta, with the Retrieved Articles. */
  onArticles: (articles: ArticleRef[]) => void;
  /** Called with each piece of the answer, in order. */
  onDelta: (text: string) => void;
}

/**
 * Stream the answer to `question`. Resolves once the stream is over: with
 * `null` if the whole answer arrived, or with the first `Failure` met.
 */
export async function streamAnswer({
  question,
  accessToken,
  onArticles,
  onDelta,
}: StreamAnswerOptions): Promise<Failure | null> {
  let response: Response;
  try {
    response = await fetch(QUERY_STREAM_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
        Accept: "application/x-ndjson",
      },
      body: JSON.stringify({ question }),
    });
  } catch {
    return { kind: "network" };
  }
  if (!response.ok) {
    return { kind: "http", status: response.status };
  }
  if (response.body === null) {
    return { kind: "interrupted" };
  }

  const parser = new StreamParser();
  const reader = response.body.getReader();
  let failure: Failure | null = null;
  const handle = (events: ParsedEvent[]): void => {
    for (const event of events) {
      const eventFailure = toFailure(event);
      if (eventFailure !== null) {
        failure ??= eventFailure;
      } else if (event.type === "articles") {
        onArticles(event.articles);
      } else if (event.type === "delta") {
        onDelta(event.text);
      }
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      handle(parser.push(value));
    }
  } catch {
    // The connection dropped mid-answer (e.g. the API stopped).
    return failure ?? { kind: "interrupted" };
  }
  handle(parser.end());
  return failure;
}

function toFailure(event: ParsedEvent): Failure | null {
  switch (event.type) {
    case "error":
    case "malformed":
      return { kind: "stream-error" };
    case "interrupted":
      return { kind: "interrupted" };
    default:
      return null;
  }
}
