import { describe, expect, it } from "vitest";

import { StreamParser, type ParsedEvent } from "../src/streamParser";

const encoder = new TextEncoder();

/** Feed `bytes` to a fresh parser in pieces cut at `cuts`, then end it. */
function parseInPieces(bytes: Uint8Array, cuts: number[]): ParsedEvent[] {
  const parser = new StreamParser();
  const events: ParsedEvent[] = [];
  let start = 0;
  for (const cut of [...cuts, bytes.length]) {
    events.push(...parser.push(bytes.slice(start, cut)));
    start = cut;
  }
  events.push(...parser.end());
  return events;
}

const COMPLETE_STREAM =
  '{"type":"articles","articles":[{"ref":"LEGIARTI1","sectionParentTitre":"Titre préliminaire"}]}\n' +
  '{"type":"delta","text":"Une loi entre en vigueur "}\n' +
  '{"type":"delta","text":"à la date qu\'elle fixe. ⚖️"}\n' +
  '{"type":"done"}\n';

const COMPLETE_EVENTS: ParsedEvent[] = [
  {
    type: "articles",
    articles: [{ ref: "LEGIARTI1", sectionParentTitre: "Titre préliminaire" }],
  },
  { type: "delta", text: "Une loi entre en vigueur " },
  { type: "delta", text: "à la date qu'elle fixe. ⚖️" },
  { type: "done" },
];

describe("StreamParser", () => {
  it("parses a whole Streamed Answer received in one piece", () => {
    expect(parseInPieces(encoder.encode(COMPLETE_STREAM), [])).toEqual(COMPLETE_EVENTS);
  });

  it("gives the same events whichever byte the stream is split at", () => {
    const bytes = encoder.encode(COMPLETE_STREAM);
    for (let cut = 1; cut < bytes.length; cut++) {
      expect(parseInPieces(bytes, [cut]), `split at byte ${cut}`).toEqual(COMPLETE_EVENTS);
    }
  });

  it("gives the same events when every byte arrives on its own", () => {
    const bytes = encoder.encode(COMPLETE_STREAM);
    const everyByte = Array.from({ length: bytes.length - 1 }, (_, i) => i + 1);
    expect(parseInPieces(bytes, everyByte)).toEqual(COMPLETE_EVENTS);
  });

  it("joins a multi-byte character split between two pieces", () => {
    const bytes = encoder.encode('{"type":"delta","text":"é"}\n{"type":"done"}\n');
    // "é" is two bytes in UTF-8: cut between them.
    const insideE = bytes.indexOf(0xc3) + 1;
    expect(parseInPieces(bytes, [insideE])).toEqual([
      { type: "delta", text: "é" },
      { type: "done" },
    ]);
  });

  it("reports a stream that ends without done or error as interrupted", () => {
    const bytes = encoder.encode('{"type":"articles","articles":[]}\n{"type":"delta","text":"Une loi"}\n');
    expect(parseInPieces(bytes, [10])).toEqual([
      { type: "articles", articles: [] },
      { type: "delta", text: "Une loi" },
      { type: "interrupted" },
    ]);
  });

  it("reports a stream cut inside a line as malformed, then interrupted", () => {
    const bytes = encoder.encode('{"type":"delta","text":"Une loi"}\n{"type":"del');
    expect(parseInPieces(bytes, [])).toEqual([
      { type: "delta", text: "Une loi" },
      { type: "malformed", line: '{"type":"del' },
      { type: "interrupted" },
    ]);
  });

  it("reports a line that isn't a known message as malformed, and keeps parsing", () => {
    const bytes = encoder.encode(
      'not json\n{"type":"surprise"}\n{"type":"delta","text":42}\n{"type":"done"}\n',
    );
    expect(parseInPieces(bytes, [])).toEqual([
      { type: "malformed", line: "not json" },
      { type: "malformed", line: '{"type":"surprise"}' },
      { type: "malformed", line: '{"type":"delta","text":42}' },
      { type: "done" },
    ]);
  });

  it("treats an error message as the end of the stream, not an interruption", () => {
    const bytes = encoder.encode(
      '{"type":"delta","text":"Une loi"}\n{"type":"error","message":"Answer generation failed."}\n',
    );
    expect(parseInPieces(bytes, [])).toEqual([
      { type: "delta", text: "Une loi" },
      { type: "error", message: "Answer generation failed." },
    ]);
  });

  it("parses a last line without its trailing newline", () => {
    expect(parseInPieces(encoder.encode('{"type":"done"}'), [])).toEqual([{ type: "done" }]);
  });
});
