import { describe, expect, it } from "vitest";

import { linkCitations } from "../src/citationLinker";

const RETRIEVED = ["Code civil, art. 1240", "Code civil, art. 1241", "Code civil, art. 16-1"];

/** Link `html` against `refs`, into a detached element to query as DOM. */
function linked(html: string, refs: readonly string[] = RETRIEVED, partial = false): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = linkCitations(html, refs, { partial });
  return container;
}

/** Each citation link as [its text, the Article ref it carries]. */
function citations(container: HTMLElement): [string, string][] {
  return Array.from(container.querySelectorAll<HTMLAnchorElement>("a.citation"), (link) => [
    link.textContent ?? "",
    link.dataset.articleRef ?? "",
  ]);
}

describe("linkCitations", () => {
  it("links a body mention of a Retrieved Article", () => {
    expect(citations(linked("<p>Selon l'art. 1240, la faute oblige.</p>"))).toEqual([
      ["art. 1240", "Code civil, art. 1240"],
    ]);
  });

  it.each([
    ["article 1240", "<p>L'obligation naît de l'article 1240.</p>", "article 1240", "Code civil, art. 1240"],
    ["l'article 1240", "<p>Voir l'article 1240 du Code civil.</p>", "article 1240", "Code civil, art. 1240"],
    ["Article 1240", "<p>Article 1240 : la faute oblige.</p>", "Article 1240", "Code civil, art. 1240"],
    ["a hyphenated number", "<p>Selon l'article 16-1, le corps est inviolable.</p>", "article 16-1", "Code civil, art. 16-1"],
  ])("links %s", (_, html, text, ref) => {
    expect(citations(linked(html))).toEqual([[text, ref]]);
  });

  it("leaves mentions of Articles that weren't retrieved as plain text", () => {
    const container = linked("<p>Voir l'article 1242, l'art. 16 et l'article 1240-1.</p>");

    expect(citations(container)).toEqual([]);
    expect(container.textContent).toBe("Voir l'article 1242, l'art. 16 et l'article 1240-1.");
  });

  it("links each number of a list of articles", () => {
    expect(citations(linked("<p>Les articles 1240 et 1241 fondent la responsabilité.</p>"))).toEqual([
      ["articles 1240", "Code civil, art. 1240"],
      ["1241", "Code civil, art. 1241"],
    ]);
  });

  it("links the retrieved numbers of a list joined by commas or « ou »", () => {
    expect(citations(linked("<p>(articles 146, 1241 ou 16-1)</p>"))).toEqual([
      ["1241", "Code civil, art. 1241"],
      ["16-1", "Code civil, art. 16-1"],
    ]);
  });

  describe("numbers that aren't another article of the mention", () => {
    const refs = ["Code civil, art. 1", "Code civil, art. 2", "Code civil, art. 247"];

    it.each([
      ["a paragraph number after a comma", "<p>(article 247, 1°)</p>"],
      ["a quantity after a comma", "<p>Selon l'article 247, 2 ans suffisent.</p>"],
      ["a comma-separated list that « et » or « ou » doesn't close", "<p>Les articles 247, 2 et suivants.</p>"],
    ])("doesn't link %s", (_, html) => {
      expect(citations(linked(html, refs)).map(([, ref]) => ref)).toEqual(["Code civil, art. 247"]);
    });

    it("reads « 1er » as article 1, and links it whole", () => {
      expect(citations(linked("<p>Articles 1er et 2 du Code civil.</p>", refs))).toEqual([
        ["Articles 1er", "Code civil, art. 1"],
        ["2", "Code civil, art. 2"],
      ]);
    });
  });

  it("links the whole ref in a « Fondement juridique » line", () => {
    const html =
      "<ul><li>Article Code civil, art. 1240 (Chapitre II : De la responsabilité) : la faute.</li></ul>";

    expect(citations(linked(html))).toEqual([
      ["Article Code civil, art. 1240", "Code civil, art. 1240"],
    ]);
  });

  it("doesn't link inside existing links or code", () => {
    const html =
      '<p><a href="https://example.org">article 1240</a> <code>art. 1241</code></p>' +
      "<pre><code>article 16-1</code></pre>";

    const container = linked(html);

    expect(citations(container)).toEqual([]);
    expect(container.innerHTML).toBe(html);
  });

  it("leaves an Out-of-Scope Answer unlinked", () => {
    const html =
      "<p>Je ne peux pas répondre à cette question à partir des informations récupérées\n" +
      "dans le Code civil.</p>\n<p>Ce chatbot répond uniquement en s'appuyant sur les articles " +
      "du Code civil disponibles dans sa base documentaire (extraction du 21 septembre 2025).</p>";

    expect(linked(html).innerHTML).toBe(html);
  });

  describe("on partial (mid-stream) text", () => {
    const refs = ["Code civil, art. 12", "Code civil, art. 16", "Code civil, art. 1240"];

    it.each([
      ["a number that may still grow", "<p>Selon l'article 12</p>\n"],
      ["a number whose hyphenated part hasn't arrived", "<p>Selon l'article 16-</p>"],
    ])("doesn't link %s at the end of the text", (_, html) => {
      expect(citations(linked(html, refs, true))).toEqual([]);
    });

    it("links a mention as soon as text follows it", () => {
      expect(citations(linked("<p>Selon l'article 1240 et l'art. 12</p>", refs, true))).toEqual([
        ["article 1240", "Code civil, art. 1240"],
      ]);
    });

    it("links a number at the end once the text is complete", () => {
      expect(citations(linked("<p>Selon l'article 12</p>\n", refs, false))).toEqual([
        ["article 12", "Code civil, art. 12"],
      ]);
    });
  });
});
