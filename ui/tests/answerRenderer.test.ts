import { describe, expect, it } from "vitest";

import { renderAnswer } from "../src/answerRenderer";

/** Render `markdown` into a detached element, to query the result as DOM. */
function rendered(markdown: string): HTMLElement {
  const container = document.createElement("div");
  container.innerHTML = renderAnswer(markdown);
  return container;
}

describe("renderAnswer", () => {
  it("renders Markdown headings, lists and bold", () => {
    const answer = rendered("## Titre\n\nUn **point** clé.\n\n- premier\n- second\n");

    expect(answer.querySelector("h2")?.textContent).toBe("Titre");
    expect(answer.querySelector("strong")?.textContent).toBe("point");
    expect(Array.from(answer.querySelectorAll("ul > li"), (li) => li.textContent)).toEqual([
      "premier",
      "second",
    ]);
  });

  it("strips <script> elements", () => {
    const answer = rendered('Avant <script>alert("x")</script> après');

    expect(answer.querySelector("script")).toBeNull();
    expect(answer.textContent).not.toContain("alert");
  });

  it("strips inline event handlers", () => {
    const answer = rendered('<img src="x" onerror="alert(1)"> <b onclick="alert(2)">gras</b>');

    for (const el of answer.querySelectorAll("*")) {
      expect(el.getAttributeNames().filter((name) => name.startsWith("on"))).toEqual([]);
    }
  });

  it("strips images and form controls, which Markdown answers never need", () => {
    const answer = rendered(
      '![pixel](https://tracker.example/p.png) <img src="https://tracker.example/q.png">\n\n' +
        '<form action="https://evil.example"><input name="x"><button>Envoyer</button></form>',
    );

    expect(answer.querySelector("img, form, input, button")).toBeNull();
  });

  it("strips javascript: URLs", () => {
    const answer = rendered('[Markdown](javascript:alert(1)) <a href="javascript:alert(2)">HTML</a>');

    for (const link of answer.querySelectorAll("a")) {
      expect(link.getAttribute("href") ?? "").not.toMatch(/javascript:/i);
    }
  });

  it("turns the Grounded Answer's label lines into headings", () => {
    const answer = rendered(
      "Réponse :\nLa loi entre en vigueur le lendemain de sa publication.\n\n" +
        "**Fondement juridique :**\n- Article 1 (Titre préliminaire) : fixe l'entrée en vigueur.\n",
    );

    expect(Array.from(answer.querySelectorAll("h3"), (h) => h.textContent)).toEqual([
      "Réponse",
      "Fondement juridique",
    ]);
    expect(answer.querySelector("p")?.textContent).toBe(
      "La loi entre en vigueur le lendemain de sa publication.",
    );
    expect(answer.querySelector("ul > li")?.textContent).toBe(
      "Article 1 (Titre préliminaire) : fixe l'entrée en vigueur.",
    );
  });

  it("leaves a label that is only part of a sentence as text", () => {
    const answer = rendered("Voici la Réponse : non.");

    expect(answer.querySelector("h3")).toBeNull();
    expect(answer.textContent).toBe("Voici la Réponse : non.\n");
  });

  it("renders every prefix of an answer, as it is mid-stream, without throwing", () => {
    const markdown =
      "**Réponse :**\nOui, **sous conditions** : voir [l'article](https://example.org/a?x=1).\n\n" +
      "Fondement juridique :\n- Article 1240 : *responsabilité*\n  - détail `code`\n\n" +
      "| a | b |\n|---|---|\n| 1 | 2 |\n<script>alert(1)</script>";

    for (let end = 0; end <= markdown.length; end++) {
      const answer = rendered(markdown.slice(0, end));
      expect(answer.querySelector("script")).toBeNull();
    }
  });

  it("shows an unclosed bold marker as text until it closes", () => {
    expect(rendered("Il faut **un délai").textContent).toBe("Il faut **un délai\n");
  });

  it("keeps a link carrying an Article ref, as the citation linker adds it", () => {
    const answer = rendered(
      '- <a class="citation" href="#article-1240" data-article-ref="1240">Article 1240</a> : faute\n' +
        "- [Article 1241](#article-1241) : imprudence\n",
    );

    const [htmlLink, markdownLink] = answer.querySelectorAll("a");
    expect(htmlLink?.textContent).toBe("Article 1240");
    expect(htmlLink?.getAttribute("href")).toBe("#article-1240");
    expect(htmlLink?.dataset.articleRef).toBe("1240");
    expect(htmlLink?.classList.contains("citation")).toBe(true);
    expect(markdownLink?.getAttribute("href")).toBe("#article-1241");
  });

  it("renders the Out-of-Scope Answer as plain paragraphs", () => {
    const answer = rendered(
      "Je ne peux pas répondre à cette question à partir des informations récupérées\n" +
        "dans le Code civil.\n\n" +
        "Si vous pensez que le Code civil contient une réponse à votre question,\n" +
        "essayez de la reformuler.",
    );

    expect(answer.querySelector("h1, h2, h3, h4, h5, h6, ul, ol")).toBeNull();
    expect(Array.from(answer.querySelectorAll("p"), (p) => p.textContent)).toEqual([
      "Je ne peux pas répondre à cette question à partir des informations récupérées\ndans le Code civil.",
      "Si vous pensez que le Code civil contient une réponse à votre question,\nessayez de la reformuler.",
    ]);
  });

  it("renders the English refusal as one paragraph", () => {
    const refusal =
      "This tool only answers questions in French, about the Code civil. " +
      "Please rephrase your question in French.";

    expect(rendered(refusal).innerHTML).toBe(`<p>${refusal}</p>\n`);
  });

  describe("with the Retrieved Articles' refs", () => {
    const refs = ["Code civil, art. 1240", "Code civil, art. 1241", "Code civil, art. 16-1"];
    const groundedAnswer =
      "Réponse :\nOui. Selon l'article 1240, toute faute oblige à réparer ; les articles 1240 et " +
      "1241 l'étendent à l'imprudence (voir aussi l'art. 1242).\n\n" +
      "Fondement juridique :\n" +
      "- Article Code civil, art. 1240 (Chapitre II : De la responsabilité) : la faute.\n" +
      "- Article Code civil, art. 1241 (Chapitre II : De la responsabilité) : l'imprudence.";

    /** Each citation link as [its text, the Article ref it carries]. */
    function citations(markdown: string, partial: boolean): [string, string][] {
      const container = document.createElement("div");
      container.innerHTML = renderAnswer(markdown, { articleRefs: refs, partial });
      return Array.from(container.querySelectorAll<HTMLAnchorElement>("a.citation"), (link) => [
        link.textContent ?? "",
        link.dataset.articleRef ?? "",
      ]);
    }

    it("keeps the citation linker's links through sanitizing", () => {
      const container = document.createElement("div");
      container.innerHTML = renderAnswer("Selon l'article 16-1, le corps est inviolable.", {
        articleRefs: refs,
        partial: false,
      });

      const link = container.querySelector<HTMLAnchorElement>("a.citation");
      expect(link?.textContent).toBe("article 16-1");
      expect(link?.dataset.articleRef).toBe("Code civil, art. 16-1");
      expect(link?.getAttribute("href")).toBe("#Code%20civil%2C%20art.%2016-1");
    });

    it("links a Grounded Answer's citations, and only the Retrieved Articles", () => {
      expect(citations(groundedAnswer, false)).toEqual([
        ["article 1240", "Code civil, art. 1240"],
        ["articles 1240", "Code civil, art. 1240"],
        ["1241", "Code civil, art. 1241"],
        ["Article Code civil, art. 1240", "Code civil, art. 1240"],
        ["Article Code civil, art. 1241", "Code civil, art. 1241"],
      ]);
    });

    it("never shows a link mid-stream that the finished answer doesn't have", () => {
      const final = citations(groundedAnswer, false);

      for (let end = 0; end <= groundedAnswer.length; end++) {
        const partial = citations(groundedAnswer.slice(0, end), true);
        expect(final.slice(0, partial.length)).toEqual(partial);
      }
    });

    it("doesn't link inside a code span or a link that hasn't closed yet", () => {
      const answer =
        "Voir `art. 1240 du Code` et [l'article 1241 ici](https://example.org), puis l'art. 16-1.";
      const final = citations(answer, false);
      expect(final).toEqual([["art. 16-1", "Code civil, art. 16-1"]]);

      for (let end = 0; end <= answer.length; end++) {
        const partial = citations(answer.slice(0, end), true);
        expect(final.slice(0, partial.length)).toEqual(partial);
      }
    });
  });
});
