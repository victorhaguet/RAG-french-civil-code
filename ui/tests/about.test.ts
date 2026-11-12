import { describe, expect, it } from "vitest";

import { fetchAbout, formatFrenchDate, renderAboutFacts, type About } from "../src/about";

const ABOUT: About = {
  dataset_as_of: "2025-09-21",
  article_count: 2288,
  model: "gpt-4o-mini",
};

/** Intl's French grouping uses a narrow no-break space: compare on plain spaces. */
function plainSpaces(text: string | null): string {
  return (text ?? "").replace(/[  ]/g, " ");
}

describe("formatFrenchDate", () => {
  it("spells the month out in French", () => {
    expect(formatFrenchDate("2025-09-21")).toBe("21 septembre 2025");
  });

  it("writes the first of the month as 1er", () => {
    expect(formatFrenchDate("2026-01-01")).toBe("1er janvier 2026");
  });
});

describe("renderAboutFacts", () => {
  it("presents the live Article count, the French date and the model", () => {
    const container = document.createElement("p");

    renderAboutFacts(container, ABOUT);

    const text = plainSpaces(container.textContent);
    expect(text).toContain("parmi les 2 288 articles en vigueur du Code civil");
    expect(text).toContain("extraction du 21 septembre 2025");
    expect(text).toContain("avec gpt-4o-mini.");
    expect(text).toContain("Cliquez sur un article cité pour lire son texte intégral.");
    const bold = Array.from(container.querySelectorAll("strong"), (el) => plainSpaces(el.textContent));
    expect(bold).toEqual(["2 288 articles en vigueur", "gpt-4o-mini"]);
  });

  it("links the dataset to its Hugging Face page", () => {
    const container = document.createElement("p");

    renderAboutFacts(container, ABOUT);

    const link = container.querySelector("a");
    expect(link?.textContent).toBe("louisbrulenaudet/code-civil");
    expect(link?.getAttribute("href")).toBe(
      "https://huggingface.co/datasets/louisbrulenaudet/code-civil",
    );
  });

  it("shows the model name as text, never as markup", () => {
    const container = document.createElement("p");

    renderAboutFacts(container, { ...ABOUT, model: "<img src=x>" });

    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img src=x>");
  });
});

describe("fetchAbout", () => {
  it("returns About from a successful response", async () => {
    const fetchFn = async () => new Response(JSON.stringify(ABOUT), { status: 200 });

    expect(await fetchAbout(fetchFn)).toEqual(ABOUT);
  });

  it.each([
    ["an error status", async () => new Response("oops", { status: 500 })],
    ["a network failure", async () => Promise.reject(new TypeError("Failed to fetch"))],
    ["a body that isn't About", async () => new Response('{"status":"ok"}', { status: 200 })],
    ["a body that isn't JSON", async () => new Response("<html>", { status: 200 })],
  ])("returns null on %s", async (_, fetchFn) => {
    expect(await fetchAbout(fetchFn)).toBeNull();
  });
});
