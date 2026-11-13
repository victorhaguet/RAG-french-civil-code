import { describe, expect, it } from "vitest";

import { createArticleLoader, formatDateDebut, type Article } from "../src/articles";

const ARTICLE: Article = {
  ref: "Code civil, art. 1240",
  texte: "Tout fait quelconque de l'homme, qui cause à autrui un dommage, oblige celui par la faute duquel il est arrivé à le réparer.",
  dateDebut: 1475280000000,
  sectionParentTitre: "Sous-section 1 : La responsabilité du fait personnel",
};

/** A fetch that answers `response()` and records each request it gets. */
function recordingFetch(response: () => Response | Promise<Response>) {
  const requests: Request[] = [];
  const fetchFn = async (input: RequestInfo | URL, init?: RequestInit) => {
    requests.push(new Request(new URL(String(input), "http://localhost"), init));
    return response();
  };
  return { fetchFn, requests };
}

describe("createArticleLoader", () => {
  it("fetches the Article by ref with a fresh Access Token", async () => {
    const { fetchFn, requests } = recordingFetch(() => Response.json(ARTICLE));
    let tokens = 0;
    const load = createArticleLoader({
      fetchFn,
      getAccessToken: async () => `token-${++tokens}`,
    });

    expect(await load(ARTICLE.ref)).toEqual({ ok: true, article: ARTICLE });
    expect(requests).toHaveLength(1);
    expect(new URL(requests[0]!.url).pathname).toBe("/articles/Code%20civil%2C%20art.%201240");
    expect(requests[0]!.headers.get("Authorization")).toBe("Bearer token-1");
  });

  it("reopens an Article already loaded without a new request", async () => {
    const { fetchFn, requests } = recordingFetch(() => Response.json(ARTICLE));
    const load = createArticleLoader({ fetchFn, getAccessToken: async () => "token" });

    await load(ARTICLE.ref);
    expect(await load(ARTICLE.ref)).toEqual({ ok: true, article: ARTICLE });

    expect(requests).toHaveLength(1);
  });

  it.each([
    [
      "an expired session",
      () => new Response("", { status: 401 }),
      "Votre session a expiré. Rechargez la page pour vous reconnecter.",
    ],
    ["a missing Article", () => new Response("", { status: 404 }), "Cet article est introuvable."],
    [
      "a server error",
      () => new Response("", { status: 500 }),
      "Impossible de charger le texte de l'article. Réessayez dans un instant.",
    ],
    [
      "a network failure",
      () => Promise.reject(new TypeError("Failed to fetch")),
      "Impossible de charger le texte de l'article. Réessayez dans un instant.",
    ],
    [
      "a body that isn't an Article",
      () => Response.json({ ref: "x" }),
      "Impossible de charger le texte de l'article. Réessayez dans un instant.",
    ],
  ])("reports %s in French, and tries again next time", async (_, response, message) => {
    const { fetchFn, requests } = recordingFetch(response);
    const load = createArticleLoader({ fetchFn, getAccessToken: async () => "token" });

    expect(await load(ARTICLE.ref)).toEqual({ ok: false, message });
    await load(ARTICLE.ref);

    expect(requests).toHaveLength(2);
  });

  it("reports a session that can't be refreshed as expired, without a request", async () => {
    const { fetchFn, requests } = recordingFetch(() => Response.json(ARTICLE));
    const load = createArticleLoader({
      fetchFn,
      getAccessToken: () => Promise.reject(new Error("refresh failed")),
    });

    expect(await load(ARTICLE.ref)).toEqual({
      ok: false,
      message: "Votre session a expiré. Rechargez la page pour vous reconnecter.",
    });
    expect(requests).toHaveLength(0);
  });
});

describe("formatDateDebut", () => {
  it("gives the in-force date in French", () => {
    expect(formatDateDebut(1475280000000)).toBe("1er octobre 2016");
  });

  it("handles dates before 1970", () => {
    expect(formatDateDebut(-5233334400000)).toBe("1er mars 1804");
  });
});
