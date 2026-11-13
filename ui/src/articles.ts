/**
 * Loading an Article's full text for the modal a citation link opens: from
 * `GET /articles/{ref}`, with a fresh Access Token, cached for the session
 * so reopening an Article is instant.
 */

import { formatFrenchDate } from "./about";
import { ARTICLES_URL } from "./config";
import { SESSION_EXPIRED } from "./errorMessages";

/** The part of `GET /articles/{ref}`'s body the modal shows. */
export interface Article {
  ref: string;
  texte: string;
  /** When this version came into force, in milliseconds since the epoch (UTC). */
  dateDebut: number;
  sectionParentTitre: string;
}

export type ArticleResult = { ok: true; article: Article } | { ok: false; message: string };

export interface ArticleLoaderOptions {
  fetchFn?: typeof fetch;
  /** A valid Access Token, refreshed first if it's about to expire. */
  getAccessToken: () => Promise<string>;
}

const NOT_FOUND = "Cet article est introuvable.";
const LOAD_FAILED = "Impossible de charger le texte de l'article. Réessayez dans un instant.";

/**
 * A function that loads an Article by ref. Only a successful load is kept:
 * after a failure, opening the Article again tries again.
 */
export function createArticleLoader({
  fetchFn = fetch,
  getAccessToken,
}: ArticleLoaderOptions): (ref: string) => Promise<ArticleResult> {
  const loaded = new Map<string, Article>();
  return async (ref) => {
    const cached = loaded.get(ref);
    if (cached !== undefined) {
      return { ok: true, article: cached };
    }
    let accessToken: string;
    try {
      accessToken = await getAccessToken();
    } catch {
      return { ok: false, message: SESSION_EXPIRED };
    }
    let response: Response;
    let body: unknown;
    try {
      response = await fetchFn(`${ARTICLES_URL}/${encodeURIComponent(ref)}`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      body = response.ok ? await response.json() : null;
    } catch {
      return { ok: false, message: LOAD_FAILED };
    }
    if (response.status === 401) {
      return { ok: false, message: SESSION_EXPIRED };
    }
    if (response.status === 404) {
      return { ok: false, message: NOT_FOUND };
    }
    if (!isArticle(body)) {
      return { ok: false, message: LOAD_FAILED };
    }
    loaded.set(ref, body);
    return { ok: true, article: body };
  };
}

/** An Article's `dateDebut` in French, e.g. `1er octobre 2016`. */
export function formatDateDebut(dateDebut: number): string {
  return formatFrenchDate(new Date(dateDebut).toISOString().slice(0, 10));
}

function isArticle(value: unknown): value is Article {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.ref === "string" &&
    typeof candidate.texte === "string" &&
    typeof candidate.dateDebut === "number" &&
    typeof candidate.sectionParentTitre === "string"
  );
}
