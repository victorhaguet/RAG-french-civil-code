/**
 * About: the facts the header presents (Article count, dataset snapshot date,
 * generation model), read from the running server's public `/about` so they
 * can't drift from it. Without them, the header keeps its title and
 * disclaimer.
 */

import { ABOUT_URL, DATASET_NAME, DATASET_URL } from "./config";

/** `GET /about`'s body. */
export interface About {
  /** The dataset snapshot date, as an ISO date (`YYYY-MM-DD`). */
  dataset_as_of: string;
  article_count: number;
  model: string;
}

const FRENCH_DATE = new Intl.DateTimeFormat("fr-FR", {
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});
const FRENCH_NUMBER = new Intl.NumberFormat("fr-FR");

/** An ISO date in French, e.g. `2025-09-21` → `21 septembre 2025`, with `1er` for the first. */
export function formatFrenchDate(isoDate: string): string {
  // Read as UTC midnight, and formatted in UTC: the day can't shift with the
  // browser's time zone.
  const date = new Date(`${isoDate}T00:00:00Z`);
  const formatted = FRENCH_DATE.format(date);
  return date.getUTCDate() === 1 ? formatted.replace(/^1(?=\s)/, "1er") : formatted;
}

function isAbout(value: unknown): value is About {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.dataset_as_of === "string" &&
    /^\d{4}-\d{2}-\d{2}$/.test(candidate.dataset_as_of) &&
    typeof candidate.article_count === "number" &&
    typeof candidate.model === "string"
  );
}

/** About from the server, or `null` if it can't be had, whatever the reason. */
export async function fetchAbout(fetchFn: typeof fetch = fetch): Promise<About | null> {
  try {
    const response = await fetchFn(ABOUT_URL);
    if (!response.ok) {
      return null;
    }
    const body: unknown = await response.json();
    return isAbout(body) ? body : null;
  } catch {
    return null;
  }
}

function strong(text: string): HTMLElement {
  const element = document.createElement("strong");
  element.textContent = text;
  return element;
}

/** Fill `container` with the header's paragraph presenting the project, from `about`. */
export function renderAboutFacts(container: HTMLElement, about: About): void {
  const dataset = document.createElement("a");
  dataset.href = DATASET_URL;
  dataset.textContent = DATASET_NAME;

  container.replaceChildren(
    "Posez une question en français sur le droit civil : l'assistant recherche les articles " +
      "pertinents parmi les ",
    strong(`${FRENCH_NUMBER.format(about.article_count)} articles en vigueur`),
    " du Code civil (jeu de données ",
    dataset,
    `, extraction du ${formatFrenchDate(about.dataset_as_of)}), puis rédige une réponse ` +
      "fondée uniquement sur ces articles avec ",
    strong(about.model),
    ". Recherche hybride (mots-clés + sémantique) et reranking. Cliquez sur un article cité " +
      "pour lire son texte intégral.",
  );
}
