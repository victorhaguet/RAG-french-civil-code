/**
 * The citation linker: turns the answer's mentions of Retrieved Articles
 * into links carrying the Article's ref, which the Web UI opens in a modal.
 * Works on the HTML marked produces, before sanitizing, so marked has
 * already decided what is code or a link. Pure: HTML in, HTML out.
 */

/**
 * An article number: `1240`, hyphenated, `16-1`, or `1er` (article 1). Not
 * followed by more of a number, a letter or `°`: "1°" is a paragraph, not an
 * article.
 */
const NUMBER = String.raw`\d+(?:-\d+)*(?:er)?(?![\d\p{L}°]|-\d)`;
/**
 * The exact form of a "Fondement juridique" line, "Article Code civil, art.
 * 1240"; "art. 1240", "article 1240", "Article 1240" ("l'" stays out of the
 * link); or a list that « et » or « ou » closes: "articles 1240 et 1241",
 * "articles 146, 1241 ou 16-1". A number after a comma alone ("article 247,
 * 2 ans") isn't an article.
 */
const MENTION = new RegExp(
  String.raw`(?<!\p{L})(?:(?:article\s+)?code\s+civil\s*,\s*art\.|art\.|articles?)\s*${NUMBER}` +
    String.raw`(?:(?:\s*,\s*${NUMBER})*\s*,?\s+(?:et|ou)\s+${NUMBER})?`,
  "giu",
);
const NUMBER_IN_MENTION = new RegExp(NUMBER, "giu");

/** Where a mention stays as it is: already a link, or code. */
const UNLINKABLE = "a, code, pre";
/** What starts a code span, a link or an HTML tag in Markdown. */
const UNCLOSED_OPENER = /[`[<]/u;

export interface LinkOptions {
  /** The answer is still streaming: don't link a mention that may not be complete yet. */
  partial: boolean;
}

/** `html` with its mentions of the Articles in `articleRefs` turned into links. */
export function linkCitations(
  html: string,
  articleRefs: readonly string[],
  { partial }: LinkOptions,
): string {
  const refByNumber = new Map(articleRefs.map((ref) => [articleNumber(ref), ref]));
  const template = document.createElement("template");
  template.innerHTML = html;
  const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT);
  const textNodes: Text[] = [];
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    textNodes.push(node as Text);
  }
  // The last text node with more than whitespace: where a partial text ends.
  const lastText = textNodes.filter((node) => node.data.trim() !== "").at(-1);
  // Where a partial text may open a code span or a link that hasn't closed
  // yet: marked shows it as plain text for now, but once it closes, a link
  // inside it would vanish. Nothing from there on is linked yet.
  let opened = false;
  for (const node of textNodes) {
    if (node.parentElement?.closest(UNLINKABLE) != null) {
      continue;
    }
    const opener = partial && !opened ? node.data.search(UNCLOSED_OPENER) : -1;
    const cutoff = opened ? 0 : opener >= 0 ? opener : Infinity;
    if (opener >= 0) {
      opened = true;
    }
    const tooEarly = (end: number): boolean =>
      end > cutoff ||
      // A number right at the end of a partial text may still grow:
      // "art. 12" may become "art. 1240", "art. 16-" "art. 16-1".
      (partial && node === lastText && /^-?\s*$/u.test(node.data.slice(end)));
    linkTextNode(node, refByNumber, tooEarly);
  }
  return template.innerHTML;
}

/** The article number in a ref, e.g. `Code civil, art. 16-1` → `16-1`. */
function articleNumber(ref: string): string {
  return ref.replace(/^.*art\.\s*/u, "");
}

/** A piece of text to link: `[start, end)` in its text node, and the ref it cites. */
interface Citation {
  start: number;
  end: number;
  ref: string;
}

/** The citations in `text`, in order. */
function findCitations(
  text: string,
  refByNumber: Map<string, string>,
  tooEarly: (end: number) => boolean,
): Citation[] {
  const found: Citation[] = [];
  for (const mention of text.matchAll(MENTION)) {
    let first = true;
    for (const number of mention[0].matchAll(NUMBER_IN_MENTION)) {
      const ref = refByNumber.get(number[0].replace(/er$/iu, ""));
      const end = mention.index + number.index + number[0].length;
      // The first number's link takes the keyword along: "article 1240".
      const start = first ? mention.index : mention.index + number.index;
      first = false;
      if (ref !== undefined && !tooEarly(end)) {
        found.push({ start, end, ref });
      }
    }
  }
  return found;
}

function linkTextNode(
  node: Text,
  refByNumber: Map<string, string>,
  tooEarly: (end: number) => boolean,
): void {
  const text = node.data;
  const citations = findCitations(text, refByNumber, tooEarly);
  if (citations.length === 0) {
    return;
  }
  const pieces: (string | Node)[] = [];
  let last = 0;
  for (const { start, end, ref } of citations) {
    pieces.push(text.slice(last, start), citationLink(text.slice(start, end), ref));
    last = end;
  }
  pieces.push(text.slice(last));
  node.replaceWith(...pieces);
}

function citationLink(text: string, ref: string): HTMLAnchorElement {
  const link = document.createElement("a");
  link.className = "citation";
  link.setAttribute("href", `#${encodeURIComponent(ref)}`);
  link.dataset.articleRef = ref;
  link.textContent = text;
  return link;
}
