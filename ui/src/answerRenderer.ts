/**
 * The answer renderer: turns the Streamed Answer's `delta`s joined so far
 * (complete or cut mid-way) from Markdown into HTML that is safe to put in
 * the page.
 */

import DOMPurify from "dompurify";
import { marked } from "marked";

import { linkCitations } from "./citationLinker";

/**
 * A line holding nothing but one of the Grounded Answer's labels, possibly
 * bold or already a Markdown heading. The labels are the ones
 * `prompts/rag_answer_fr.jinja2` asks for ("Réponse :", "Fondement
 * juridique :"): change them there, change them here.
 */
const LABEL_LINE =
  /^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?[ \t]*(Réponse|Fondement juridique)[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$/gimu;

/**
 * Only what Markdown text produces, plus the attributes of the citation
 * linker's links (`class`, `data-article-ref`). No images or form controls:
 * an answer never needs them, and they could reach outside the page.
 */
const SANITIZE_OPTIONS = {
  ALLOWED_TAGS: [
    "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "del",
    "ul", "ol", "li", "a", "code", "pre", "blockquote",
    "table", "thead", "tbody", "tr", "th", "td",
  ],
  ALLOWED_ATTR: ["href", "title", "class", "data-article-ref", "start", "align"],
  ALLOW_DATA_ATTR: false,
};

export interface RenderOptions {
  /** The Retrieved Articles' refs, whose mentions become citation links. */
  articleRefs: readonly string[];
  /** The answer is still streaming: `markdown` may be cut mid-way. */
  partial: boolean;
}

/** Render `markdown` to sanitized HTML, its citations of Retrieved Articles linked. */
export function renderAnswer(
  markdown: string,
  { articleRefs, partial }: RenderOptions = { articleRefs: [], partial: false },
): string {
  const withHeadings = markdown.replace(LABEL_LINE, "### $1");
  const html = marked.parse(withHeadings, { async: false });
  return DOMPurify.sanitize(linkCitations(html, articleRefs, { partial }), SANITIZE_OPTIONS);
}
