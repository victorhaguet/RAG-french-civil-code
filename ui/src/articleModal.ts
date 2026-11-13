/**
 * The modal a citation link opens: the Article's ref, section title, full
 * text and in-force date, in the page's native `<dialog>`. Closes with Esc
 * (native), the × button or a click outside it.
 */

import { formatDateDebut, type ArticleResult } from "./articles";

function part<T extends HTMLElement>(dialog: HTMLDialogElement, selector: string): T {
  const found = dialog.querySelector<T>(selector);
  if (found === null) {
    throw new Error(`Missing ${selector} in #article-dialog`);
  }
  return found;
}

/** Wire `dialog`, and return the function that opens it on an Article. */
export function setUpArticleModal(
  dialog: HTMLDialogElement,
  loadArticle: (ref: string) => Promise<ArticleResult>,
): (ref: string) => Promise<void> {
  const title = part<HTMLElement>(dialog, ".article-dialog__title");
  const section = part<HTMLElement>(dialog, ".article-dialog__section");
  const text = part<HTMLElement>(dialog, ".article-dialog__text");
  const date = part<HTMLElement>(dialog, ".article-dialog__date");
  const close = part<HTMLButtonElement>(dialog, ".article-dialog__close");

  close.addEventListener("click", () => dialog.close());
  // The dialog's content fills it edge to edge: a click on the dialog
  // itself is a click on the backdrop, outside the content.
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      dialog.close();
    }
  });

  // Only the last Article asked for may fill the modal, however the loads
  // finish.
  let latest = 0;
  return async (ref) => {
    const request = ++latest;
    title.textContent = ref;
    section.textContent = "";
    date.textContent = "";
    text.classList.remove("notice");
    text.textContent = "Chargement…";
    dialog.setAttribute("aria-busy", "true");
    if (!dialog.open) {
      dialog.showModal();
    }

    const result = await loadArticle(ref);
    if (request !== latest) {
      return;
    }
    dialog.removeAttribute("aria-busy");
    if (!result.ok) {
      text.classList.add("notice");
      text.textContent = result.message;
      return;
    }
    const { article } = result;
    title.textContent = article.ref;
    section.textContent = article.sectionParentTitre;
    text.textContent = article.texte;
    date.textContent = `En vigueur depuis le ${formatDateDebut(article.dateDebut)}.`;
  };
}
