/**
 * Web UI entry point: log in to Keycloak, then ask questions and show the
 * Streamed Answer as it arrives.
 */

import Keycloak from "keycloak-js";

import {
  KEYCLOAK_CLIENT_ID,
  KEYCLOAK_REALM,
  KEYCLOAK_URL,
  TOKEN_MIN_VALIDITY_SECONDS,
  WEB_UI_PATH,
} from "./config";
import { fetchAbout, renderAboutFacts } from "./about";
import { renderAnswer } from "./answerRenderer";
import { setUpArticleModal } from "./articleModal";
import { createArticleLoader } from "./articles";
import { errorMessage } from "./errorMessages";
import { streamAnswer } from "./streamAnswer";
import "./styles.css";

const EMPTY_QUESTION_MESSAGE = "Saisissez une question avant de l'envoyer.";

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`Missing #${id} in index.html`);
  }
  return found as T;
}

const keycloak = new Keycloak({
  url: KEYCLOAK_URL,
  realm: KEYCLOAK_REALM,
  clientId: KEYCLOAK_CLIENT_ID,
});

const loading = element<HTMLElement>("loading");

/** Fill the header's facts from `/about`, independently of login: it's public. */
async function showAbout(): Promise<void> {
  const about = await fetchAbout();
  if (about === null) {
    // The header keeps its title and disclaimer, without the facts.
    return;
  }
  const facts = element<HTMLElement>("about");
  renderAboutFacts(facts, about);
  facts.hidden = false;
}

async function main(): Promise<void> {
  // Redirects to Keycloak's login page when there's no session yet, and
  // comes back here with one.
  await keycloak.init({
    onLoad: "login-required",
    pkceMethod: "S256",
    checkLoginIframe: false,
  });

  const webUi = element<HTMLElement>("web-ui");
  const form = element<HTMLFormElement>("question-form");
  const question = element<HTMLTextAreaElement>("question");
  const submit = element<HTMLButtonElement>("submit");
  const answerSection = element<HTMLElement>("answer-section");
  const answer = element<HTMLElement>("answer");
  const notice = element<HTMLElement>("notice");
  const usernameLabel = element<HTMLElement>("username");
  const logout = element<HTMLAnchorElement>("logout");

  /** A valid Access Token, refreshed first if it's about to expire. */
  const freshAccessToken = async (): Promise<string> => {
    await keycloak.updateToken(TOKEN_MIN_VALIDITY_SECONDS);
    return keycloak.token ?? "";
  };
  const openArticle = setUpArticleModal(
    element<HTMLDialogElement>("article-dialog"),
    createArticleLoader({ getAccessToken: freshAccessToken }),
  );
  answer.addEventListener("click", (event) => {
    const link = event.target instanceof Element ? event.target.closest("a.citation") : null;
    const ref = link instanceof HTMLAnchorElement ? link.dataset.articleRef : undefined;
    if (ref !== undefined) {
      event.preventDefault();
      void openArticle(ref);
    }
  });

  const username = keycloak.tokenParsed?.preferred_username;
  usernameLabel.textContent = typeof username === "string" ? username : "";

  logout.addEventListener("click", (event) => {
    event.preventDefault();
    void keycloak.logout({ redirectUri: new URL(WEB_UI_PATH, window.location.origin).href });
  });

  const showNotice = (message: string | null): void => {
    notice.textContent = message ?? "";
    notice.hidden = message === null;
  };

  const ask = async (): Promise<void> => {
    const text = question.value.trim();
    if (text === "") {
      showNotice(EMPTY_QUESTION_MESSAGE);
      question.focus();
      return;
    }

    submit.disabled = true;
    answerSection.hidden = false;
    answerSection.setAttribute("aria-busy", "true");
    answer.replaceChildren();
    showNotice(null);
    // The answer's Markdown so far, re-rendered whole after each delta: at
    // most once per frame, however fast the deltas come.
    let markdown = "";
    let articleRefs: string[] = [];
    let pendingFrame: number | null = null;
    const showAnswer = (partial: boolean): void => {
      pendingFrame = null;
      answer.innerHTML = renderAnswer(markdown, { articleRefs, partial });
    };
    try {
      let accessToken: string;
      try {
        accessToken = await freshAccessToken();
      } catch {
        // The Keycloak session itself is over: log in again.
        await keycloak.login();
        return;
      }
      const failure = await streamAnswer({
        question: text,
        accessToken,
        onArticles: (articles) => {
          articleRefs = articles.map((article) => article.ref);
        },
        onDelta: (piece) => {
          markdown += piece;
          pendingFrame ??= requestAnimationFrame(() => showAnswer(true));
        },
      });
      if (pendingFrame !== null) {
        cancelAnimationFrame(pendingFrame);
      }
      showAnswer(false);
      if (failure !== null) {
        showNotice(errorMessage(failure));
      }
      // Nothing streamed (e.g. a 403): don't leave an empty answer box.
      answerSection.hidden = markdown === "";
    } finally {
      answerSection.removeAttribute("aria-busy");
      submit.disabled = false;
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void ask();
  });
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (!submit.disabled) {
        void ask();
      }
    }
  });

  webUi.hidden = false;
  loading.hidden = true;
  question.focus();
}

void showAbout();

main().catch(() => {
  loading.textContent =
    "Impossible de joindre le serveur de connexion. Vérifiez que Keycloak est démarré, puis rechargez la page.";
  loading.classList.add("notice");
});
