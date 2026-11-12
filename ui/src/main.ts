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
    answer.textContent = "";
    showNotice(null);
    try {
      try {
        await keycloak.updateToken(TOKEN_MIN_VALIDITY_SECONDS);
      } catch {
        // The Keycloak session itself is over: log in again.
        await keycloak.login();
        return;
      }
      const failure = await streamAnswer({
        question: text,
        accessToken: keycloak.token ?? "",
        onDelta: (piece) => {
          answer.textContent += piece;
        },
      });
      if (failure !== null) {
        showNotice(errorMessage(failure));
      }
      // Nothing streamed (e.g. a 403): don't leave an empty answer box.
      answerSection.hidden = answer.textContent === "";
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
