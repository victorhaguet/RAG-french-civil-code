/**
 * The French message the Web UI shows for each way asking a question can
 * fail. Pure: the caller says what went wrong, this says how to put it.
 */

/**
 * What went wrong:
 * - `http`: the API answered with an error status, before any streaming;
 * - `network`: the request never got a response;
 * - `interrupted`: the Streamed Answer stopped before `done` (connection lost);
 * - `stream-error`: the Streamed Answer ended with an `error` message, or sent
 *   a line the UI couldn't read.
 *
 * The last two keep the partial answer on screen, so their message says it's
 * incomplete.
 */
export type Failure =
  | { kind: "http"; status: number }
  | { kind: "network" }
  | { kind: "interrupted" }
  | { kind: "stream-error" };

const GENERIC_ERROR = "Une erreur est survenue. Réessayez dans un instant.";
/** Also shown when an Article can't be loaded for the same reason (`articles.ts`). */
export const SESSION_EXPIRED = "Votre session a expiré. Rechargez la page pour vous reconnecter.";

/** The French message to show for `failure`. */
export function errorMessage(failure: Failure): string {
  switch (failure.kind) {
    case "http":
      if (failure.status === 401) {
        return SESSION_EXPIRED;
      }
      if (failure.status === 403) {
        return "Votre compte n'a pas le rôle requis (code-civil-query) pour poser une question.";
      }
      return GENERIC_ERROR;
    case "network":
      return GENERIC_ERROR;
    case "interrupted":
      return "La connexion a été interrompue : la réponse ci-dessus est incomplète.";
    case "stream-error":
      return "La génération de la réponse a échoué : la réponse ci-dessus est incomplète.";
  }
}
