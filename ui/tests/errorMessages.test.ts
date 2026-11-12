import { describe, expect, it } from "vitest";

import { errorMessage } from "../src/errorMessages";

describe("errorMessage", () => {
  it("asks to log in again on a 401", () => {
    expect(errorMessage({ kind: "http", status: 401 })).toBe(
      "Votre session a expiré. Rechargez la page pour vous reconnecter.",
    );
  });

  it("says the account lacks the required role on a 403", () => {
    expect(errorMessage({ kind: "http", status: 403 })).toBe(
      "Votre compte n'a pas le rôle requis (code-civil-query) pour poser une question.",
    );
  });

  it.each([500, 502, 503])("gives the generic error on a %i", (status) => {
    expect(errorMessage({ kind: "http", status })).toBe(
      "Une erreur est survenue. Réessayez dans un instant.",
    );
  });

  it("gives the generic error on a network failure", () => {
    expect(errorMessage({ kind: "network" })).toBe(
      "Une erreur est survenue. Réessayez dans un instant.",
    );
  });

  it("warns that the answer is incomplete when the stream is interrupted", () => {
    expect(errorMessage({ kind: "interrupted" })).toBe(
      "La connexion a été interrompue : la réponse ci-dessus est incomplète.",
    );
  });

  it("warns that the answer is incomplete when the stream ends with an error message", () => {
    expect(errorMessage({ kind: "stream-error" })).toBe(
      "La génération de la réponse a échoué : la réponse ci-dessus est incomplète.",
    );
  });
});
