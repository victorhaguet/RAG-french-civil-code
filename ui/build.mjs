// Builds the Web UI into dist/ (gitignored), which the API serves at /ui/.
// `node build.mjs` builds once; `node build.mjs --watch` rebuilds on change.
// Everything is bundled: the page loads nothing from a CDN.
import { cpSync, mkdirSync, watch } from "node:fs";
import * as esbuild from "esbuild";

const OUT_DIR = "dist";
const isWatch = process.argv.includes("--watch");

function copyStatic() {
  mkdirSync(OUT_DIR, { recursive: true });
  cpSync("src/index.html", `${OUT_DIR}/index.html`);
  cpSync("assets", `${OUT_DIR}/assets`, { recursive: true });
}

/** @type {import("esbuild").BuildOptions} */
const options = {
  entryPoints: ["src/main.ts"],
  bundle: true,
  format: "esm",
  target: "es2022",
  outdir: OUT_DIR,
  minify: !isWatch,
  sourcemap: true,
  logLevel: "info",
};

copyStatic();
if (isWatch) {
  const context = await esbuild.context(options);
  await context.watch();
  for (const path of ["src/index.html", "assets"]) {
    watch(path, { recursive: true }, () => copyStatic());
  }
} else {
  await esbuild.build(options);
}
