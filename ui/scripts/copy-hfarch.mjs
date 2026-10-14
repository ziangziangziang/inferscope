// hfarch is hand-written, not part of the Vite build, so a rebuild that
// empties dist/ would otherwise delete it and take /hfarch with it.
import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const from = join(here, "..", "..", "hfarch");
const to = join(here, "..", "dist", "hfarch");
await rm(to, { recursive: true, force: true });
await mkdir(to, { recursive: true });
await cp(from, to, { recursive: true });
console.log("hfarch ->", to);
