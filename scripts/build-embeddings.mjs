// scripts/build-embeddings.mjs
import fs from "fs";
import fsp from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import fg from "fast-glob";
import OpenAI from "openai";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const DOCS = path.join(ROOT, "docs");
const DATA = path.join(ROOT, "data");

function norm(s) {
  return (s || "")
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function tokensFromName(file) {
  return norm(file).split(" ").filter(Boolean);
}
function chunkText(txt, max = 1800) {
  const chunks = [];
  let i = 0;
  while (i < txt.length) {
    chunks.push(txt.slice(i, i + max));
    i += max;
  }
  return chunks.filter(c => c.trim().length > 0);
}

async function main() {
  console.log("↪ Escaneando PDFs en /docs …");
  if (!fs.existsSync(DOCS)) throw new Error("No existe /docs/");
  await fsp.mkdir(DATA, { recursive: true });

  const files = await fg("**/*.pdf", { cwd: DOCS, dot: false });
  console.log(`• Encontrados ${files.length} PDF(s)`);

  const chunks = [];
  const embRows = [];
  const filesIndex = [];

  for (let idx = 0; idx < files.length; idx++) {
    const rel = files[idx];
    const abs = path.join(DOCS, rel);
    const fileName = path.basename(rel);

    console.log(`  - Leyendo: ${fileName}`);
    const buf = await fsp.readFile(abs);

    let text = "";
    try {
      const res = await pdfParse(buf);
      text = (res.text || "").replace(/\r/g, "").trim();
    } catch (e) {
      console.log(`    ! pdf-parse falló en ${fileName} (seguiré)`);
    }

    // Índice por nombre (para "Dame PDF de …")
    filesIndex.push({
      file: fileName,
      path: `docs/${rel}`,
      norm: norm(fileName.replace(/\.pdf$/i, "")),
      tokens: tokensFromName(fileName.replace(/\.pdf$/i, ""))
    });

    if (!text || text.length < 30) {
      console.log(`    (sin texto utilizable → solo index por nombre)`);
      continue;
    }

    const parts = chunkText(text);
    for (let c = 0; c < parts.length; c++) {
      chunks.push({
        doc: fileName,
        chunk: c,
        path: `docs/${rel}`,
        text: parts[c]
      });
    }
  }

  // Embeddings sólo de los chunks con texto
  console.log(`↪ Generando embeddings de ${chunks.length} fragmentos…`);
  for (let i = 0; i < chunks.length; i++) {
    const { doc, chunk, text } = chunks[i];
    const { data } = await client.embeddings.create({
      model: EMB_MODEL,
      input: text
    });
    embRows.push({
      doc,
      chunk,
      embedding: data[0].embedding
    });
    if ((i + 1) % 50 === 0) console.log(`  • ${i + 1}/${chunks.length}`);
  }

  // Guardar
  const chunksPath = path.join(DATA, "chunks.jsonl");
  const embPath = path.join(DATA, "embeddings.jsonl");
  const filesIdxPath = path.join(DATA, "files_index.json");

  await fsp.writeFile(chunksPath, chunks.map(o => JSON.stringify(o)).join("\n"));
  await fsp.writeFile(embPath,   embRows.map(o => JSON.stringify(o)).join("\n"));
  await fsp.writeFile(filesIdxPath, JSON.stringify(filesIndex, null, 2));

  console.log("✔ Listo:");
  console.log("  -", chunksPath);
  console.log("  -", embPath);
  console.log("  -", filesIdxPath);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});