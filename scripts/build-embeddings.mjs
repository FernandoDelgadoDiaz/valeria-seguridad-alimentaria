// scripts/build-embeddings.mjs
// v2025-08-13-fix1 — extractor robusto + filtros anti-chunks vacíos

import fs from "fs";
import path from "path";
import pdfParse from "pdf-parse";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, "docs");
const DATA_DIR = path.join(ROOT, "data");
const OUT_FILE = path.join(DATA_DIR, "embeddings.json");

// ------------ utils ------------
function clean(s = "") {
  return s.replace(/\s+/g, " ").replace(/[ \t]+/g, " ").trim();
}

function chunkText(s, max = 1200, overlap = 200) {
  const text = clean(s);
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + max);
    let slice = text.slice(i, end);
    // intenta cortar en el último punto para no romper frases
    const lastDot = slice.lastIndexOf(".");
    if (lastDot > 800) slice = slice.slice(0, lastDot + 1);
    if (slice.replace(/\s/g, "").length >= 200) chunks.push(slice.trim());
    i += Math.max(1, max - overlap);
  }
  return chunks;
}

async function embedBatch(texts, model = "text-embedding-3-small") {
  const res = await openai.embeddings.create({ model, input: texts });
  return res.data.map((d) => d.embedding);
}

// ------------ main ------------
async function main() {
  if (!fs.existsSync(DOCS_DIR)) {
    console.error("No existe docs/:", DOCS_DIR);
    process.exit(1);
  }
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

  const files = fs.readdirSync(DOCS_DIR)
    .filter((f) => f.toLowerCase().endsWith(".pdf"))
    .sort();

  const allChunks = [];
  let kept = 0;
  let total = 0;

  for (const f of files) {
    const full = path.join(DOCS_DIR, f);
    const buff = fs.readFileSync(full);
    const pdf = await pdfParse(buff).catch(() => ({ text: "" }));
    const text = clean(pdf.text || "");

    if (text.length < 300) {
      console.log(`${f} → sin texto útil (¿escaneado sin OCR?). Omitido.`);
      continue;
    }

    const chunks = chunkText(text, 1200, 250);
    total += chunks.length;

    // Filtro duro de utilidad
    const useful = chunks.filter((c) => c.replace(/\s/g, "").length >= 200);
    kept += useful.length;

    // Embeddings en lotes
    const BATCH = 64;
    for (let i = 0; i < useful.length; i += BATCH) {
      const slice = useful.slice(i, i + BATCH);
      const embs = await embedBatch(slice);
      for (let j = 0; j < slice.length; j++) {
        const textChunk = slice[j];
        allChunks.push({
          text: textChunk,
          preview: textChunk.slice(0, 320),
          source: f,
          title: path.basename(f, ".pdf"),
          embedding: embs[j],
        });
      }
      console.log(`${f} → ${Math.min(i + BATCH, useful.length)}/${useful.length} chunks`);
    }
  }

  const out = {
    meta: {
      createdAt: new Date().toISOString(),
      model: "text-embedding-3-small",
      files: files.length,
      chunksTotal: total,
      chunksKept: kept,
    },
    chunks: allChunks,
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(out));
  const size = fs.statSync(OUT_FILE).size;
  console.log(`Listo: ${OUT_FILE} (${size} bytes) con ${out.chunks.length} chunks.`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});