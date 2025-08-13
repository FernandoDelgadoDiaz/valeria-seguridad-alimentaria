// scripts/build-embeddings.mjs
// Construye data/embeddings.json a partir de los PDFs en /docs usando OpenAI Embeddings.
// Compatible con Netlify (Node 20) y pdfjs-dist LEGACY en entorno Node.

import fs from "node:fs";
import path from "node:path";
import OpenAI from "openai";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs"; // <- LEGACY para Node

// ---------- Config ----------
const DOCS_DIR = process.env.DOCS_DIR || "docs";
const OUT_DIR  = "data";
const OUT_FILE = path.join(OUT_DIR, "embeddings.json");
const MODEL    = process.env.EMBEDDING_MODEL || "text-embedding-3-small";
const MAX_CHARS_PER_CHUNK = 1200;
const CHUNK_OVERLAP = 200;

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- Utils ----------
function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else yield full;
  }
}

function chunkText(text, max = MAX_CHARS_PER_CHUNK, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + max);
    chunks.push(text.slice(i, end));
    if (end === text.length) break;
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return chunks;
}

async function extractPdfText(filePath) {
  // *** FIX CLAVE ***
  // pdfjs-dist (legacy) en Node requiere Uint8Array, NO Buffer
  const uint8 = new Uint8Array(fs.readFileSync(filePath));

  // En Node no seteamos workerSrc; el LEGACY ya funciona sin worker.
  const pdf = await getDocument({ data: uint8 }).promise;

  let fullText = "";
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const pageText = content.items.map((it) => it.str || "").join(" ");
    fullText += (pageText + "\n");
  }
  // Si el PDF es escaneado (muy poco texto), devolvemos vacío para saltearlo
  return fullText.trim();
}

async function embedBatch(texts) {
  const res = await client.embeddings.create({
    model: MODEL,
    input: texts,
  });
  return res.data.map((x) => x.embedding);
}

// ---------- Main ----------
(async () => {
  if (!process.env.OPENAI_API_KEY || !process.env.OPENAI_API_KEY.trim()) {
    console.error("Falta OPENAI_API_KEY en ‘Environment variables’ de Netlify → Build & deploy.");
    process.exit(1);
  }

  ensureDir(OUT_DIR);

  const pdfFiles = [...walk(DOCS_DIR)].filter((f) => f.toLowerCase().endsWith(".pdf"));
  if (pdfFiles.length === 0) {
    console.warn(`[build-embeddings] No encontré PDFs en ${DOCS_DIR}. Nada para indexar.`);
  }

  const allRecords = [];
  const docsList = [];
  let totalChunks = 0;

  console.log(`[build-embeddings] Encontré ${pdfFiles.length} PDF(s). Extrayendo texto…`);

  for (const filePath of pdfFiles) {
    const rel = path.relative(DOCS_DIR, filePath);
    const title = path.basename(filePath);
    docsList.push(title);

    process.stdout.write(`[build-embeddings] • ${title} → `);

    let text = "";
    try {
      text = await extractPdfText(filePath);
    } catch (err) {
      console.warn(`error al leer: ${err?.message || err}`);
      continue;
    }

    if (!text || text.replace(/\s+/g, "").length < 100) {
      console.warn(`poco texto (¿escaneado?). Salteado.`);
      continue;
    }

    const chunks = chunkText(text);
    totalChunks += chunks.length;

    // Embeddings por tandas para evitar límites (100 por tanda suele ir bien)
    const BATCH = 100;
    for (let i = 0; i < chunks.length; i += BATCH) {
      const batch = chunks.slice(i, i + BATCH);
      const vecs = await embedBatch(batch);
      for (let j = 0; j < batch.length; j++) {
        allRecords.push({
          title,
          source: title,
          text: batch[j],
          // guardamos el vector como arreglo de floats
          embedding: vecs[j],
        });
      }
    }

    console.log(`${chunks.length} chunks`);
  }

  const payload = {
    version: `v${new Date().toISOString().slice(0, 10)}-build`,
    createdAt: new Date().toISOString(),
    docs: docsList.length,
    embeddings: allRecords.length,
    records: allRecords,
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(payload));
  console.log(`[build-embeddings] Listo: ${OUT_FILE} con ${allRecords.length} chunks de ${docsList.length} doc(s).`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});