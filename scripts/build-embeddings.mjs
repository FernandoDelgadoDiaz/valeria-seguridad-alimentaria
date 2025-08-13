// scripts/build-embeddings.mjs
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
// Usamos el build "legacy" (compatible Node 20) y SIN worker
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.resolve(__dirname, "..", "docs");
const OUT_DIR  = path.resolve(__dirname, "..", "data");
const OUT_FILE = path.join(OUT_DIR, "embeddings.json");

// --- Config de chunking y modelo ---
const EMBEDDING_MODEL = "text-embedding-3-small"; // 1536 dims, económico
const CHUNK_SIZE = 1200;   // ~1.2k chars por chunk
const CHUNK_OVERLAP = 200; // solape para continuidad

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
if (!process.env.OPENAI_API_KEY) {
  console.error("Falta OPENAI_API_KEY en Netlify > Site settings > Environment.");
  process.exit(1);
}

// -------- Utilidades --------
function toUint8Array(buffer) {
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

function chunkText(txt, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const clean = txt.replace(/\s+\n/g, "\n").replace(/\n{2,}/g, "\n\n");
  const parts = [];
  let i = 0;
  while (i < clean.length) {
    const end = Math.min(clean.length, i + size);
    parts.push(clean.slice(i, end));
    i = end - overlap;
    if (i < 0) i = 0;
  }
  return parts.map((t) => t.trim()).filter(Boolean);
}

async function extractPdfText(filePath) {
  const buf = fs.readFileSync(filePath);
  const data = toUint8Array(buf);

  // Nada de worker; 100% modo Node
  const loadingTask = pdfjs.getDocument({
    data,
    disableWorker: true,
    useWorkerFetch: false,
    isEvalSupported: false,
    isOffscreenCanvasSupported: false,
    disableFontFace: true,
    verbosity: 0,
  });

  const pdf = await loadingTask.promise;
  let out = "";
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const tc = await page.getTextContent();
    const line = tc.items.map((i) => i.str).join(" ");
    out += (out ? "\n" : "") + line;
  }
  return out;
}

async function embedBatch(texts) {
  const res = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
  });
  return res.data.map((d) => d.embedding);
}

async function main() {
  const pdfs = fs.readdirSync(DOCS_DIR).filter((f) => f.toLowerCase().endsWith(".pdf"));
  if (pdfs.length === 0) {
    console.error("No hay PDFs en /docs");
    process.exit(1);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const chunks = [];
  let docIdx = 0;

  for (const pdfName of pdfs) {
    docIdx++;
    const full = path.join(DOCS_DIR, pdfName);
    console.log(`[${docIdx}/${pdfs.length}] ${pdfName} → extrayendo texto…`);
    const text = await extractPdfText(full);
    const parts = chunkText(text);
    parts.forEach((t, i) => {
      const id = `${pdfName}#c${String(i + 1).padStart(4, "0")}`;
      chunks.push({ id, source: pdfName, text: t });
    });
  }

  console.log(`Total chunks a embedir: ${chunks.length}`);

  // Embeddings por lotes (rápido y estable)
  const BATCH = 64;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const slice = chunks.slice(i, i + BATCH);
    const vecs = await embedBatch(slice.map((c) => c.text));
    slice.forEach((c, j) => (c.embedding = vecs[j]));
    console.log(`Embeddings ${i + slice.length}/${chunks.length}`);
  }

  const outJson = {
    meta: {
      version: "v1",
      createdAt: new Date().toISOString(),
      docsCount: pdfs.length,
      chunksCount: chunks.length,
      embeddingModel: EMBEDDING_MODEL,
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP,
    },
    chunks,
  };

  fs.writeFileSync(OUT_FILE, JSON.stringify(outJson));
  const stat = fs.statSync(OUT_FILE);
  console.log(`OK → ${OUT_FILE} (${stat.size} bytes)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});