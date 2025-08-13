// scripts/build-embeddings.mjs
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
// Usamos el build legacy compatible con Node 20 y SIN worker.
import pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";
pdfjsLib.GlobalWorkerOptions.workerSrc = undefined; // evitar error de worker

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.resolve(__dirname, "..", "docs");
const DATA_DIR = path.resolve(__dirname, "..", "data");
const OUT_FILE = path.join(DATA_DIR, "embeddings.json");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function chunkText(text, maxLen = 1200, overlap = 200) {
  const chunks = [];
  const clean = text.replace(/\s+/g, " ").trim();
  for (let i = 0; i < clean.length; i += (maxLen - overlap)) {
    chunks.push(clean.slice(i, i + maxLen));
  }
  return chunks;
}

async function extractPdfText(pdfPath) {
  const u8 = new Uint8Array(await fs.readFile(pdfPath)); // **Uint8Array**, no Buffer
  const loadingTask = pdfjsLib.getDocument({ data: u8 });
  const pdf = await loadingTask.promise;

  let out = "";
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    out += content.items.map((i) => i.str).join(" ") + "\n";
    page.cleanup?.();
  }
  pdf.cleanup?.();
  return out;
}

async function ensureDirs() {
  await fs.mkdir(DATA_DIR, { recursive: true });
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY no está disponible en el build.");
  }

  await ensureDirs();
  const files = (await fs.readdir(DOCS_DIR))
    .filter((f) => f.toLowerCase().endsWith(".pdf"))
    .sort();

  const embeddings = [];
  let totalChunks = 0;

  for (const file of files) {
    const full = path.join(DOCS_DIR, file);
    const text = await extractPdfText(full);
    const parts = chunkText(text);
    totalChunks += parts.length;

    // Batches de hasta 100 entradas por request
    for (let i = 0; i < parts.length; i += 100) {
      const batch = parts.slice(i, i + 100);
      const resp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: batch,
      });

      resp.data.forEach((d, j) => {
        embeddings.push({
          source: file,
          idx: i + j,
          embedding: d.embedding, // <-- clave que leerá chat.js
        });
      });

      // respiro mínimo para rate limits
      await new Promise((r) => setTimeout(r, 120));
    }
    console.log(`OK: ${file} → ${parts.length} chunks`);
  }

  const payload = {
    ok: true,
    version: "v2025-08-13-fix2",
    docs: files.length,
    chunks: totalChunks,
    embeddingsCount: embeddings.length,
    embeddings,
  };

  await fs.writeFile(OUT_FILE, JSON.stringify(payload));
  console.log(`Listo: ${embeddings.length} vectores → ${OUT_FILE}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});