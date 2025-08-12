// scripts/build-embeddings.mjs
// Genera data/embeddings.json desde /docs usando OpenAI (Node 18+)

import fs from "fs";
import path from "path";
// FIX: usar la implementación directa, evita el modo CLI del paquete
import pdf from "pdf-parse/lib/pdf-parse.js";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("Falta OPENAI_API_KEY en el entorno.");
  process.exit(1);
}

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, "docs");
const OUT_DIR  = path.join(ROOT, "data");
const OUT_PATH = path.join(OUT_DIR, "embeddings.json");

const MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

// Chunking
const CHUNK_SIZE = 1400;
const CHUNK_OVERLAP = 200;

// Utilidades
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();

const embed = async (text) => {
  // backoff simple por si hay rate limit
  for (let i=0;i<5;i++){
    const res = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, input: text })
    });
    if (res.ok) {
      const data = await res.json();
      return data.data[0].embedding;
    }
    const t = await res.text().catch(()=>String(res.status));
    console.warn(`Embeddings API intento ${i+1}/5 → ${res.status}: ${t}`);
    await sleep(500 * (i+1));
  }
  throw new Error("Embeddings API agotó reintentos");
};

const chunkText = (raw) => {
  const text = raw.replace(/\s+/g, " ").trim();
  const chunks = [];
  for (let i = 0; i < text.length; i += (CHUNK_SIZE - CHUNK_OVERLAP)) {
    const slice = text.slice(i, i + CHUNK_SIZE);
    if (slice.length > 200) chunks.push(slice);
  }
  return chunks;
};

const main = async () => {
  if (!fs.existsSync(DOCS_DIR)) { console.error("No existe /docs."); process.exit(1); }
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  const files = fs.readdirSync(DOCS_DIR).filter(f => f.toLowerCase().endsWith(".pdf"));
  if (files.length === 0) { console.error("No hay PDFs en /docs."); process.exit(1); }

  const output = [];
  for (const file of files) {
    const filePath = path.join(DOCS_DIR, file);
    let text = "";
    try {
      const data = await pdf(fs.readFileSync(filePath));
      text = (data.text || "").trim();
    } catch (e) {
      console.warn(`PDF no legible (o imagen escaneada) → ${file} (se saltea)`);
      continue;
    }
    if (!text) { console.warn(`Sin texto → ${file} (se saltea)`); continue; }

    const title = file.replace(/\.pdf$/i, "");
    const chunks = chunkText(text);
    let n = 0;
    for (const chunk of chunks) {
      n++;
      const emb = await embed(chunk);
      output.push({
        id: `${toLowerNoAccents(title)}_${n.toString().padStart(3,"0")}`,
        title,
        source: file,
        chunk,
        embedding: emb
      });
      process.stdout.write(`\r${file} → ${n}/${chunks.length} chunks`);
    }
    process.stdout.write("\n");
  }

  fs.writeFileSync(OUT_PATH, JSON.stringify(output, null, 2), "utf-8");
  console.log(`\nListo: ${OUT_PATH} con ${output.length} chunks.`);
};

main().catch(err => { console.error(err); process.exit(1); });