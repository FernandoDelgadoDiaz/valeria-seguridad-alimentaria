// scripts/build-embeddings.mjs
// Genera data/embeddings.json desde todos los PDF en /docs (usa OPENAI_API_KEY)

import fs from "fs";
import path from "path";
import pdf from "pdf-parse";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("Falta OPENAI_API_KEY en el entorno.");
  process.exit(1);
}

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, "docs");
const OUT_DIR = path.join(ROOT, "data");
const OUT_PATH = path.join(OUT_DIR, "embeddings.json");
const MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

// Chunking simple
const CHUNK_SIZE = 1400;
const CHUNK_OVERLAP = 200;

const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();

const embed = async (text) => {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, input: text })
  });
  if (!res.ok) throw new Error(`Embeddings API ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.data[0].embedding;
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
      console.warn(`PDF sin texto/legible → ${file} (se saltea)`);
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