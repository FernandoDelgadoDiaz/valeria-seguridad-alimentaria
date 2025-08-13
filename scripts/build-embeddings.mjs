// scripts/build-embeddings.mjs
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import pdfjsLib from "pdfjs-dist/legacy/build/pdf.js";
import OpenAI from "openai";

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, "docs");
const OUT_DIR = path.join(ROOT, "data");
const OUT_FILE = path.join(OUT_DIR, "embeddings.json");

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";

function clean(txt) {
  return (txt || "")
    .replace(/\u0000/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function extractPdfText(filePath) {
  const buf = fs.readFileSync(filePath);
  const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  let out = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    out +=
      " " +
      content.items
        .map((it) => (typeof it.str === "string" ? it.str : ""))
        .join(" ");
  }
  return clean(out);
}

function chunk(text, size = 1100, overlap = 120) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const part = clean(text.slice(i, i + size));
    if (part.length >= 30) chunks.push(part); // filtro anti-vacíos
    i += Math.max(1, size - overlap);
  }
  return chunks;
}

async function* walkPdfs(dir) {
  const entries = await fsp.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) yield* walkPdfs(p);
    else if (e.isFile() && /\.pdf$/i.test(e.name)) yield p;
  }
}

async function embed(text) {
  const res = await client.embeddings.create({
    model: MODEL,
    input: text
  });
  return res.data[0].embedding;
}

(async () => {
  console.log("🧱 Construyendo embeddings…");
  await fsp.mkdir(OUT_DIR, { recursive: true });

  const vectors = [];
  let docsCount = 0;

  const pdfPaths = [];
  for await (const p of walkPdfs(DOCS_DIR)) pdfPaths.push(p);

  if (pdfPaths.length === 0) {
    console.log("⚠️  No se encontraron PDFs en", DOCS_DIR);
  }

  for (let idx = 0; idx < pdfPaths.length; idx++) {
    const pdfPath = pdfPaths[idx];
    const title = path.basename(pdfPath);
    docsCount++;
    console.log(
      `${new Date().toLocaleTimeString()}: ${title} → extrayendo texto… (${idx + 1}/${pdfPaths.length})`
    );

    const text = await extractPdfText(pdfPath);
    const parts = chunk(text);
    if (parts.length === 0) {
      console.log(`   → ⚠️  ${title}: sin texto utilizable (¿escaneado?).`);
      continue;
    }

    for (let i = 0; i < parts.length; i++) {
      const t = parts[i];
      const vec = await embed(t);
      vectors.push({
        id: `${title}#${i + 1}`,
        source: title,
        text: t,
        embedding: vec
      });
      if ((i + 1) % 5 === 0) {
        console.log(`   → ${title}: ${i + 1}/${parts.length} chunks`);
      }
    }
  }

  const payload = {
    version: `v${new Date().toISOString().slice(0, 10)}-fix2`,
    docs: docsCount,
    embeddings: vectors.length,
    items: vectors
  };

  await fsp.writeFile(OUT_FILE, JSON.stringify(payload));
  const stats = fs.statSync(OUT_FILE);
  console.log(
    `✅ Listo: ${OUT_FILE} (${stats.size} bytes) con ${vectors.length} chunks de ${docsCount} PDF(s).`
  );
})();