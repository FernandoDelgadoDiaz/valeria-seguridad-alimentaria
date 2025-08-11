// netlify/functions/chat.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

// --- Carga perezosa del índice ---
let LOADED = false;
let CHUNKS = [];
let EMBS = [];

function loadDataOnce() {
  if (LOADED) return;
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const root = path.resolve(__dirname, "..", ".."); // repo root
  const dataDir = path.join(root, "data");

  const chunksPath = path.join(dataDir, "chunks.jsonl");
  const embsPath = path.join(dataDir, "embeddings.jsonl");

  if (!fs.existsSync(chunksPath) || !fs.existsSync(embsPath)) {
    // Aún no generaste embeddings (siguiente paso del diff)
    LOADED = true;
    CHUNKS = [];
    EMBS = [];
    return;
  }

  CHUNKS = fs.readFileSync(chunksPath, "utf-8").trim().split("\n").map(JSON.parse);
  EMBS = fs.readFileSync(embsPath, "utf-8").trim().split("\n").map(JSON.parse);
  LOADED = true;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

export default async (req, res) => {
  try {
    loadDataOnce();

    const { query } = JSON.parse(req.body || "{}");
    if (!query || !query.trim()) {
      return res.status(400).json({ error: "missing query" });
    }

    if (!EMBS.length || !CHUNKS.length) {
      // Sin índice aún
      return res.status(200).json({
        answer:
`⚠️ Aún no está listo el índice semántico.
Subí /docs con los PDFs y agregá los siguientes archivos (siguiente paso del diff):
- package.json
- scripts/build-embeddings.mjs
- netlify.toml

Netlify generará /data/chunks.jsonl y /data/embeddings.jsonl automáticamente durante el build.`
      });
    }

    // Embed de la consulta
    const e = await client.embeddings.create({ model: EMB_MODEL, input: query });
    const qv = e.data[0].embedding;

    // Rank por coseno (top-5)
    const scored = EMBS
      .map((r, idx) => ({ idx, s: cosine(qv, r.embedding) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, 5);

    // Contexto (texto) de los mejores chunks
    const ctx = scored.map(({ idx }) => {
      const r = EMBS[idx];
      return CHUNKS.find(c => c.doc === r.doc && c.chunk === r.chunk);
    }).filter(Boolean);

    // Armar respuesta segura (resumen corto + links)
    const bullets = ctx.map(c => {
      const head = (c.text || "").split(/\n+/)[0].trim().slice(0, 240);
      return `• ${head || c.doc}`;
    });

    const links = [...new Set(ctx.map(c => c.path))] // docs/<archivo>.pdf
      .slice(0, 3)
      .map(p => `<a href="/${p}">PDF</a>`)
      .join(" · ");

    const answer =
`📌 <b>Resumen basado en documentación interna</b>:
${bullets.join("\n")}

🔗 ${links}`;

    return res.status(200).json({ answer });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
};