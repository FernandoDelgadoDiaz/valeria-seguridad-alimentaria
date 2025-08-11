// functions/chat.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Modelo por defecto (compatible y económico). Si querés, cambiá por env EMB_MODEL
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

// ---------- util: localizar /data en Netlify, local, etc. ----------
function findDataDir() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const candidates = [
    path.join(__dirname, "data"),
    path.join(__dirname, "..", "data"),
    path.join(__dirname, "..", "..", "data"),
    path.resolve("data"),
    path.join(process.cwd(), "data"),
  ];
  for (const d of candidates) {
    const c = path.join(d, "chunks.jsonl");
    const e = path.join(d, "embeddings.jsonl");
    if (fs.existsSync(c) && fs.existsSync(e)) return d;
  }
  return null;
}

// ---------- util: cargar JSONL o JSON array ----------
function readJsonlOrArray(filePath) {
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  if (!raw) return [];
  // Si empieza con [, es un JSON array
  if (raw[0] === "[") {
    try { return JSON.parse(raw); } catch { return []; }
  }
  // JSONL (una línea por objeto)
  return raw.split("\n").filter(Boolean).map((l) => {
    try { return JSON.parse(l); } catch { return null; }
  }).filter(Boolean);
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

let LOADED = false, DATADIR = null, CHUNKS = [], EMBS = [];

function loadOnce() {
  if (LOADED) return;
  DATADIR = findDataDir();
  if (DATADIR) {
    const chunksPath = path.join(DATADIR, "chunks.jsonl");
    const embsPath   = path.join(DATADIR, "embeddings.jsonl");
    CHUNKS = readJsonlOrArray(chunksPath);
    EMBS   = readJsonlOrArray(embsPath);
  }
  LOADED = true;
}

export default async (req, res) => {
  try {
    loadOnce();

    const body = req.body ? JSON.parse(req.body) : {};
    const query = (body.query || "").trim();
    if (!query) return res.status(400).json({ error: "missing query" });

    if (!DATADIR || !CHUNKS.length || !EMBS.length) {
      return res.status(200).json({
        answer:
`⚠️ Aún no está listo el índice semántico.
Verificá:
1) /docs tiene PDFs (no ZIP).
2) El build generó /data/chunks.jsonl y /data/embeddings.jsonl.
3) netlify.toml incluye:
   [functions]
   directory = "functions"
   node_bundler = "esbuild"
   included_files = ["data/*"]`
      });
    }

    // Embedding de la consulta
    const embResp = await client.embeddings.create({
      model: EMB_MODEL,
      input: query
    });
    const qv = embResp.data[0].embedding;

    // Ranking por coseno
    const scored = EMBS.map((r, i) => ({ i, s: cosine(qv, r.embedding || r.emb || r.vec || []) }))
                       .sort((a, b) => b.s - a.s)
                       .slice(0, 5);

    const ctx = scored.map(({ i }) => {
      const r = EMBS[i];
      // Match por doc + chunk si existen (formato JSONL) o solo por doc (formato array simple)
      const byKeys = CHUNKS.find(c => c.doc === r.doc && c.chunk === r.chunk);
      if (byKeys) return byKeys;
      const alt = CHUNKS.find(c => c.file === r.file || c.doc === r.doc);
      return alt || { text: r.text || "", doc: r.doc || r.file, path: r.path || `docs/${r.doc || r.file}` };
    }).filter(Boolean);

    const bullets = ctx.map(c => '• ' + ((c.text || '').split(/\n+/)[0].slice(0, 240) || c.doc)).join("\n");
    const links = [...new Set(ctx.map(c => c.path || `docs/${c.doc}`))].slice(0, 3)
      .map(p => `<a href="/${p}">PDF</a>`).join(" · ");

    return res.status(200).json({
      answer: `📌 <b>Resumen basado en documentación interna</b>:\n${bullets}\n\n🔗 ${links}`
    });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
};