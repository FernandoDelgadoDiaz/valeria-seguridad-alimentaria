// functions/chat.js  (Netlify Function)
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

function findDataDir() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const cands = [
    path.join(__dirname, "data"),
    path.join(__dirname, "..", "data"),
    path.join(process.cwd(), "data"),
    path.resolve("data")
  ];
  for (const d of cands) {
    const need = ["chunks.jsonl", "embeddings.jsonl", "files_index.json"].map(f => path.join(d, f));
    if (need.every(p => fs.existsSync(p))) return d;
  }
  return null;
}
function readJsonlOrArray(p) {
  const raw = fs.readFileSync(p, "utf-8").trim();
  if (!raw) return [];
  if (raw[0] === "[") { try { return JSON.parse(raw); } catch { return []; } }
  return raw.split("\n").filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
}
function norm(s) {
  return (s || "")
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[_\-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
function cosine(a, b) {
  let d = 0, na = 0, nb = 0, n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { const x = a[i], y = b[i]; d += x * y; na += x * x; nb += y * y; }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

// carga única
let LOADED = false, DATADIR = null, CHUNKS = [], EMBS = [], FILES = [];
function loadOnce() {
  if (LOADED) return;
  DATADIR = findDataDir();
  if (DATADIR) {
    CHUNKS = readJsonlOrArray(path.join(DATADIR, "chunks.jsonl"));
    EMBS   = readJsonlOrArray(path.join(DATADIR, "embeddings.jsonl"));
    FILES  = JSON.parse(fs.readFileSync(path.join(DATADIR, "files_index.json"), "utf-8"));
  }
  LOADED = true;
}

function bestFileByName(query) {
  const m = norm(query).match(/(?:dame\s+)?pdf\s+de\s+(.+)$/i);
  const target = m ? m[1] : "";
  if (!target) return null;
  const qTokens = new Set(target.split(" ").filter(Boolean));
  let best = null;
  for (const f of FILES) {
    const name = f.norm || norm(f.file || "");
    const tokens = new Set((f.tokens && f.tokens.length ? f.tokens : name.split(" ")).filter(Boolean));
    const overlap = [...qTokens].filter(t => tokens.has(t)).length;
    const inc = name.includes(target) ? 1 : 0;
    const score = overlap + inc;
    if (score > 0 && (!best || score > best.score)) best = { ...f, score };
  }
  return best;
}

export async function handler(event) {
  try {
    loadOnce();
    const { query = "" } = event.body ? JSON.parse(event.body) : {};
    if (!query) {
      return { statusCode: 400, body: JSON.stringify({ error: "missing query" }) };
    }

    if (!DATADIR || !EMBS.length || !CHUNKS.length || !FILES.length) {
      return {
        statusCode: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer:
`⚠️ Índice no listo.
Verificá:
1) /docs con PDFs.
2) Build generó /data/chunks.jsonl, /data/embeddings.jsonl, /data/files_index.json.
3) netlify.toml incluye included_files = ["data/*"].`
        })
      };
    }

    // 1) Si pide PDF → búsqueda por nombre
    if (/pdf/i.test(query)) {
      const hit = bestFileByName(query);
      if (hit) {
        return {
          statusCode: 200,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ answer: `📄 <b>Documento:</b> <a href="/${hit.path}">${hit.file}</a>` })
        };
      }
    }

    // 2) RAG
    const emb = await client.embeddings.create({ model: EMB_MODEL, input: query });
    const qv = emb.data[0].embedding;
    const top = EMBS.map((r, i) => ({ i, s: cosine(qv, r.embedding) }))
                    .sort((a, b) => b.s - a.s).slice(0, 5);
    const ctx = top.map(({ i }) => {
      const r = EMBS[i];
      return CHUNKS.find(c => c.doc === r.doc && c.chunk === r.chunk);
    }).filter(Boolean);

    const bullets = ctx.map(c => '• ' + (c.text || '').split(/\n+/)[0].slice(0, 240)).join("\n");
    const links = [...new Set(ctx.map(c => c.path))].slice(0, 3).map(p => `<a href="/${p}">PDF</a>`).join(" · ");

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        answer: bullets ? `📌 <b>Resumen basado en documentación</b>:\n${bullets}\n\n🔗 ${links}`
                        : `No encontré suficiente contexto. Probá con algo más específico: "temperaturas de conservación", "POES sector pollos".`
      })
    };
  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ error: String(err) }) };
  }
}