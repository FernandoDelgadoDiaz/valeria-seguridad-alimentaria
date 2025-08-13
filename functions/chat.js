// functions/chat.js
const fs = require("fs");
const path = require("path");

function resolveDataPath(rel) {
  const roots = [
    process.env.LAMBDA_TASK_ROOT,
    path.dirname(require.main.filename),
    process.cwd(),
  ].filter(Boolean);

  for (const root of roots) {
    const p = path.join(root, rel);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function loadJSONSafe(absPath) {
  try {
    if (!absPath || !fs.existsSync(absPath)) return null;
    return JSON.parse(fs.readFileSync(absPath, "utf8"));
  } catch {
    return null;
  }
}

const DATA_FILE = resolveDataPath("data/embeddings.json");
const DOCS_DIR = resolveDataPath("docs");

let STORE = null;
function getStore() {
  if (!STORE) STORE = loadJSONSafe(DATA_FILE) || {};
  return STORE;
}

function countPDFs(dir) {
  try {
    if (!dir) return 0;
    return fs.readdirSync(dir).filter(f => f.toLowerCase().endsWith(".pdf")).length;
  } catch {
    return 0;
  }
}

function json(body, statusCode = 200) {
  return {
    statusCode,
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  };
}

function simpleSearch(store, query, limit = 5) {
  if (!query) return [];
  const q = String(query).toLowerCase();
  const chunks = Array.isArray(store?.chunks) ? store.chunks
               : Array.isArray(store) ? store
               : [];
  const hits = [];
  for (const ch of chunks) {
    const text = (ch.text || ch.content || "").toLowerCase();
    if (!text) continue;
    if (text.includes(q)) {
      hits.push({
        title: ch.title || ch.source || "fragmento",
        source: ch.source || ch.file || "desconocido",
        score: 1,
        preview: text.slice(0, 220),
      });
      if (hits.length >= limit) break;
    }
  }
  return hits;
}

exports.handler = async (event) => {
  const qs = event.queryStringParameters || {};
  const store = getStore();

  if (qs.ping) {
    return json({
      ok: true,
      version: "v2025-08-13-fix1",
      docs: countPDFs(DOCS_DIR),
      embeddings:
        (Array.isArray(store?.embeddings) && store.embeddings.length) ||
        (Array.isArray(store?.chunks) && store.chunks.length) ||
        (Array.isArray(store) && store.length) || 0,
      dataFileSeen: Boolean(DATA_FILE),
    });
  }

  if (qs.diag) {
    return json({
      ok: true,
      version: "v2025-08-13-fix1",
      env: {
        LAMBDA_TASK_ROOT: process.env.LAMBDA_TASK_ROOT || null,
        cwd: process.cwd(),
        handlerDir: path.dirname(require.main.filename),
      },
      paths: {
        dataFile: DATA_FILE,
        dataFileExists: Boolean(DATA_FILE && fs.existsSync(DATA_FILE)),
        docsDir: DOCS_DIR,
        docsCount: countPDFs(DOCS_DIR),
      },
    });
  }

  if (qs.debug) {
    const q = (qs.q || "").toString();
    const tops = simpleSearch(store, q, 5);
    return json({
      ok: true,
      version: "v2025-08-13-fix1",
      docs: countPDFs(DOCS_DIR),
      embeddings:
        (Array.isArray(store?.embeddings) && store.embeddings.length) ||
        (Array.isArray(store?.chunks) && store.chunks.length) ||
        (Array.isArray(store) && store.length) || 0,
      query: q,
      expanded: [q, "bpm", "caa", "capítulo v rotulación", "capítulo ii establecimientos", "información nutricional", "poes", "mip"],
      maxScore: tops.length ? tops[0].score : 0,
      tops,
    });
  }

  // --- Aquí iría tu lógica de conversación normal (LLM/RAG) ---
  return json({ ok: true, message: "Endpoint /chat activo" });
};