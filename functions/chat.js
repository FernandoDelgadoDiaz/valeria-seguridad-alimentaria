// netlify/functions/chat.js
// CommonJS para mayor compatibilidad en Netlify Functions (Node 18/20)

const fs = require("fs");
const path = require("path");

// ---------- Helpers de paths ----------
// Busca un archivo relativo al root del bundle de la función.
function resolveDataPath(rel) {
  const roots = [
    process.env.LAMBDA_TASK_ROOT,                // root del bundle en producción
    path.dirname(require.main.filename),         // carpeta del handler
    process.cwd(),                               // fallback
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
    const raw = fs.readFileSync(absPath, "utf8");
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

// Cargamos embeddings y (opcional) índice de documentos.
const DATA_FILE = resolveDataPath("data/embeddings.json");
const DOCS_DIR_A = resolveDataPath("docs");
const DOCS_DIR_B = resolveDataPath("documentos");

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

// ---------- Endpoints de diagnóstico ----------

function json(body, statusCode = 200) {
  return {
    statusCode,
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  };
}

// Búsqueda muy simple por texto (fallback para debug)
function simpleSearch(store, query, limit = 5) {
  if (!query) return [];
  const q = String(query).toLowerCase();

  // El formato de tu embeddings.json puede variar.
  // Soportamos dos variantes comunes:
  //  - { chunks: [{ text, title, source, ... }], ... }
  //  - [{ text, title, source, ... }, ...]
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
        score: 1, // al ser keyword match, dejamos score simbólico
        preview: text.slice(0, 220),
      });
      if (hits.length >= limit) break;
    }
  }
  return hits;
}

// ---------- Handler principal ----------
exports.handler = async (event) => {
  const qs = event.queryStringParameters || {};
  const store = getStore();

  // /chat?ping=1 → para ver conteos
  if (qs.ping) {
    const pdfs = countPDFs(DOCS_DIR_A) + countPDFs(DOCS_DIR_B);
    const embCount =
      (Array.isArray(store?.embeddings) && store.embeddings.length) ||
      (Array.isArray(store?.chunks) && store.chunks.length) ||
      (Array.isArray(store) && store.length) ||
      0;

    return json({
      ok: true,
      version: "v2025-08-13-fix1",
      docs: pdfs,
      embeddings: embCount,
      dataFileSeen: Boolean(DATA_FILE),
    });
  }

  // /chat?diag=1 → info de rutas dentro del bundle (muy útil si ping da 0)
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
        docsDirA: DOCS_DIR_A,
        docsDirB: DOCS_DIR_B,
        dataFileExists: Boolean(DATA_FILE && fs.existsSync(DATA_FILE)),
        docsCount: countPDFs(DOCS_DIR_A) + countPDFs(DOCS_DIR_B),
      },
    });
  }

  // /chat?debug=1&q=...
  if (qs.debug) {
    const q = (qs.q || "").toString();
    const pdfs = countPDFs(DOCS_DIR_A) + countPDFs(DOCS_DIR_B);
    const embCount =
      (Array.isArray(store?.embeddings) && store.embeddings.length) ||
      (Array.isArray(store?.chunks) && store.chunks.length) ||
      (Array.isArray(store) && store.length) ||
      0;

    const tops = simpleSearch(store, q, 5);
    const expanded = [
      q,
      "caa",
      "capítulo v rotulación",
      "capítulo ii establecimientos",
      "información nutricional",
      "bpm",
      "manual bpm",
      "poes",
      "mip",
    ];

    return json({
      ok: true,
      version: "v2025-08-13-fix1",
      docs: pdfs,
      embeddings: embCount,
      query: q,
      expanded,
      maxScore: tops.length ? tops[0].score : 0,
      tops,
    });
  }

  // ---------- Aquí va tu lógica normal de chat ----------
  // (Mantuvimos este archivo enfocado en diagnóstico y carga de data.
  //  Pegá debajo la parte de conversación/LLM que ya tenías.)

  return json({ ok: true, message: "Endpoint /chat activo" });
};