// functions/chat.js
// Valeria · Seguridad Alimentaria – Netlify Function (CommonJS)
// Versión: v2025-08-13-fix1

const fs = require("fs");
const path = require("path");
const OpenAI = require("openai");

// ---------- Config ----------
const VERSION = "v2025-08-13-fix1";
const DOCS_DIR = process.env.DOCS_DIR || path.join("/var/task", "docs");
const DATA_FILE = process.env.DATA_FILE || path.join("/var/task", "data", "embeddings.json");
const MODEL_GEN = process.env.MODEL_GEN || "gpt-4o-mini";
const MODEL_EMB = process.env.MODEL_EMB || "text-embedding-3-large";
const MAX_CHARS_CTX = 6000;      // contexto enviado al modelo
const TOP_K = 8;                 // chunks a recuperar
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization"
};

// ---------- Estado en frío ----------
let indexCache = null;
let docsCache = null;
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- Utilidades ----------
function uniq(arr) { return [...new Set(arr)]; }

function safeReadJSON(file) {
  try {
    const raw = fs.readFileSync(file, "utf8");
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function loadIndex() {
  if (indexCache) return indexCache;
  const data = safeReadJSON(DATA_FILE);
  if (!data || !Array.isArray(data)) return null;

  // Estructura esperada por build-embeddings.mjs:
  // [{ id, source, title, page, text, embedding: number[] }, ...]
  indexCache = data;
  // cache de lista de docs únicos
  docsCache = uniq(indexCache.map(c => c.source));
  return indexCache;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length && i < b.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) { return Math.sqrt(a.reduce((s, x) => s + x * x, 0)); }
function cosine(a, b) { const n = norm(a) * norm(b); return n ? dot(a, b) / n : 0; }

function takeTopK(items, k) {
  return items.sort((x, y) => y.score - x.score).slice(0, k);
}

function buildPreview(txt, n = 220) {
  if (!txt) return "";
  return txt.replace(/\s+/g, " ").slice(0, n);
}

// expansión liviana de consulta (sin LLM) para debug y recall
function expandQuery(q) {
  const t = (q || "").toLowerCase();
  const extra = [];
  if (/\bbpm\b|buenas practicas|buenas prácticas/.test(t)) {
    extra.push("bpm", "buenas prácticas de manufactura", "manual bpm", "bpma");
  }
  if (/caa|codigo alimentario|código alimentario/.test(t)) {
    extra.push("código alimentario argentino", "caa", "capítulo v rotulación", "capítulo ii establecimientos", "rotulación", "información nutricional");
  }
  if (/poes|procedimiento|saneamiento|sanit/.test(t)) {
    extra.push("poes", "limpieza y desinfección", "sanitización");
  }
  if (/carnicer|pollo|aves|vacuna|porcina/.test(t)) {
    extra.push("carnicería", "recepción de perecederos", "temperaturas", "descongelado");
  }
  return uniq([q, ...extra].filter(Boolean));
}

async function embed(text) {
  const res = await client.embeddings.create({ model: MODEL_EMB, input: text });
  return res.data[0].embedding;
}

function buildSystemPrompt() {
  return [
    "Sos Valeria, especialista en Seguridad Alimentaria para una operación retail argentina.",
    "Tu alcance: BPM/BPMA, CAA (normativa aplicable), POES/MIP y documentos internos de la empresa.",
    "Respuestas: prácticas, en español, con pasos, temperaturas, tolerancias y controles.",
    "No inventes normativa. Si el material recuperado es insuficiente, decí que no encontraste un documento directo y sugerí dónde buscar dentro de los PDFs cargados.",
    "Si corresponde, cerrá con un breve 'Fuentes' listando archivo y página si está.",
    "Si te preguntan algo fuera de alcance, indicá el límite (solo seguridad alimentaria / BPM / CAA / procedimientos internos)."
  ].join(" ");
}

function formatSources(hits) {
  const pairs = hits.map(h => `${h.source}${h.page ? ` p.${h.page}` : ""}`);
  return uniq(pairs).slice(0, 6).join(" · ");
}

async function retrieve(query) {
  const idx = loadIndex();
  if (!idx) return { hits: [], context: "" };

  const qvec = await embed(query);
  const scored = idx.map(ch => ({
    ...ch,
    score: cosine(qvec, ch.embedding || [])
  }));

  const top = takeTopK(scored, TOP_K);
  let ctx = "";
  for (const h of top) {
    if (ctx.length >= MAX_CHARS_CTX) break;
    ctx += `\n### ${h.title || h.source}${h.page ? ` (p.${h.page})` : ""}\n${h.text}\n`;
  }
  return { hits: top, context: ctx.trim() };
}

function json(statusCode, body) {
  return {
    statusCode,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
    body: JSON.stringify(body)
  };
}

// ---------- Handlers de utilería (debug) ----------
function handlePing() {
  const idx = loadIndex();
  return json(200, {
    ok: true,
    version: VERSION,
    docs: docsCache ? docsCache.length : 0,
    embeddings: idx ? idx.length : 0,
    docsDir: DOCS_DIR,
    dataFile: DATA_FILE
  });
}

function handleDiag() {
  let exists = false, size = 0, docsCount = 0;
  try {
    const st = fs.statSync(DATA_FILE);
    exists = st.isFile();
    size = st.size;
  } catch (_) {}
  try {
    docsCount = fs.readdirSync(DOCS_DIR).filter(f => f.toLowerCase().endsWith(".pdf")).length;
  } catch (_) {}
  const idx = loadIndex();
  return json(200, {
    ok: true,
    version: VERSION,
    dataFileExists: exists,
    dataFileSize: size,
    incrustaciones: idx ? idx.length : 0,
    docsDir: DOCS_DIR,
    docsCount
  });
}

async function handleRerank(qs) {
  const q = (qs.query || qs.q || "").toString();
  const expanded = expandQuery(q);

  const idx = loadIndex();
  if (!idx || !q) {
    return json(200, {
      ok: true,
      version: VERSION,
      docs: docsCache ? docsCache.length : 0,
      embeddings: idx ? idx.length : 0,
      query: q,
      expanded: expanded.join(" | "),
      maxScore: 0,
      tops: []
    });
  }

  const qvec = await embed(q);
  const scored = idx.map(ch => ({
    title: ch.title || path.basename(ch.source, ".pdf"),
    source: ch.source,
    page: ch.page,
    preview: buildPreview(ch.text, 260),
    score: cosine(qvec, ch.embedding || [])
  }));

  const tops = takeTopK(scored, 10).map(t => ({
    title: t.title,
    source: t.source,
    score: +t.score.toFixed(4),
    preview: t.preview
  }));

  const maxScore = tops.length ? tops[0].score : 0;
  return json(200, {
    ok: true,
    version: VERSION,
    docs: docsCache.length,
    embeddings: idx.length,
    query: q,
    expanded: expanded.join(" | "),
    maxScore,
    tops
  });
}

// ---------- Handler principal ----------
exports.handler = async (event) => {
  try {
    if (event.httpMethod === "OPTIONS") {
      return { statusCode: 200, headers: CORS_HEADERS, body: "" };
    }

    // Debug endpoints por querystring
    const qs = event.queryStringParameters || {};
    if (qs.ping) return handlePing();
    if (qs.diag) return handleDiag();
    if (qs.rerank) return await handleRerank(qs);

    // Parseo de input (POST JSON con { message, history? })
    if (event.httpMethod !== "POST") {
      return json(405, { ok: false, error: "Método no permitido. Usá POST." });
    }
    const body = JSON.parse(event.body || "{}");
    const userMsg = (body.message || "").toString().trim();
    const history = Array.isArray(body.history) ? body.history : [];

    if (!userMsg) {
      return json(400, { ok: false, error: "Falta 'message'." });
    }
    if (!process.env.OPENAI_API_KEY) {
      return json(500, { ok: false, error: "OPENAI_API_KEY no configurada." });
    }

    // Recuperación
    const { hits, context } = await retrieve(userMsg);

    // Si no hay contexto útil, damos respuesta de alcance
    if (!context) {
      return json(200, {
        ok: true,
        answer:
          "No encontré un documento directo para esa consulta. Probá con: \"POES carnicería\", \"Recepción de perecederos\", \"Fraccionamiento de quesos\".",
        meta: {
          version: VERSION,
          sources: [],
          recovered: 0
        }
      });
    }

    // Construcción del prompt
    const systemPrompt = buildSystemPrompt();
    const sourcesStr = formatSources(hits);

    const messages = [
      { role: "system", content: systemPrompt },
      ...(history || []).slice(-6),
      {
        role: "system",
        content:
          "Contexto recuperado de los documentos internos (usar como evidencia, citar brevemente al final en 'Fuentes'):\n" +
          context
      },
      { role: "user", content: userMsg }
    ];

    const completion = await client.chat.completions.create({
      model: MODEL_GEN,
      messages,
      temperature: 0.2,
      max_tokens: 500
    });

    const answer = completion.choices?.[0]?.message?.content?.trim() || "";

    return json(200, {
      ok: true,
      answer: answer + (sourcesStr ? `\n\nFuentes: ${sourcesStr}` : ""),
      meta: {
        version: VERSION,
        recovered: hits.length,
        topScore: hits.length ? +hits[0].score.toFixed(4) : 0,
        sources: uniq(hits.map(h => h.source)).slice(0, 6)
      }
    });
  } catch (err) {
    return json(500, { ok: false, error: err.message || String(err), version: VERSION });
  }
};