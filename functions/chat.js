// functions/chat.js
// Valeria · Netlify Functions (Node 20)
// Rutas:
//   GET  /api/ping                 -> ping rápido (tocable)
//   GET  /api/diag[?q=texto&k=5]   -> estado / búsqueda (tocable)
//   POST /api/diag                 -> idem GET pero con body JSON { q, k }
//   POST /api/chat                 -> conversación RAG { messages:[...] } o { query:"..." }

import fs from "fs";
import path from "path";
import OpenAI from "openai";

const VERSION = "v2025-08-13-getdiag";
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
};
const json = (obj, status = 200) => ({
  statusCode: status,
  headers: { "Content-Type": "application/json; charset=utf-8", ...CORS_HEADERS },
  body: JSON.stringify(obj),
});
const text = (msg, status = 200) => ({
  statusCode: status,
  headers: { "Content-Type": "text/plain; charset=utf-8", ...CORS_HEADERS },
  body: msg,
});

// Ubicaciones empaquetadas por Netlify (process.cwd() -> /var/task)
const DOCS_DIR = path.join(process.cwd(), "docs");
const DATA_FILE = path.join(process.cwd(), "data", "embeddings.json");

// ------- Utilidades -------

function fileExists(p) {
  try { fs.accessSync(p, fs.constants.R_OK); return true; } catch { return false; }
}
function loadEmbeddings() {
  if (!fileExists(DATA_FILE)) return { items: [], meta: { chunks: 0 } };
  const raw = fs.readFileSync(DATA_FILE, "utf8");
  if (!raw) return { items: [], meta: { chunks: 0 } };
  const data = JSON.parse(raw);
  // Soportar {items:[...]} o array directo
  const items = Array.isArray(data) ? data : (data.items ?? []);
  const meta  = Array.isArray(data) ? { chunks: items.length } : (data.meta ?? { chunks: items.length });
  return { items, meta };
}

function dot(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s;
}
function norm(a) {
  return Math.sqrt(a.reduce((s, v) => s + v * v, 0));
}
function cosine(a, b) {
  const na = norm(a), nb = norm(b);
  if (!na || !nb) return 0;
  return dot(a, b) / (na * nb);
}

function pickModel(pref = "gpt-4o-mini") {
  // Permite override por env
  return process.env.CHAT_MODEL || pref;
}

async function embedQuery(client, text) {
  const model = process.env.EMBED_MODEL || "text-embedding-3-small";
  const r = await client.embeddings.create({ model, input: text });
  return r.data[0].embedding;
}

function baseStats() {
  const docsCount =
    fileExists(DOCS_DIR)
      ? fs.readdirSync(DOCS_DIR).filter(n => n.toLowerCase().endsWith(".pdf")).length
      : 0;

  const dataFileExists = fileExists(DATA_FILE);
  const dataFileSize = dataFileExists ? fs.statSync(DATA_FILE).size : 0;

  const { items, meta } = loadEmbeddings();
  return {
    ok: true,
    version: VERSION,
    docsDir: "/var/task/docs",
    dataFile: "/var/task/data/embeddings.json",
    docs: docsCount,
    embeddings: items.length,
    chunks: meta?.chunks ?? items.length,
    dataFileExists,
    dataFileSize,
  };
}

// ------- Handlers -------

async function handlePing() {
  return json(baseStats());
}

async function handleDiagGet(url, body) {
  const params = url.searchParams;
  const q = (body?.q ?? params.get("q") ?? "").trim();
  const k = Math.max(1, Math.min(10, parseInt(body?.k ?? params.get("k") ?? "5", 10)));

  const stats = baseStats();
  if (!q) {
    // Sólo estado
    return json({ ...stats, query: "", tops: [] });
  }

  const { items } = loadEmbeddings();
  if (!items.length) {
    return json({ ...stats, query: q, tops: [], maxScore: 0 });
  }

  // Si hay API key, embebemos y rankeamos
  const key = process.env.OPENAI_API_KEY ?? "";
  if (!key) {
    return json({
      ...stats,
      query: q,
      error: "Falta OPENAI_API_KEY para calcular similitud en /diag",
      tops: [],
      maxScore: 0,
    }, 200);
  }

  const client = new OpenAI({ apiKey: key });
  const qvec = await embedQuery(client, q);

  // items: esperamos { id, title, source, vector, preview?, score? }
  const scored = items.map(it => {
    const v = it.vector || it.embedding || it.vec || [];
    const score = v.length ? cosine(qvec, v) : 0;
    return {
      title: it.title || it.file || it.source || "chunk",
      source: it.source || it.file || it.path || "",
      preview: (it.preview || it.text || "").slice(0, 200),
      score,
    };
  }).sort((a, b) => b.score - a.score);

  const tops = scored.slice(0, k);
  const maxScore = tops.length ? tops[0].score : 0;

  return json({
    ...stats,
    query: q,
    maxScore: Number(maxScore.toFixed(4)),
    tops: tops.map(t => ({ ...t, score: Number(t.score.toFixed(4)) })),
  });
}

async function handleChatPost(url, body) {
  const key = process.env.OPENAI_API_KEY ?? "";
  if (!key) return json({ ok: false, error: "Falta OPENAI_API_KEY" }, 500);

  const client = new OpenAI({ apiKey: key });

  const userQuery =
    body?.query ??
    body?.q ??
    (Array.isArray(body?.messages)
      ? (body.messages.find(m => m.role === "user")?.content ?? "")
      : "");

  const { items } = loadEmbeddings();
  let contextBlocks = [];

  if (userQuery && items.length) {
    const qvec = await embedQuery(client, userQuery);
    const ranked = items.map(it => {
      const v = it.vector || it.embedding || it.vec || [];
      const score = v.length ? cosine(qvec, v) : 0;
      return { it, score };
    }).sort((a, b) => b.score - a.score).slice(0, 6);

    contextBlocks = ranked.map(({ it }) => `Título: ${it.title || it.source}\nFragmento: ${(it.text || it.preview || "").slice(0, 750)}`);
  }

  const system =
    "Sos Valeria, asistente de Seguridad Alimentaria. Respondé breve, claro y con pasos accionables. " +
    "Limitá tus respuestas a normativa argentina (CAA), BPM/BPMA y procedimientos internos. " +
    "Si no estás segura, pedí precisión o sugerí documentos del repositorio.";

  const prompt =
    (contextBlocks.length
      ? `Usá SOLO la información de estas fuentes para responder:\n\n${contextBlocks.join("\n---\n")}\n\n`
      : "No se encontraron fuentes relevantes; respondé de forma general y SUAVE sobre el tema sin inventar normativa.\n\n") +
    `Pregunta: ${userQuery}`;

  const model = pickModel();
  const completion = await client.chat.completions.create({
    model,
    temperature: 0.2,
    messages: [
      { role: "system", content: system },
      ...(Array.isArray(body?.messages) ? body.messages.filter(m => m.role !== "system") : [{ role: "user", content: userQuery }]),
      { role: "user", content: prompt },
    ],
  });

  const answer = completion.choices?.[0]?.message?.content?.trim() || "No pude generar respuesta.";
  return json({ ok: true, model, answer });
}

// ------- Router -------

export async function handler(event) {
  try {
    if (event.httpMethod === "OPTIONS") return json({}, 204);

    // Reconstruimos URL para rutear
    const host = event.headers["x-forwarded-host"] || event.headers.host || "localhost";
    const proto = (event.headers["x-forwarded-proto"] || "https");
    const rawUrl = event.rawUrl || `${proto}://${host}${event.path}${event.rawQuery ? "?" + event.rawQuery : ""}`;
    const url = new URL(rawUrl);
    const p = url.pathname;

    // Normalizamos (soporta /api/* y /.netlify/functions/chat/*)
    const is = (name) =>
      p.endsWith(`/api/${name}`) ||
      p.endsWith(`/.netlify/functions/chat/${name}`) ||
      p.endsWith(`${name}`) && p.includes("/api/");

    // PING: GET tocable
    if (event.httpMethod === "GET" && (is("ping") || p.endsWith("/api") || p.endsWith("/api/")))
      return handlePing();

    // DIAG: GET o POST
    if ((event.httpMethod === "GET" && is("diag")) || (event.httpMethod === "POST" && is("diag"))) {
      const body = event.httpMethod === "POST" && event.body ? JSON.parse(event.body) : null;
      return handleDiagGet(url, body);
    }

    // CHAT: sólo POST
    if (is("chat")) {
      if (event.httpMethod !== "POST")
        return json({ ok: false, error: "Método no permitido. Usá POST." }, 405);

      const body = event.body ? JSON.parse(event.body) : {};
      return handleChatPost(url, body);
    }

    // Si cae aquí, mostramos un índice simple
    return json({
      ok: true,
      version: VERSION,
      routes: {
        "GET /api/ping": "Estado rápido (tocable desde el navegador).",
        "GET /api/diag": "Estado; agregar ?q=texto&k=5 para ver coincidencias.",
        "POST /api/diag": "Body JSON { q, k }",
        "POST /api/chat": "Body JSON { query } o { messages:[...] }",
      },
    });
  } catch (err) {
    console.error(err);
    return json({ ok: false, error: String(err?.message || err) }, 500);
  }
}