// functions/chat.js
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DOCS_DIR = path.resolve("docs");
const DATA_FILE = path.resolve("data", "embeddings.json");
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-4o-mini";
const EMBED_MODEL = "text-embedding-3-small";

let CACHE = { loaded: false, meta: null, chunks: [] };

function loadEmbeddings() {
  if (!fs.existsSync(DATA_FILE)) return;
  const raw = fs.readFileSync(DATA_FILE, "utf8");
  const data = JSON.parse(raw);
  const chunks = Array.isArray(data) ? data : (data.chunks || []);
  CACHE = { loaded: true, meta: data.meta || null, chunks };
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

async function retrieve(query, k = 5) {
  if (!CACHE.loaded) loadEmbeddings();
  if (CACHE.chunks.length === 0) return [];
  const { data } = await client.embeddings.create({ model: EMBED_MODEL, input: query });
  const qv = data[0].embedding;
  const scored = CACHE.chunks.map(c => ({ ...c, score: cosine(qv, c.embedding) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

function json(status, body) {
  return {
    statusCode: status,
    headers: { "content-type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  };
}

export async function handler(event) {
  const url = new URL(event.rawUrl);
  const path = url.pathname;
  const method = event.httpMethod;

  // --- GET /api/ping ---
  if (method === "GET" && path.endsWith("/api/ping")) {
    const exists = fs.existsSync(DATA_FILE);
    const size = exists ? fs.statSync(DATA_FILE).size : 0;
    const docs = fs.existsSync(DOCS_DIR) ? fs.readdirSync(DOCS_DIR).filter(f => f.endsWith(".pdf")).length : 0;
    return json(200, { ok: true, version: "v2025-08-13-fix2", docs, dataFileExists: exists, dataFileSize: size });
  }

  // --- GET /api/diag ---
  if (method === "GET" && path.endsWith("/api/diag")) {
    const q = url.searchParams.get("q") || "";
    const k = Number(url.searchParams.get("k") || 5);
    if (!CACHE.loaded) loadEmbeddings();
    const exists = fs.existsSync(DATA_FILE);
    const size = exists ? fs.statSync(DATA_FILE).size : 0;
    const docs = fs.existsSync(DOCS_DIR) ? fs.readdirSync(DOCS_DIR).filter(f => f.endsWith(".pdf")).length : 0;

    let tops = [];
    if (q && CACHE.chunks.length > 0) {
      tops = await retrieve(q, k);
      tops = tops.map(t => ({ title: t.source, source: t.source, score: Number(t.score.toFixed(4)) }));
    }

    return json(200, {
      ok: true,
      version: "v2025-08-13-fix2",
      docs,
      embeddings: CACHE.chunks.length,
      dataFileExists: exists,
      dataFileSize: size,
      query: q || undefined,
      tops,
      maxScore: tops[0]?.score || 0
    });
  }

  // --- POST /api/chat ---
  if (method !== "POST" || !path.endsWith("/api/chat")) {
    return json(405, { ok: false, error: "Método no permitido. Usá POST." });
  }

  let body;
  try { body = JSON.parse(event.body || "{}"); }
  catch { return json(400, { ok: false, error: "Body JSON inválido." }); }

  const userQuery = (body.query || body.q || "").toString().trim();
  if (!userQuery) return json(400, { ok: false, error: "Falta 'query'." });

  if (!CACHE.loaded) loadEmbeddings();
  const context = await retrieve(userQuery, 6);
  const contextText = context.map((c, i) => `#${i+1} [${c.source}] ${c.text}`).join("\n\n");

  const system = `Sos Valeria (seguridad alimentaria). Solo respondé con base en los fragmentos.\nSi no hay contexto suficiente, pedí reformular o sugerí términos (POES carnicería, recepción de perecederos, etc.).`;
  const messages = [
    { role: "system", content: system },
    { role: "user", content: `Consulta: ${userQuery}\n\nFragmentos:\n${contextText || "(sin coincidencias)"}` }
  ];

  const completion = await client.chat.completions.create({
    model: CHAT_MODEL,
    temperature: 0.2,
    messages
  });

  return json(200, {
    ok: true,
    answer: completion.choices?.[0]?.message?.content || "",
    used: { k: context.length, model: CHAT_MODEL, embedModel: EMBED_MODEL }
  });
}