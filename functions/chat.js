// functions/chat.js
import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Rutas a los artefactos empaquetados en la Lambda
const DOCS_DIR = "/var/task/docs";
const DATA_FILE = "/var/task/data/embeddings.json";
const VERSION = "v2025-08-13-fix2";

const json = (obj, status = 200) => ({
  statusCode: status,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(obj),
});

// ---------- utilidades ----------
const normalize = (s = "") =>
  s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, ""); // s/acentos

const dot = (a, b) => {
  const n = Math.min(a?.length || 0, b?.length || 0);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += (a[i] || 0) * (b[i] || 0);
  return sum;
};
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a);
  const nb = norm(b);
  if (!na || !nb) return 0;
  return dot(a, b) / (na * nb);
};

async function embed(text) {
  const { data } = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return data[0].embedding;
}

// ---------- carga y normalización de datos ----------
let CACHE = null;

function pickChunks(raw) {
  // Acepta varias formas: chunks / items / entries / data
  let arr =
    raw?.chunks ||
    raw?.items ||
    raw?.entries ||
    raw?.data ||
    Array.isArray(raw) ? raw : [];

  // Si vino envuelto en { vectors: [...] } o similar, intenta abrir
  if (!Array.isArray(arr) && typeof raw === "object") {
    for (const k of Object.keys(raw)) {
      if (Array.isArray(raw[k])) {
        arr = raw[k];
        break;
      }
    }
  }

  // Mapear a formato canónico
  const mapped = (arr || []).map((c, i) => ({
    id: c.id ?? c.i ?? i,
    source: c.source ?? c.src ?? c.file ?? "",
    title: c.title ?? "",
    text: c.text ?? c.content ?? c.body ?? "",
    vec: c.vec ?? c.embedding ?? c.emb ?? c.v ?? [],
  }));

  return mapped.filter((c) => c.text && Array.isArray(c.vec) && c.vec.length);
}

async function loadData() {
  if (CACHE) return CACHE;

  const exists = fs.existsSync(DATA_FILE);
  const stats = exists ? fs.statSync(DATA_FILE) : null;
  let chunks = [];
  let dims = 0;

  if (exists) {
    try {
      const raw = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
      chunks = pickChunks(raw);
      if (chunks.length && Array.isArray(chunks[0].vec))
        dims = chunks[0].vec.length;
    } catch (e) {
      // si el JSON está corrupto, dejamos chunks=[]
      console.error("Error leyendo embeddings.json:", e);
    }
  }

  // Precalcular texto normalizado para búsquedas fallback
  for (const c of chunks) c._norm = normalize(c.text);

  CACHE = {
    version: VERSION,
    docsCount: fs.existsSync(DOCS_DIR)
      ? fs.readdirSync(DOCS_DIR).filter((f) => f.toLowerCase().endsWith(".pdf"))
          .length
      : 0,
    dataFileExists: !!exists,
    dataFileSize: stats?.size || 0,
    chunks,
    dims,
  };
  return CACHE;
}

// ---------- búsqueda ----------
function topK(chunks, qVec, k = 5) {
  const scored = [];
  for (const c of chunks) {
    if (!c.vec || !c.vec.length) continue;
    const s = cosSim(qVec, c.vec);
    if (s > 0) scored.push({ score: s, title: c.title, source: c.source });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

// ---------- handler ----------
export async function handler(event) {
  const rawUrl = event.rawUrl || "";
  const method = event.httpMethod || "GET";
  const url = new URL(rawUrl);

  // Ruteo “suave” leyendo la URL original antes del redirect
  const isPing = rawUrl.includes("/api/ping");
  const isDiag = rawUrl.includes("/api/diag");

  // GET /api/ping  → salud simple
  if (isPing) {
    return json({ ok: true, version: VERSION });
  }

  // GET /api/diag  → estado / búsqueda de prueba
  if (isDiag) {
    const q = url.searchParams.get("q") || "";
    const k = Math.max(1, Math.min(20, Number(url.searchParams.get("k")) || 5));

    const data = await loadData();

    if (!q) {
      return json({
        ok: true,
        version: data.version,
        docs: data.docsCount,
        dataFileExists: data.dataFileExists,
        dataFileSize: data.dataFileSize,
        embeddings: data.chunks.length,
        dims: data.dims,
        hint:
          'Agregá ?q=texto&k=5 para probar coincidencias. Ej: /api/diag?q=BPM&k=5',
      });
    }

    // con query → probar ANN
    try {
      const qVec = await embed(q);
      const tops = topK(data.chunks, qVec, k);
      const maxScore = tops.length ? tops[0].score : 0;
      return json({
        ok: true,
        version: data.version,
        docs: data.docsCount,
        dataFileExists: data.dataFileExists,
        dataFileSize: data.dataFileSize,
        query: q,
        tops,
        maxScore,
      });
    } catch (e) {
      return json({ ok: false, error: String(e?.message || e) }, 500);
    }
  }

  // POST /api/chat  → conversación
  if (method === "POST") {
    try {
      const body = JSON.parse(event.body || "{}");
      const userQuery = body.query || body.q || "";
      const k = Math.max(1, Math.min(10, Number(body.k) || 5));

      const data = await loadData();
      const q = userQuery || (body.messages || []).map((m) => m?.content).join(" ").slice(0, 2000) || "";

      const qVec = await embed(q);
      const tops = topK(data.chunks, qVec, k);

      // Respuesta mínima: devolvemos fuentes relevantes
      return json({
        ok: true,
        answer:
          tops.length
            ? "Encontré referencias útiles en tus documentos."
            : "No encontré coincidencias claras en los documentos.",
        refs: tops,
      });
    } catch (e) {
      return json({ ok: false, error: String(e?.message || e) }, 500);
    }
  }

  // Cualquier otro método → 405
  return json({ ok: false, error: "Método no permitido. Usá POST." }, 405);
}