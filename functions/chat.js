// functions/chat.js
// ─────────────────────────────────────────────────────────────
// Valeria · RAG para Seguridad Alimentaria (Argentina)
// - GET  /.netlify/functions/chat?ping=1                → ping
// - GET  /.netlify/functions/chat?debug=1&q=...         → ver top-k del RAG
// - POST /.netlify/functions/chat   { message }         → respuesta usando RAG
//
// Estructura esperada: data/embeddings.json = array de chunks
//  { text | content, source, title, embedding: number[] }
//
// Node 18 en Netlify. Sin dependencias externas.

const fs = require("fs");
const path = require("path");

// ── Config general ───────────────────────────────────────────
const VERSION = (() => {
  const d = new Date();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `v${d.getFullYear()}-${mm}-${dd}-rag`;
})();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

// RAG: umbrales y límites (ajustables por ENV si querés)
const RAG_MIN_SCORE = Number(process.env.RAG_MIN_SCORE || 0.28);
const RAG_MIN_CHUNKS = Number(process.env.RAG_MIN_CHUNKS || 3);
const RAG_TOP_K = Number(process.env.RAG_TOP_K || 8);
const MAX_CONTEXT_CHUNKS = Number(process.env.MAX_CONTEXT_CHUNKS || 6);

// CORS básico para el front
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};

// ── Utilidades ───────────────────────────────────────────────
let VDB = null;
function loadVDB() {
  if (VDB) return VDB;
  // data/embeddings.json está una carpeta arriba de /functions
  const file = path.join(__dirname, "..", "data", "embeddings.json");
  const raw = fs.readFileSync(file, "utf8");
  const arr = JSON.parse(raw);

  // Normalizamos campos mínimos
  VDB = arr
    .filter((c) => Array.isArray(c.embedding) && (c.text || c.content))
    .map((c) => ({
      text: c.text || c.content || "",
      source: c.source || c.doc || c.file || "desconocido",
      title: c.title || c.source || c.doc || "fragmento",
      embedding: c.embedding,
    }));

  return VDB;
}

function uniqueDocsCount(chunks) {
  return new Set(chunks.map((c) => c.source)).size;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  return Math.sqrt(dot(a, a));
}
function cosine(a, b) {
  const na = norm(a);
  const nb = norm(b);
  return na && nb ? dot(a, b) / (na * nb) : 0;
}

async function embed(text) {
  if (!OPENAI_API_KEY) throw new Error("Falta OPENAI_API_KEY");
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: EMB_MODEL, input: text }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Embeddings error ${res.status}: ${t}`);
  }
  const data = await res.json();
  return data.data[0].embedding;
}

// Expande consultas muy cortas con contexto del dominio
function expandUserQuery(q) {
  q = (q || "").trim();
  const contexto =
    "retail de alimentos en Argentina: BPM, CAA, POES, MIP, limpieza y sanitización, rotulado, temperaturas, carnicería, fiambrería, panadería y lácteos";
  if (q.length < 24) return `${q}. Contexto: ${contexto}`;
  return `${q}. Contexto: ${contexto}`;
}

async function search(queryEmbedding, k = RAG_TOP_K) {
  const base = loadVDB();
  const scored = base.map((c) => ({
    title: c.title,
    source: c.source,
    text: c.text,
    score: cosine(queryEmbedding, c.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

function previewOf(t) {
  return t.replace(/\s+/g, " ").slice(0, 300);
}

async function callChat(systemPrompt, userPrompt) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      temperature: 0.2,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Chat error ${res.status}: ${t}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
}

function json(statusCode, body) {
  return {
    statusCode,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS },
    body: JSON.stringify(body),
  };
}

// ── Handler Netlify ──────────────────────────────────────────
exports.handler = async (event) => {
  try {
    if (event.httpMethod === "OPTIONS") {
      return { statusCode: 204, headers: CORS, body: "" };
    }

    // Carga VDB (para ping y demás)
    const vdb = loadVDB();
    const counts = { docs: uniqueDocsCount(vdb), embeddings: vdb.length };

    // Ping
    const qs = event.queryStringParameters || {};
    if (qs.ping) {
      return json(200, { ok: true, version: VERSION, ...counts });
    }

    // Debug RAG
    if (qs.debug) {
      const rawQ = String(qs.q || "").trim();
      if (!rawQ) return json(400, { ok: false, error: "Falta ?q=" });
      const q = expandUserQuery(rawQ);
      const qEmb = await embed(q);
      const tops = await search(qEmb, RAG_TOP_K);
      const payload = {
        ok: true,
        version: VERSION,
        ...counts,
        query: rawQ,
        maxScore: tops[0]?.score || 0,
        tops: tops.map((t) => ({
          title: t.title,
          source: t.source,
          score: Number(t.score.toFixed(4)),
          preview: previewOf(t.text),
        })),
      };
      return json(200, payload);
    }

    // Chat (POST preferido)
    if (event.httpMethod === "POST") {
      if (!OPENAI_API_KEY) return json(500, { ok: false, error: "Falta OPENAI_API_KEY" });

      const body = JSON.parse(event.body || "{}");
      const userQueryRaw = String(body.message || "").trim();
      if (!userQueryRaw) return json(400, { ok: false, error: "Falta { message }" });

      // 1) Embedding + búsqueda
      const userQuery = expandUserQuery(userQueryRaw);
      const qEmb = await embed(userQuery);
      const tops = await search(qEmb, RAG_TOP_K);

      const maxScore = tops[0]?.score || 0;
      const suficientes = tops.filter((x) => x.score >= RAG_MIN_SCORE).length >= RAG_MIN_CHUNKS;

      // 2) Si no hay suficiente señal, da una guía corta (pero útil)
      if (!suficientes) {
        return json(200, {
          ok: true,
          reply:
            "Necesito un poco más de precisión para ubicarlo en los documentos. Probá con: “BPM en panadería”, “temperaturas de heladera según CAA”, “POES carnicería: sanitizante”.",
          meta: { version: VERSION, ...counts, maxScore: Number(maxScore.toFixed(4)) },
        });
      }

      // 3) Armar contexto y pedir respuesta
      const contextChunks = tops
        .filter((x) => x.score >= RAG_MIN_SCORE)
        .slice(0, MAX_CONTEXT_CHUNKS);

      const contextText = contextChunks
        .map(
          (c, i) =>
            `[#${i + 1}] Título: ${c.title} (src: ${c.source})\n${c.text}`
        )
        .join("\n\n");

      const systemPrompt = `
Sos **Valeria**, especialista en Seguridad Alimentaria para retail de alimentos en Argentina.
Tu objetivo es dar respuestas prácticas, accionables y seguras, usando SIEMPRE la evidencia de los documentos suministrados (BPM, CAA, POES, MIP y documentos internos).
Escribe en español rioplatense, con pasos, puntos clave y advertencias. Incluí límites numéricos (temperaturas, concentraciones, tiempos) cuando corresponda.
Referenciá los fragmentos usados con [#n] (según el contexto provisto). Si algo no está en las fuentes, acláralo.
`;

      const userPrompt = `
Consulta: ${userQueryRaw}

Usá SOLO el siguiente contexto para responder. No inventes datos.
${contextText}

Entrega:
- Pasos/puntos concretos para la operación diaria.
- Valores numéricos cuando apliquen (°C, ppm, días).
- Cita breve con [#n] al final de cada afirmación relevante.
`;

      const reply = await callChat(systemPrompt, userPrompt);

      // 4) Empaquetar respuesta + citas básicas
      const cites = contextChunks.map((c, i) => ({
        n: i + 1,
        title: c.title,
        source: c.source,
      }));

      return json(200, {
        ok: true,
        reply,
        citations: cites,
        meta: {
          version: VERSION,
          ...counts,
          maxScore: Number(maxScore.toFixed(4)),
        },
      });
    }

    // Si alguien hace GET sin debug, devolvemos info básica
    return json(200, {
      ok: true,
      version: VERSION,
      ...counts,
      tip:
        "Usá POST con { message } para chatear o GET con ?debug=1&q=... para ver los resultados del RAG.",
    });
  } catch (err) {
    return json(500, { ok: false, error: String(err.message || err) });
  }
};