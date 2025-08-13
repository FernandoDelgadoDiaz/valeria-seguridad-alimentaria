// v2025-08-13-fix1  —  Chat + RAG + endpoints ping/diag/debug
// Funciona con: /data/embeddings.json y PDFs en /docs/*

const fs = require("fs");
const path = require("path");
const OpenAI = require("openai");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ------------------------ utils de paths ------------------------
function resolveDataPath(file = "embeddings.json") {
  const candidates = [
    path.join(__dirname, "../data", file),
    path.join(__dirname, "../../data", file),
    path.join(process.cwd(), "data", file),
    path.resolve("data", file),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return candidates[0]; // último intento (aunque no exista)
}

function listDocsDir() {
  const candidates = [
    path.join(__dirname, "../docs"),
    path.join(__dirname, "../../docs"),
    path.join(process.cwd(), "docs"),
    path.resolve("docs"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return candidates[0];
}

// ------------------------ carga de embeddings ------------------------
let EMB = null;
function loadEmbeddings() {
  if (EMB) return EMB;
  const file = resolveDataPath("embeddings.json");
  if (!fs.existsSync(file)) {
    EMB = { meta: { chunks: 0 }, chunks: [] };
    return EMB;
  }
  EMB = JSON.parse(fs.readFileSync(file, "utf8"));
  // formatos esperados: {meta?, chunks:[{text, source, title, embedding:[]}, ...]}
  if (!EMB.chunks && Array.isArray(EMB)) EMB = { meta: {}, chunks: EMB };
  if (!EMB.meta) EMB.meta = {};
  EMB.meta.file = file;
  EMB.meta.chunks = EMB.chunks?.length || 0;
  return EMB;
}

// ------------------------ embedding y similitud ------------------------
async function embedQuery(q) {
  const res = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: q,
  });
  return res.data[0].embedding;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom ? dot / denom : 0;
}

function expandQuery(q) {
  const base = (q || "").toLowerCase();
  const synonyms = {
    bpm: ["bpm","buenas prácticas de manufactura","manual bpm","bpma"],
    caa: ["caa","código alimentario argentino","código alimentario"],
    poes: ["poes","procedimientos operativos estandarizados"],
    mip: ["mip","manejo integrado de plagas","manual mip"],
    fraccionado: ["fraccionado","fraccionamiento de quesos","etiquetado hormas"],
    temperatura: ["temperatura","0–5 °c","cadena de frío","refrigeración"],
  };
  let extra = [];
  for (const k of Object.keys(synonyms)) {
    if (base.includes(k)) extra = extra.concat(synonyms[k]);
  }
  // siempre agregamos algunas claves del negocio
  extra.push("carnicería","recepción de perecederos","rotulación","capítulo v rotulación","capítulo ii establecimientos");
  const uniq = Array.from(new Set([base, ...extra])).filter(Boolean);
  return uniq.join(" | ");
}

async function retrieve(q, k = 6) {
  const data = loadEmbeddings();
  if (!data.chunks || data.chunks.length === 0) {
    return { tops: [], maxScore: 0 };
  }
  const expanded = expandQuery(q);
  const qEmb = await embedQuery(expanded);
  const scored = data.chunks.map((c) => ({
    ...c,
    score: c.embedding ? cosine(qEmb, c.embedding) : 0,
  }));
  scored.sort((a, b) => b.score - a.score);
  const tops = scored.slice(0, k).map(c => ({
    title: c.title || c.source || "",
    source: c.source || "",
    score: Number(c.score.toFixed(4)),
    preview: (c.text || "").slice(0, 260),
  }));
  return { tops, maxScore: tops[0]?.score || 0, expanded };
}

// ------------------------ respuestas ------------------------
function valeriaSystemPrompt() {
  return [
    {
      role: "system",
      content:
        "Eres Valeria, especialista en seguridad alimentaria para retail. " +
        "Respondes en español, breve y accionable. Prioriza BPM/CAA y los POES internos. " +
        "Si la pregunta es ambigua, sugiere el documento interno más pertinente. " +
        "Incluye pasos concretos, rangos de temperatura y controles cuando corresponda.",
    },
  ];
}

function buildContextBlock(tops) {
  if (!tops?.length) return "No hay contexto.";
  return tops
    .map((t, i) => `[#${i + 1}] ${t.title} — ${t.source}\n${t.preview}`)
    .join("\n\n");
}

async function answerWithRAG(userText) {
  const { tops } = await retrieve(userText, 6);
  const context = buildContextBlock(tops);
  const messages = [
    ...valeriaSystemPrompt(),
    {
      role: "user",
      content:
        "Pregunta: " + userText + "\n\n" +
        "Fragmentos de contexto (usar cuando ayuden, NO inventes):\n" +
        context +
        "\n\nResponde de forma práctica para operación diaria. Si corresponde, sugiere el nombre del documento interno exacto.",
    },
  ];
  const chat = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages,
    temperature: 0.2,
  });
  const text = chat.choices[0].message.content.trim();
  return { text, sources: tops };
}

// ------------------------ endpoints de diagnóstico ------------------------
function jsonOK(obj) {
  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(obj),
  };
}

exports.handler = async (event) => {
  const url = new URL(event.rawUrl);
  const q = url.searchParams.get("q") || "";
  const ping = url.searchParams.get("ping");
  const diag = url.searchParams.get("diag");
  const debug = url.searchParams.get("debug");

  if (ping) {
    return jsonOK({
      ok: true,
      version: "v2025-08-13-fix1",
      docsDir: listDocsDir(),
      dataFile: resolveDataPath("embeddings.json"),
    });
  }

  if (diag) {
    const dataFile = resolveDataPath("embeddings.json");
    const dataExists = fs.existsSync(dataFile);
    let size = 0, chunks = 0;
    if (dataExists) {
      const stat = fs.statSync(dataFile);
      size = stat.size;
      const loaded = loadEmbeddings();
      chunks = loaded.meta?.chunks || loaded.chunks?.length || 0;
    }
    const docsDir = listDocsDir();
    const docsCount = fs.existsSync(docsDir)
      ? fs.readdirSync(docsDir).filter(f => f.toLowerCase().endsWith(".pdf")).length
      : 0;
    return jsonOK({
      ok: true,
      version: "v2025-08-13-fix1",
      dataFileExists: dataExists,
      dataFileSize: size,
      embeddings: chunks,
      docsDir,
      docsCount,
    });
  }

  if (debug) {
    const emb = loadEmbeddings();
    const { tops, maxScore, expanded } = await retrieve(q || "", 8);
    return jsonOK({
      ok: true,
      version: "v2025-08-13-fix1",
      docs: fs.existsSync(listDocsDir()) ? fs.readdirSync(listDocsDir()).length : 0,
      embeddings: emb.meta?.chunks || 0,
      query: q,
      expanded,
      maxScore,
      tops,
    });
  }

  // -------- endpoint normal (chat) --------
  try {
    if (event.httpMethod === "POST") {
      const body = JSON.parse(event.body || "{}");
      const userText = String(body.message || body.q || "").trim();
      if (!userText) return jsonOK({ ok: false, error: "Mensaje vacío" });
      const out = await answerWithRAG(userText);
      return jsonOK({ ok: true, ...out });
    } else {
      // GET con ?q=...
      const userText = (q || "").trim();
      if (!userText) return jsonOK({ ok: true, message: "Usa POST {message} o GET ?q=" });
      const out = await answerWithRAG(userText);
      return jsonOK({ ok: true, ...out });
    }
  } catch (err) {
    return jsonOK({ ok: false, error: String(err) });
  }
};