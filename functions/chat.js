// functions/chat.js
// Versión: v2025-08-12m-final
// Netlify Function - Node 18 (CommonJS)

const fs = require("fs");
const path = require("path");
const fetch = global.fetch || require("node-fetch"); // por compat

// ---------- Config ----------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const MODEL_CHAT = "gpt-4o-mini";
const MODEL_EMB = "text-embedding-3-small";
const MIN_SCORE_STRONG = 0.25; // si supera esto, respondemos con confianza
const MIN_SCORE_WEAK  = 0.18;  // debajo de esto aún respondemos, pero con red de seguridad
const MAX_CTX_CHUNKS  = 8;

// cache caliente entre invocaciones
let INDEX = null;
let INDEX_META = { docs: 0, embeddings: 0 };

function loadIndexOnce() {
  if (INDEX) return;
  const idxPath = path.join(process.cwd(), "data", "embeddings.json");
  const raw = fs.readFileSync(idxPath, "utf8");
  INDEX = JSON.parse(raw);
  INDEX_META.embeddings = INDEX.length;
  // contar docs únicos por "source"
  const sources = new Set(INDEX.map(x => x.source));
  INDEX_META.docs = sources.size;
}

// coseno
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const L = Math.min(a.length, b.length);
  for (let i = 0; i < L; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

// expansión de consulta (BPM/CAA/rotulado…)
function expandQuery(q) {
  const t = q.toLowerCase();
  const extra = [];
  if (/\bbpm\b/.test(t) || /buenas prácticas/.test(t)) {
    extra.push("buenas prácticas de manufactura", "manual BPM", "POES");
  }
  if (/\bcaa\b/.test(t) || /c[oó]digo alimentario/.test(t)) {
    extra.push("Código Alimentario Argentino", "rotulación", "capítulo V rotulación");
  }
  if (/rotulaci[oó]n|etiquet/.test(t)) {
    extra.push("información nutricional complementaria", "capítulo V rotulación");
  }
  if (/queso|fraccionamiento/.test(t)) {
    extra.push("fraccionamiento de quesos", "sobrante de quesos");
  }
  if (/desinsectaci[oó]n|plaga|mip\b/.test(t)) {
    extra.push("manejo integrado de plagas", "desinsectación química");
  }
  if (/carnicer[ií]a|pollo|vacuna|porcina|ovina/.test(t)) {
    extra.push("POES carnicería", "manejo de carne");
  }
  if (extra.length) return `${q}. Palabras clave: ${[...new Set(extra)].join(", ")}`;
  return q;
}

async function embedQuery(q) {
  const r = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ input: q, model: MODEL_EMB })
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`Embeddings API error: ${r.status} ${txt}`);
  }
  const j = await r.json();
  return j.data[0].embedding;
}

function retrieve(queryVec, q) {
  // score por coseno + pequeño boost si contiene palabras de la query
  const ql = q.toLowerCase();
  const parts = ql.split(/\W+/).filter(w => w.length > 3);
  const tops = INDEX.map(ch => {
    const s = cosine(queryVec, ch.vector);
    let boost = 0;
    for (const w of parts) {
      if (ch.text && ch.text.toLowerCase().includes(w)) boost += 0.01;
      if (ch.title && ch.title.toLowerCase().includes(w)) boost += 0.01;
    }
    return {
      title: ch.title,
      source: ch.source,
      text: ch.text,
      preview: ch.preview || (ch.text || "").slice(0, 200),
      score: s + Math.min(boost, 0.04)
    };
  }).sort((a,b) => b.score - a.score);

  // diversificar por source
  const seen = new Set();
  const diverse = [];
  for (const t of tops) {
    if (!seen.has(t.source)) {
      diverse.push(t);
      seen.add(t.source);
    }
    if (diverse.length >= MAX_CTX_CHUNKS) break;
  }
  const maxScore = tops.length ? tops[0].score : 0;
  return { tops: diverse, maxScore };
}

function buildPrompt(userQ, tops) {
  const ctx = tops.map((t, i) =>
    `### Fragmento ${i+1} — ${t.title} (${t.source})\n${t.text}`
  ).join("\n\n");

  const fuentes = tops.slice(0,5).map(t => `• ${t.title} — ${t.source}`).join("\n");

  return {
    system: `Eres Valeria, especialista en seguridad alimentaria de un supermercado argentino.
Respondes SIEMPRE en español rioplatense, con tono práctico, pasos accionables y cifras concretas.
Priorizá normas internas, BPM, CAA y POES. Si hay discrepancias, indica la más estricta`.
    trim(),
    user: `Pregunta: ${userQ}

Usa SOLO la información de los fragmentos si es suficiente. Si falta algún dato, sé prudente y acláralo.
Entregá la respuesta en bullets o pasos claros. Al final, lista “Fuentes:” con título y archivo.

${ctx}

Si la pregunta es muy general (p.ej. “qué es BPM”), brindá una definición breve útil para operación diaria (no académica) y luego pasos prácticos “en la tienda”.`
      .trim(),
    fuentes
  };
}

async function chatCompletion(messages) {
  const r = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: MODEL_CHAT,
      temperature: 0.2,
      messages
    })
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`Chat API error: ${r.status} ${txt}`);
  }
  const j = await r.json();
  return j.choices[0].message.content;
}

function jsonResponse(obj, status=200) {
  return {
    statusCode: status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(obj)
  };
}

exports.handler = async (event) => {
  try {
    if (!OPENAI_API_KEY) {
      return jsonResponse({ ok:false, error:"OPENAI_API_KEY no configurada" }, 500);
    }

    loadIndexOnce();

    // --- ping / debug por GET ---
    const params = event.queryStringParameters || {};
    if (params.ping) {
      return jsonResponse({
        ok: true,
        version: "v2025-08-12m-final",
        docs: INDEX_META.docs,
        embeddings: INDEX_META.embeddings
      });
    }

    if (params.debug && params.q) {
      const q = params.q;
      const qx = expandQuery(q);
      const qvec = await embedQuery(qx);
      const { tops, maxScore } = retrieve(qvec, qx);
      return jsonResponse({
        ok: true,
        version: "v2025-08-12m-final",
        docs: INDEX_META.docs,
        embeddings: INDEX_META.embeddings,
        query: q,
        maxScore,
        tops: tops.map(t => ({
          title: t.title, source: t.source, score: +t.score.toFixed(4),
          preview: (t.preview || "").slice(0, 240)
        }))
      });
    }

    // --- POST normal desde la UI ---
    let q = "";
    if (event.httpMethod === "POST") {
      const body = event.body ? JSON.parse(event.body) : {};
      q = body?.q || body?.question || body?.message || "";
    } else {
      q = params.q || "";
    }
    q = (q || "").trim();
    if (!q) return jsonResponse({ ok:false, error:"Falta 'q'." }, 400);

    const qx = expandQuery(q);
    const qvec = await embedQuery(qx);
    const { tops, maxScore } = retrieve(qvec, qx);

    // si no hay nada, devolvemos guía mínima en vez de “no encontré…”
    if (!tops.length) {
      return jsonResponse({
        ok: true,
        answer:
`Puedo ayudarte con seguridad alimentaria, BPM/CAA y procedimientos internos.
Probá ser más específico (ej.: “POES carnicería”, “Recepción de perecederos”, “Fraccionamiento de quesos”).`,
        sources: []
      });
    }

    const { system, user, fuentes } = buildPrompt(q, tops);

    // lógica de confianza: umbrales más bajos + sin silencios
    let answer;
    if (maxScore >= MIN_SCORE_STRONG) {
      answer = await chatCompletion([
        { role: "system", content: system },
        { role: "user",   content: user + `\n\nIMPORTANTE: contestá directo, no des disclaimers innecesarios.` }
      ]);
    } else if (maxScore >= MIN_SCORE_WEAK) {
      answer = await chatCompletion([
        { role: "system", content: system },
        { role: "user",   content: user + `\n\nSi te falta algún dato, indicá supuestos razonables de operación y pedí validación interna.` }
      ]);
    } else {
      // aún por debajo, igual damos algo útil general + mejores pistas
      const tips = tops.slice(0,3).map(t => `• ${t.title} — ${t.source}`).join("\n");
      answer =
`Esto es lo más cercano que encontré. Te dejo una guía breve y dónde mirar:

1) Explicá el objetivo y el paso a paso aplicable en sala/producción.
2) Indicá temperaturas/tiempos si corresponde.
3) Cerrá con verificación (registros/tiras/evidencias).

Fuentes sugeridas:
${tips}`;
    }

    // adjuntar fuentes (si el modelo no las agregó)
    if (!/Fuentes:/i.test(answer)) {
      answer += `\n\nFuentes:\n${fuentes}`;
    }

    return jsonResponse({
      ok: true,
      maxScore: +maxScore.toFixed(4),
      answer,
      sources: tops.slice(0,5).map(t => ({ title: t.title, source: t.source, score: +t.score.toFixed(4) }))
    });

  } catch (err) {
    return jsonResponse({ ok:false, error: String(err) }, 500);
  }
};