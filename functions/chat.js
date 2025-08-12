// /netlify/functions/chat.js
// Valeria · Backend RAG sobre base vectorial en /data/embeddings.json + entrega directa de PDFs en /docs
// Mantiene interfaz validado1 (frontend sin cambios). API Key via Netlify env.
// © 2025

import fs from "fs";
import path from "path";

// --------- Config ----------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const OPENAI_EMBEDDINGS_MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

// Rutas relativas al bundle de Netlify
const DATA_DIR = path.resolve("./data");
const DOCS_DIR = path.resolve("./docs");
const EMBEDDINGS_PATH = path.join(DATA_DIR, "embeddings.json"); // [{id, title, source, chunk, embedding:[...]}]
const DOCS_MANIFEST_PATH = path.join(DOCS_DIR, "manifest.json"); // opcional: [{filename, title}]
const MAX_CONTEXT_CHUNKS = 6;

// Umbrales
const MIN_SIMILARITY_GOOD = 0.82; // exige alta afinidad
const MIN_SIMILARITY_ANY = 0.72;  // si no llega, rechazamos para evitar alucinaciones

// Palabras guía para dominio (filtro rápido)
const DOMAIN_HINTS = [
  "bpm","poes","poe","inocuidad","seguridad alimentaria","sanitizante","amonio cuaternario","j-512",
  "higiene","limpieza","desinfección","alérgenos","etiquetado","cadea de frío","cadena de frio",
  "freezer","refrigeración","refrigeracion","cocción","coccion","temperatura","pcc","haccp","sop",
  "procedimiento","registro","registro de","manual","código alimentario","codigo alimentario","caa",
  "fssp","fraccionado","queso","dulce","carnes","pollo","ovina","bpm manual","5s","plagas","mip","mip manejo integrado de plagas"
];

// Alias y claves de entrega directa (coincidencias parciales y robustas)
// Nota: nombres de archivo sin guiones, criterio confirmado por el usuario.
const DIRECT_DOC_ALIASES = [
  { keys: ["sanitizante","sanitizantes","medición de sanitizante","medicion de sanitizante","g-pg-007"], file: "procedimiento sanitizantes.pdf", title:"Procedimiento de Sanitizantes" },
  { keys: ["queso","quesos","fraccionado de queso","fraccionado de quesos"], file: "procedimiento de quesos.pdf", title:"Procedimiento de Fraccionado de Quesos" },
  { keys: ["dulce","dulces","fraccionado de dulce","fraccionado de dulces"], file: "procedimiento de dulces fraccionados.pdf", title:"Procedimiento de Fraccionado de Dulces" },
  { keys: ["caa","codigo alimentario argentino","código alimentario argentino"], file: "codigo alimentario argentino.pdf", title:"Código Alimentario Argentino (CAA)" },
  { keys: ["bpm","manual de bpm"], file: "manual de bpm.pdf", title:"Manual de Buenas Prácticas de Manufactura" },
  { keys: ["5s","manual 5s","cinco s"], file: "manual 5s.pdf", title:"Manual 5S" },
  { keys: ["plagas","mip","manejo integrado de plagas"], file: "manejo integrado de plagas.pdf", title:"Manejo Integrado de Plagas (MIP)" },
  { keys: ["poes comedor","comedor"], file: "poes comedor.pdf", title:"POES del Comedor" }
];

// Cache global para Netlify
let cache = {
  embeddings: null,   // [{id,title,source,chunk,embedding:[...]}]
  docsManifest: null, // [{filename,title}]
};

// --------- Utilitarios ----------
const jsonExists = (p) => {
  try { return fs.existsSync(p); } catch { return false; }
};

const loadJSON = (p, fallback = null) => {
  try {
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf-8"));
    return fallback;
  } catch {
    return fallback;
  }
};

const dot = (a,b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(a.reduce((s, v) => s + v*v, 0));
const cosineSim = (a,b) => dot(a,b) / (norm(a)*norm(b));

const toLowerNoAccents = (s) =>
  s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();

const looksOutOfDomain = (q) => {
  const qn = toLowerNoAccents(q);
  // si no hay hints y RAG da bajo, rechazamos más abajo; acá una pista temprana
  return !DOMAIN_HINTS.some(k => qn.includes(toLowerNoAccents(k)));
};

const containsAny = (text, arr) => {
  const t = toLowerNoAccents(text);
  return arr.some(k => t.includes(toLowerNoAccents(k)));
};

const wantsDocument = (q) =>
  containsAny(q, ["pdf","procedimiento","documento","registro","poes","manual","instructivo","planilla"]);

const findDirectDocMatches = (q, docsList) => {
  const qn = toLowerNoAccents(q);
  const hits = [];

  // 1) Por alias robusto
  for (const alias of DIRECT_DOC_ALIASES) {
    if (alias.keys.some(k => qn.includes(toLowerNoAccents(k)))) {
      const candidates = (docsList || []).filter(d =>
        toLowerNoAccents(d.filename) === toLowerNoAccents(alias.file)
        || toLowerNoAccents(d.title || "") === toLowerNoAccents(alias.title)
      );
      if (candidates.length > 0) {
        hits.push({ filename: candidates[0].filename, title: candidates[0].title || alias.title });
      }
    }
  }

  // 2) Búsqueda flexible por nombre de archivo si el usuario lo sugiere
  if (docsList && hits.length === 0) {
    const direct = docsList.filter(d =>
      qn.includes(toLowerNoAccents((d.title||""))) || qn.includes(toLowerNoAccents(d.filename))
    ).slice(0,3);
    direct.forEach(d => hits.push({ filename: d.filename, title: d.title || d.filename }));
  }

  // Limpieza de duplicados
  const uniq = [];
  const seen = new Set();
  for (const h of hits) {
    const key = `${h.filename}|${h.title}`;
    if (!seen.has(key)) { seen.add(key); uniq.push(h); }
  }
  return uniq;
};

// Carga en caliente
const ensureLoaded = () => {
  if (!cache.embeddings && jsonExists(EMBEDDINGS_PATH)) {
    cache.embeddings = loadJSON(EMBEDDINGS_PATH, []);
  }
  if (!cache.docsManifest && jsonExists(DOCS_MANIFEST_PATH)) {
    cache.docsManifest = loadJSON(DOCS_MANIFEST_PATH, []);
  }
};

// Embedding de consulta
const embedQuery = async (q) => {
  const body = {
    input: q,
    model: OPENAI_EMBEDDINGS_MODEL
  };
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Embeddings API error: ${res.status} ${t}`);
  }
  const data = await res.json();
  return data.data[0].embedding;
};

// Recuperación por similitud
const retrieve = async (q) => {
  ensureLoaded();
  if (!cache.embeddings || cache.embeddings.length === 0) return { chunks: [], maxScore: 0 };

  const qEmb = await embedQuery(q);
  const scored = cache.embeddings.map(item => ({
    ...item,
    score: cosineSim(qEmb, item.embedding),
  })).sort((a,b) => b.score - a.score);

  const top = scored.slice(0, MAX_CONTEXT_CHUNKS);
  const max = scored[0]?.score || 0;
  return { chunks: top, maxScore: max };
};

// Construye prompt con grounding
const buildSystemPrompt = () => `
Eres **Valeria**, especialista en seguridad alimentaria en Argentina (BPM, POES, CAA y procedimientos internos del retail).
Reglas críticas:
1) Responde SOLO sobre seguridad alimentaria e inocuidad (BPM, POES/POE, CAA, sanitización, temperaturas, cadena de frío, alérgenos, registros, etc.). Si la consulta no es del dominio, rechaza con cortesía.
2) Usa un tono profesional, pedagógico y claro, estilo Argentina (usa "freezer", "carne vacuna", etc.). Formatea con énfasis moderado (negritas para términos clave) y emojis sobrios cuando aporten.
3) Prioriza el contenido de los documentos adjuntos (base vectorial). Cita el **título del documento** cuando ayude a dar contexto, sin enlaces externos.
4) Si el usuario pide un **procedimiento/documento/registro/manual** presente en /docs, **entrega el/los enlaces directos** (formato: "✅ Título" + link). No ofrezcas documentos que no existan.
5) No inventes. Si la evidencia es insuficiente, di qué falta y ofrece una acción concreta (p.ej., "puedo revisar el documento X si lo subís a /docs").
`;

const buildUserPrompt = (q, retrieved) => {
  const ctx = retrieved.chunks.map((c, i) => {
    const title = c.title || c.source || `fragmento_${i+1}`;
    return `● [${title}] → ${c.chunk.trim()}`;
  }).join("\n");

  return `
Consulta del usuario:
"${q}"

Contexto recuperado (usa solo lo pertinente):
${ctx || "(sin contexto recuperado)"}

Responde en español (Argentina), con precisión. Si das rangos normativos o límites críticos, sé concreto.
`;
};

const openAIChat = async (messages) => {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      temperature: 0.2,
      messages,
    }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Chat API error: ${res.status} ${t}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
};

const buildDirectLinksResponse = (matches) => {
  // Render de enlaces compactos, con check ✅ como pidió el usuario
  const lines = matches.map(m => {
    const url = `/docs/${encodeURI(m.filename)}`;
    const title = m.title || m.filename;
    return `✅ ${title}\n${url}`;
  }).join("\n\n");

  // Mensaje conciso y sin rodeos
  return `Acá tenés lo pedido:\n\n${lines}`;
};

// --------- Netlify Handler ----------
export async function handler(event) {
  try {
    if (event.httpMethod === "OPTIONS") {
      return {
        statusCode: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
      };
    }

    if (event.httpMethod !== "POST") {
      return {
        statusCode: 405,
        body: "Method Not Allowed",
      };
    }

    if (!OPENAI_API_KEY) {
      return { statusCode: 500, body: "Falta OPENAI_API_KEY en Netlify." };
    }

    const { message } = JSON.parse(event.body || "{}");
    const userQuery = (message || "").trim();
    if (!userQuery) {
      return { statusCode: 400, body: "Falta 'message'." };
    }

    ensureLoaded();

    // 1) Si pide documento → intentar entrega directa (solo si existe en /docs)
    const docsList = cache.docsManifest || safeListDocsDir();
    const directMatches = wantsDocument(userQuery) ? findDirectDocMatches(userQuery, docsList) : [];
    if (directMatches.length > 0) {
      return ok(buildText(buildDirectLinksResponse(directMatches)));
    }

    // 2) Recuperación semántica
    const retrieved = await retrieve(userQuery);

    // 3) Control de dominio + umbral
    const likelyOut = looksOutOfDomain(userQuery);
    const lowScore = retrieved.maxScore < MIN_SIMILARITY_ANY;
    if (likelyOut && lowScore) {
      // Rechazo cortés: fuera de dominio
      const msg = "Solo respondo temas de **seguridad alimentaria** (BPM, POES, CAA, sanitización, temperaturas, cadena de frío, etc.). Si querés, reformulá tu consulta dentro de ese alcance. 🙏";
      return ok(buildText(msg));
    }

    // Si el score es buenísimo, seguimos. Si es medio, respondemos pero con foco en lo recuperado.
    const sys = buildSystemPrompt();
    const usr = buildUserPrompt(userQuery, retrieved);
    const answer = await openAIChat([
      { role: "system", content: sys },
      { role: "user", content: usr },
    ]);

    // 4) Post-procesado: jamás ofrecer PDFs si no hay match real
    // (El prompt del sistema ya instruye, pero garantizamos.)
    const sanitized = answer
      .replace(/¿Querés el PDF\??/gi, "")
      .replace(/Quieres el PDF\??/gi, "");

    return ok(buildText(sanitized));

  } catch (err) {
    console.error(err);
    return {
      statusCode: 500,
      body: `Error interno: ${err.message}`,
      headers: { "Access-Control-Allow-Origin": "*" },
    };
  }
}

// --------- Helpers específicos ----------
const ok = (body) => ({
  statusCode: 200,
  body: JSON.stringify(body),
  headers: {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  },
});

// Formato que espera el frontend validado1 (mensaje plano)
const buildText = (text) => ({ type: "text", content: text });

// Lectura segura del listado de /docs si no hay manifest.json
function safeListDocsDir() {
  try {
    if (!fs.existsSync(DOCS_DIR)) return [];
    const files = fs.readdirSync(DOCS_DIR)
      .filter(f => f.toLowerCase().endsWith(".pdf"))
      .map(filename => ({ filename, title: humanize(filename) }));
    return files;
  } catch {
    return [];
  }
}

function humanize(filename) {
  const base = filename.replace(/\.pdf$/i,"");
  const spaced = base.replace(/[-_]+/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}