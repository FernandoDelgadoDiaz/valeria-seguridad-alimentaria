// functions/chat.js
// Valeria · RAG + PDFs · rutas robustas (LAMBDA_TASK_ROOT) + definiciones
// Versión: v2025-08-12e-F

import fs from "fs";
import path from "path";

const VERSION = "v2025-08-12e-F";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const OPENAI_EMBEDDINGS_MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

// Rutas: cuando Netlify empaqueta la Function, todo vive bajo LAMBDA_TASK_ROOT
const ROOT = process.env.LAMBDA_TASK_ROOT || path.resolve(".");
const DATA_DIR = path.join(ROOT, "data");
const DOCS_DIR = path.join(ROOT, "docs");
const EMBEDDINGS_PATH = path.join(DATA_DIR, "embeddings.json");

const MAX_CONTEXT_CHUNKS = 6;
const MIN_SIMILARITY_ANY = 0.65;

// --- Dominio ---
const DOMAIN_HINTS = [
  "seguridad alimentaria","inocuidad",
  "bpm","buenas practicas de manufactura","buenas prácticas de manufactura",
  "poes","poe","haccp","appcc","pcc","ppro","sop",
  "sanitizante","sanitizacion","sanitización","limpieza","desinfeccion","desinfección","higiene",
  "alergenos","alérgenos","trazabilidad","etiquetado",
  "cadena de frio","cadena de frío","freezer","refrigeracion","refrigeración","coccion","cocción","temperatura",
  "caa","codigo alimentario argentino","código alimentario argentino","manual de bpm","manual 5s","5s",
  "mip","manejo integrado de plagas","plagas","plaga",
  "poes comedor","procedimiento","registro","planilla","instructivo",
  "no conformidad","no conformidades","nc","desvio","desvío","desviacion","desviación",
  "acciones correctivas","accion correctiva","acciones preventivas","accion preventiva",
  "capa","5 porques","5 porqués","ishikawa","espina de pescado"
];

const DIRECT_DOC_ALIASES = [
  { keys: ["sanitizante","sanitizantes","j-512","medicion de sanitizante","medición de sanitizante"] },
  { keys: ["queso","quesos","fraccionado de queso","fraccionado de quesos"] },
  { keys: ["dulce","dulces","fraccionado de dulce","fraccionado de dulces"] },
  { keys: ["caa","codigo alimentario argentino","código alimentario argentino"] },
  { keys: ["bpm","manual de bpm"] },
  { keys: ["5s","manual 5s","cinco s"] },
  { keys: ["plagas","plaga","mip","manejo integrado de plagas"] },
  { keys: ["poes comedor","comedor"] }
];

let cache = { embeddings: null, docsList: null };

// --- Utils ---
const jsonExists = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const loadJSON  = (p, fb=null) => { try { return JSON.parse(fs.readFileSync(p,"utf-8")); } catch { return fb; } };
const dot  = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const norm = (a) => Math.sqrt(a.reduce((s,v)=>s+v*v,0));
const cosineSim = (a,b) => dot(a,b)/(norm(a)*norm(b));
const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
const normalizeName = (s) => toLowerNoAccents(s.replace(/\.pdf$/i,"").replace(/[-_]+/g," ").replace(/\s+/g," ").trim());

const isDefinitionQuery = (t0) => {
  const t = toLowerNoAccents(t0);
  const askDef = t.startsWith("que es ") || t.startsWith("qué es ") || t.includes("definicion de") || t.includes("definición de");
  if (!askDef) return false;
  const DEF_TERMS = [
    "bpm","poes","haccp","appcc","pcc","ppro","sanitizante","alergenos","alérgenos",
    "no conformidad","no conformidades","accion correctiva","acciones correctivas","accion preventiva","acciones preventivas",
    "capa","ishikawa","espina de pescado","5 porques","5 porqués","plaga","plagas","mip"
  ];
  return DEF_TERMS.some(k => t.includes(toLowerNoAccents(k)));
};
const looksInDomain = (q) => {
  const t = toLowerNoAccents(q);
  return DOMAIN_HINTS.some(k => t.includes(toLowerNoAccents(k))) || isDefinitionQuery(t);
};
const wantsDocument = (q) => {
  const t = toLowerNoAccents(q);
  if (isDefinitionQuery(t)) return false;
  return ["pdf","procedimiento","documento","registro","poes","manual","instructivo","planilla"].some(k => t.includes(k));
};

const ensureLoaded = () => {
  if (!cache.embeddings && jsonExists(EMBEDDINGS_PATH)) cache.embeddings = loadJSON(EMBEDDINGS_PATH, []);
  if (!cache.docsList) cache.docsList = safeListDocsDir();
};

const embedQuery = async (q) => {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method:"POST",
    headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
    body: JSON.stringify({ input:q, model:OPENAI_EMBEDDINGS_MODEL })
  });
  if (!res.ok) throw new Error(`Embeddings API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.data[0].embedding;
};

const retrieve = async (q) => {
  ensureLoaded();
  if (!cache.embeddings || cache.embeddings.length===0) return { chunks:[], maxScore:0 };
  const qEmb = await embedQuery(q);
  const scored = cache.embeddings.map(it => ({...it, score: cosineSim(qEmb, it.embedding)})).sort((a,b)=>b.score-a.score);
  return { chunks: scored.slice(0, MAX_CONTEXT_CHUNKS), maxScore: scored[0]?.score || 0 };
};

// --- PDFs ---
const findDirectDocMatches = (q, docsList) => {
  const qn = normalizeName(q);
  const docsNorm = docsList.map(d => ({ filename:d.filename, title:d.title || humanize(d.filename), norm: normalizeName(d.title || d.filename) }));
  let hits = docsNorm.filter(d => qn.includes(d.norm) || d.norm.includes(qn));
  if (hits.length === 0) {
    for (const alias of DIRECT_DOC_ALIASES) {
      if (alias.keys.some(k => qn.includes(normalizeName(k)))) {
        hits = hits.concat(docsNorm.filter(d => alias.keys.some(k => d.norm.includes(normalizeName(k)))));
      }
    }
  }
  if (hits.length === 0) {
    const tokens = qn.split(" ").filter(t => t.length >= 4);
    hits = hits.concat(docsNorm.filter(d => tokens.some(t => d.norm.includes(t))));
  }
  const seen = new Set(), uniq = [];
  for (const h of hits) { if (!seen.has(h.filename)) { seen.add(h.filename); uniq.push({ filename:h.filename, title:h.title }); } }
  return uniq.slice(0,3);
};
const suggestDocs = (q, docsList) => {
  const qn = normalizeName(q);
  const tokens = qn.split(" ").filter(t => t.length >= 4);
  const docsNorm = docsList.map(d => ({ filename:d.filename, title:d.title || humanize(d.filename), norm: normalizeName(d.title || d.filename) }));
  return docsNorm.map(d => ({ ...d, score: tokens.reduce((s,t)=> s+(d.norm.includes(t)?1:0),0)}))
    .filter(d => d.score>0).sort((a,b)=>b.score-a.score).slice(0,5).map(d => d.title);
};

// --- Prompting ---
const buildSystemPrompt = () => `
Sos **Valeria**, especialista en seguridad alimentaria en Argentina (BPM, POES, CAA, HACCP y procedimientos internos de retail).
Reglas:
1) Respondé SOLO sobre seguridad alimentaria. Si está fuera de alcance, rechazá con cortesía.
2) Tono profesional, pedagógico y claro, con modismos AR (freezer, carne vacuna). Usá **negritas** moderadas y emojis sobrios.
3) Priorizá lo recuperado de la base vectorial. Si la evidencia es escasa pero el tema es del dominio (definiciones BPM/POES/HACCP/NO CONFORMIDAD/PLAGA), brindá una explicación estándar y pasos prácticos.
4) Si el usuario pide un documento y no existe, indicá explícitamente que no lo encontraste; podés sugerir títulos similares que sí estén en /docs. No inventes enlaces.
`;
const buildUserPrompt = (q, retrieved) => {
  const ctx = retrieved.chunks.map((c,i)=>`● [${c.title || c.source || "fragmento"}] → ${c.chunk.trim()}`).join("\n");
  return `Consulta: "${q}"\n\nContexto recuperado (usá solo lo pertinente):\n${ctx || "(sin contexto recuperado)"}`;
};
const openAIChat = async (messages) => {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method:"POST",
    headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
    body: JSON.stringify({ model: OPENAI_MODEL, temperature: 0.15, messages })
  });
  if (!res.ok) throw new Error(`Chat API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() || "";
};
const buildDirectLinksResponse = (matches) => {
  const lines = matches.map(m => `✅ ${m.title || m.filename}\n${`/docs/${encodeURI(m.filename)}`}`).join("\n\n");
  return `Acá tenés lo pedido:\n\n${lines}`;
};

// --- Handler ---
export async function handler(event) {
  try {
    if (event.httpMethod === "GET") {
      const qp = event.queryStringParameters || {};
      if (qp.ping === "1") {
        ensureLoaded();
        return respond(200, { ok:true, version: VERSION, docs: cache.docsList?.length || 0, embeddings: cache.embeddings?.length || 0 });
      }
      return respond(405, "Method Not Allowed");
    }
    if (event.httpMethod === "OPTIONS") return respond(204, "");

    if (event.httpMethod !== "POST") return respond(405, "Method Not Allowed");
    if (!OPENAI_API_KEY) return respond(500, "Falta OPENAI_API_KEY en Netlify.");

    const { message } = JSON.parse(event.body || "{}");
    const userQuery = (message || "").trim();
    if (!userQuery) return respond(400, "Falta 'message'.");

    ensureLoaded();
    const docsList = cache.docsList || [];

    // A) PDFs
    if (wantsDocument(userQuery)) {
      const matches = findDirectDocMatches(userQuery, docsList);
      if (matches.length > 0) return respond(200, buildText(buildDirectLinksResponse(matches)));
      const sug = suggestDocs(userQuery, docsList);
      const msg = sug.length
        ? `No encontré un PDF en /docs que coincida con lo que pediste. Sugerencias: ${sug.map(s=>`“${s}”`).join(", ")}.`
        : `No encontré un PDF en /docs con esas palabras. Decime el nombre exacto del archivo y lo busco.`;
      return respond(200, buildText(msg));
    }

    // B) RAG / definición
    const retrieved = await retrieve(userQuery);
    const inDomain = looksInDomain(userQuery);
    const lowScore = retrieved.maxScore < MIN_SIMILARITY_ANY;
    if (!inDomain && lowScore) {
      const msg = "⚠️ Solo puedo ayudarte con **seguridad alimentaria** (BPM, POES, CAA, HACCP, sanitización, temperaturas, cadena de frío, etc.). Reformulá tu consulta dentro de ese alcance.";
      return respond(200, buildText(msg));
    }

    const answer = await openAIChat([
      { role:"system", content: buildSystemPrompt() },
      { role:"user", content: buildUserPrompt(userQuery, retrieved) }
    ]);
    const sanitized = answer.replace(/¿Querés el PDF\??/gi, "").replace(/Quieres el PDF\??/gi, "");
    return respond(200, buildText(sanitized));
  } catch (err) {
    console.error(err);
    return respond(500, `Error interno: ${err.message}`);
  }
}

// --- helpers ---
const baseHeaders = () => ({
  "Access-Control-Allow-Origin":"*",
  "Access-Control-Allow-Headers":"Content-Type, Authorization",
  "Access-Control-Allow-Methods":"GET, POST, OPTIONS",
  "X-Valeria-Version": VERSION
});
const buildText = (text) => ({ type:"text", content:text });
function respond(statusCode, body) {
  const payload = typeof body === "string" ? JSON.stringify({ type:"text", content: body }) : JSON.stringify(body);
  return { statusCode, body: payload, headers: { ...baseHeaders(), "Content-Type":"application/json" } };
}
function safeListDocsDir() {
  try {
    if (!fs.existsSync(DOCS_DIR)) return [];
    return fs.readdirSync(DOCS_DIR)
      .filter(f => f.toLowerCase().endsWith(".pdf"))
      .map(filename => ({ filename, title: humanize(filename) }));
  } catch { return []; }
}
function humanize(filename) {
  const base = filename.replace(/\.pdf$/i,"");
  const spaced = base.replace(/[-_]+/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}