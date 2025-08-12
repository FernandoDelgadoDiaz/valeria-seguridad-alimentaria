// /netlify/functions/chat.js
// Valeria · RAG + entrega directa de PDFs con coincidencia flexible (guiones/espacios)
// Versión: v2025-08-12c

import fs from "fs";
import path from "path";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const OPENAI_EMBEDDINGS_MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

const DATA_DIR = path.resolve("./data");
const DOCS_DIR = path.resolve("./docs");
const EMBEDDINGS_PATH = path.join(DATA_DIR, "embeddings.json");
const MAX_CONTEXT_CHUNKS = 6;

// Umbrales
const MIN_SIMILARITY_ANY = 0.65;

// Hints de dominio
const DOMAIN_HINTS = [
  "seguridad alimentaria","inocuidad","bpm","buenas practicas de manufactura","buenas prácticas de manufactura",
  "poes","poe","haccp","appcc","pcc","ppro","sanitizante","limpieza","desinfeccion","desinfección","higiene",
  "alergenos","alérgenos","cadena de frio","cadena de frío","freezer","refrigeracion","refrigeración",
  "coccion","cocción","temperatura","trazabilidad","recepcion de perecederos","recepción de perecederos",
  "fraccionado","fraccionamiento","etiquetado","caa","codigo alimentario argentino","código alimentario argentino",
  "manual de bpm","manual 5s","5s","mip","manejo integrado de plagas","poes comedor","procedimiento","registro","planilla","instructivo"
];

// Alias de intención (para mejorar recall)
const DIRECT_DOC_ALIASES = [
  { keys: ["sanitizante","sanitizantes","j-512","medicion de sanitizante","medición de sanitizante"] },
  { keys: ["queso","quesos","fraccionado de queso","fraccionado de quesos"] },
  { keys: ["dulce","dulces","fraccionado de dulce","fraccionado de dulces"] },
  { keys: ["caa","codigo alimentario argentino","código alimentario argentino"] },
  { keys: ["bpm","manual de bpm"] },
  { keys: ["5s","manual 5s","cinco s"] },
  { keys: ["plagas","mip","manejo integrado de plagas"] },
  { keys: ["poes comedor","comedor"] }
];

let cache = { embeddings: null, docsList: null };

// ---------------- Utils ----------------
const jsonExists = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const loadJSON = (p, fb=null) => { try { return JSON.parse(fs.readFileSync(p,"utf-8")); } catch { return fb; } };
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const norm = (a) => Math.sqrt(a.reduce((s,v)=>s+v*v,0));
const cosineSim = (a,b) => dot(a,b)/(norm(a)*norm(b));
const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
const looksInDomain = (q) => DOMAIN_HINTS.some(k => toLowerNoAccents(q).includes(toLowerNoAccents(k)));
const wantsDocument = (q) => ["pdf","procedimiento","documento","registro","poes","manual","instructivo","planilla"]
  .some(k => toLowerNoAccents(q).includes(k));

// normaliza nombres de archivo para comparar (quita .pdf, acentos, guiones/underscores→espacio, compacta espacios)
const normalizeName = (s) => toLowerNoAccents(
  s.replace(/\.pdf$/i,"")
   .replace(/[-_]+/g," ")
   .replace(/\s+/g," ")
   .trim()
);

// carga
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

// ----------- Matching flexible de PDFs -----------
const findDirectDocMatches = (q, docsList) => {
  const qn = normalizeName(q);
  const docsNorm = docsList.map(d => ({
    filename: d.filename,
    title: d.title || humanize(d.filename),
    norm: normalizeName(d.title || d.filename)
  }));

  // 1) si la consulta incluye el nombre (total o parcial)
  let hits = docsNorm.filter(d => qn.includes(d.norm) || d.norm.includes(qn));

  // 2) si coincide con alias temático (ej. "sanitizante") → buscar por palabra clave en el filename/título
  if (hits.length === 0) {
    for (const alias of DIRECT_DOC_ALIASES) {
      if (alias.keys.some(k => qn.includes(normalizeName(k)))) {
        const pool = docsNorm.filter(d =>
          alias.keys.some(k => d.norm.includes(normalizeName(k)))
        );
        hits = hits.concat(pool);
      }
    }
  }

  // 3) si aún no hay, usar tokens de la pregunta (>3 letras) para fuzz
  if (hits.length === 0) {
    const tokens = qn.split(" ").filter(t => t.length >= 4);
    const pool = docsNorm.filter(d => tokens.some(t => d.norm.includes(t)));
    hits = hits.concat(pool);
  }

  // únicos y top 3
  const seen = new Set(); const uniq = [];
  for (const h of hits) { const key = h.filename; if (!seen.has(key)) { seen.add(key); uniq.push({ filename:h.filename, title:h.title }); } }
  return uniq.slice(0,3);
};

const suggestDocs = (q, docsList) => {
  const qn = normalizeName(q);
  const tokens = qn.split(" ").filter(t => t.length >= 4);
  const docsNorm = docsList.map(d => ({ filename:d.filename, title:d.title || humanize(d.filename), norm: normalizeName(d.title || d.filename) }));
  const pool = docsNorm
    .map(d => ({ ...d, score: tokens.reduce((s,t)=> s + (d.norm.includes(t)?1:0), 0) }))
    .filter(d => d.score > 0)
    .sort((a,b)=> b.score - a.score)
    .slice(0,5)
    .map(d => d.title);
  return pool;
};

// ----------- Prompting -----------
const buildSystemPrompt = () => `
Sos **Valeria**, especialista en seguridad alimentaria en Argentina (BPM, POES, CAA, HACCP y procedimientos internos de retail).
Reglas:
1) Respondé SOLO sobre seguridad alimentaria. Si está fuera de alcance, rechazá con cortesía.
2) Tono profesional, pedagógico y claro, con modismos AR (freezer, carne vacuna). Usá **negritas** moderadas y emojis sobrios.
3) Priorizá lo recuperado de la base vectorial. Si la evidencia es escasa pero el tema es del dominio (p.ej. definiciones BPM/POES/HACCP), podés dar explicación estándar con pasos prácticos.
4) Si el usuario pide un documento y no existe, decí explícitamente que no lo encontraste y no ofrezcas nada alternativo salvo sugerencias de búsqueda dentro de /docs.
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

// ------------- Handler -------------
export async function handler(event) {
  try {
    if (event.httpMethod === "OPTIONS") return { statusCode:204, headers: corsHeaders() };
    if (event.httpMethod !== "POST") return { statusCode:405, body:"Method Not Allowed", headers: corsHeaders() };
    if (!OPENAI_API_KEY) return { statusCode:500, body:"Falta OPENAI_API_KEY en Netlify.", headers: corsHeaders() };

    const { message } = JSON.parse(event.body || "{}");
    const userQuery = (message || "").trim();
    if (!userQuery) return { statusCode:400, body:"Falta 'message'.", headers: corsHeaders() };

    ensureLoaded();
    const docsList = cache.docsList || [];

    // A) Intento de entrega directa (si hay intención documental)
    const docIntent = wantsDocument(userQuery);
    if (docIntent) {
      const matches = findDirectDocMatches(userQuery, docsList);
      if (matches.length > 0) {
        return ok(buildText(buildDirectLinksResponse(matches)));
      } else {
        const sug = suggestDocs(userQuery, docsList);
        const msg = sug.length
          ? `No encontré un PDF en /docs que coincida con lo que pediste. Sugerencias de búsqueda: ${sug.map(s=>`“${s}”`).join(", ")}.`
          : `No encontré un PDF en /docs con esas palabras. Decime el nombre exacto del archivo y lo busco.`;
        return ok(buildText(msg));
      }
    }

    // B) Razonamiento con RAG
    const retrieved = await retrieve(userQuery);
    const inDomain = looksInDomain(userQuery);
    const lowScore = retrieved.maxScore < MIN_SIMILARITY_ANY;

    if (!inDomain && lowScore) {
      const msg = "⚠️ Solo puedo ayudarte con **seguridad alimentaria** (BPM, POES, CAA, HACCP, sanitización, temperaturas, cadena de frío, etc.). Reformulá tu consulta dentro de ese alcance.";
      return ok(buildText(msg));
    }

    const answer = await openAIChat([
      { role:"system", content: buildSystemPrompt() },
      { role:"user", content: buildUserPrompt(userQuery, retrieved) }
    ]);

    const sanitized = answer.replace(/¿Querés el PDF\??/gi, "").replace(/Quieres el PDF\??/gi, "");
    return ok(buildText(sanitized));

  } catch (err) {
    console.error(err);
    return { statusCode:500, body:`Error interno: ${err.message}`, headers: corsHeaders() };
  }
}

const ok = (body) => ({ statusCode:200, body: JSON.stringify(body), headers: { ...corsHeaders(), "Content-Type":"application/json" } });
const buildText = (text) => ({ type:"text", content:text });
const corsHeaders = () => ({ "Access-Control-Allow-Origin":"*", "Access-Control-Allow-Headers":"Content-Type, Authorization", "Access-Control-Allow-Methods":"POST, OPTIONS" });

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