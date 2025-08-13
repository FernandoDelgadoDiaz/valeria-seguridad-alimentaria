// functions/chat.js
// Valeria · RAG + PDFs · definiciones con fallback + expansión de siglas + debug
// Versión: v2025-08-12e-J

import fs from "fs";
import path from "path";

const VERSION = "v2025-08-12e-J";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const OPENAI_EMBEDDINGS_MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

const ROOT = process.env.LAMBDA_TASK_ROOT || path.resolve(".");
const DATA_DIR = path.join(ROOT, "data");
const DOCS_DIR = path.join(ROOT, "docs");
const EMBEDDINGS_PATH = path.join(DATA_DIR, "embeddings.json");

const MAX_CONTEXT_CHUNKS = 6;
const MIN_SIMILARITY_ANY = 0.65;

// ---------- util ----------
const jsonExists = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const loadJSON  = (p, fb=null) => { try { return JSON.parse(fs.readFileSync(p,"utf-8")); } catch { return fb; } };
const dot  = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const norm = (a) => Math.sqrt(a.reduce((s,v)=>s+v*v,0));
const cosineSim = (a,b) => dot(a,b)/(norm(a)*norm(b));
const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
const normalizeName = (s) => toLowerNoAccents(s.replace(/\.pdf$/i,"").replace(/[-_]+/g," ").replace(/\s+/g," ").trim());

// ---------- dominio ----------
const DOMAIN_HINTS = [
  "seguridad alimentaria","inocuidad",
  "bpm","buenas practicas de manufactura","buenas prácticas de manufactura",
  "poes","poe","haccp","appcc","pcc","ppro","sop",
  "sanitizante","sanitizacion","sanitización","limpieza","desinfeccion","desinfección","higiene",
  "alergenos","alérgenos","trazabilidad","etiquetado",
  "cadena de frio","cadena de frío","refrigeracion","refrigeración","freezer","coccion","cocción","temperatura",
  "caa","codigo alimentario argentino","código alimentario argentino",
  "mip","manejo integrado de plagas","plagas","plaga",
  "procedimiento","registro","planilla","instructivo",
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

// ---------- cache ----------
let cache = { embeddings: null, docsList: null };

// ---------- detecciones ----------
const isDefinitionPhrase = (raw) =>
  /(^(que|qué)\s+es\b)|(\bdefinici[oó]n\s+de\b)|(\bque\s+significa\b)/.test(toLowerNoAccents(raw).trim());

const hasDomainTerm = (raw) => {
  const t = toLowerNoAccents(raw);
  return DOMAIN_HINTS.some(k => t.includes(toLowerNoAccents(k)));
};

// Solo dispara PDF cuando realmente lo piden
const wantsDocument = (q) => {
  const t = toLowerNoAccents(q);
  if (isDefinitionPhrase(t)) return false;
  return /\b(pdf|archivo|descarg|abrir|mostrar|ver doc|procedimiento|formulario|registro|planilla)\b/.test(t);
};

// Expansión de siglas para mejorar recuperación
const expandAcronyms = (q) => {
  let t = " " + toLowerNoAccents(q) + " ";
  t = t.replace(/\bcaa\b/g, " código alimentario argentino ");
  t = t.replace(/\bbpm\b/g, " buenas practicas de manufactura ");
  t = t.replace(/\bhaccp\b/g, " analisis de peligros y puntos criticos de control ");
  t = t.replace(/\bpcc\b/g, " punto critico de control ");
  t = t.replace(/\bppro\b/g, " programa prerrequisito operativo ");
  return t.trim();
};

// ---------- playbook de definiciones (fallback local, sin LLM) ----------
const DEF_PLAYBOOK = {
  bpm: `**BPM (Buenas Prácticas de Manufactura)**: conjunto de reglas y procedimientos para garantizar la **inocuidad**. Pilares: **higiene personal**, infraestructura y equipos, **limpieza y sanitización (POES)**, **control de plagas (MIP)**, recepción y almacenamiento (**temperaturas**), **trazabilidad/etiquetado**, capacitación y **registros**. Es **prerrequisito de HACCP**.`,
  temperatura: `**Temperatura**: variable crítica. “Zona de peligro” **5–60 °C** (crecen patógenos). Guías típicas: **refrigerados 0–5 °C**, **congelados ≤ −18 °C**, **cocción segura ≥ 72 °C** en el centro, **recalentado ≥ 74 °C**. Enfriado rápido: de **60→10 °C en 2 h** y de **10→5 °C en 4 h**.`,
  "no conformidad": `**No conformidad**: incumplimiento de un requisito (CAA, BPM/POES, HACCP o procedimiento interno). Gestión: **registrar**, **contener**, evaluar riesgo, **corregir**, analizar **causa raíz** (5 Porqués/Ishikawa), definir **Acción Correctiva/Preventiva (CAPA)** y **verificar eficacia**.`,
  plaga: `**Plaga**: organismo (insectos, roedores, etc.) que puede **contaminar** alimentos o áreas. Control mediante **MIP**: prevención (orden, sellado), **monitoreo**, exclusión (barreras), control **físico/químico/biológico** y **registros**.`,
  sanitizante: `**Sanitizante**: agente/solución aplicada **después de limpiar** para reducir microorganismos a niveles seguros. Claves: **concentración**, **tiempo de contacto**, **temperatura** y **verificación** (p. ej., **tiras QAC**).`
};

// normaliza consulta → clave del playbook
const defKeyFromQuery = (q) => {
  const t = toLowerNoAccents(q);
  if (t.includes("no conformidad")) return "no conformidad";
  if (t.includes("bpm")) return "bpm";
  if (t.includes("temperatura")) return "temperatura";
  if (t.includes("plaga")) return "plaga";
  if (t.includes("sanitizante")) return "sanitizante";
  return null;
};

// ---------- RAG ----------
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
  const qEmb = await embedQuery(expandAcronyms(q));
  const scored = cache.embeddings.map(it => ({...it, score: cosineSim(qEmb, it.embedding)})).sort((a,b)=>b.score-a.score);
  return { chunks: scored.slice(0, MAX_CONTEXT_CHUNKS), maxScore: scored[0]?.score || 0 };
};

// ---------- PDFs ----------
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
  return docsNorm
    .map(d => ({ ...d, score: tokens.reduce((s,t)=> s + (d.norm.includes(t)?1:0), 0) }))
    .filter(d => d.score > 0)
    .sort((a,b)=> b.score - a.score)
    .slice(0,5)
    .map(d => d.title);
};

// ---------- prompting ----------
const buildSystemPrompt = () => `
Sos **Valeria**, especialista en seguridad alimentaria en Argentina (BPM, POES, CAA, HACCP y procedimientos internos de retail).
Reglas:
1) Respondé SOLO sobre seguridad alimentaria. Si está fuera de alcance, rechazá con cortesía.
2) Tono profesional y claro, con modismos AR (freezer, cadena de frío). Usá **negritas** moderadas y emojis sobrios.
3) Priorizá lo recuperado de la base vectorial. Si la evidencia es escasa pero el tema es del dominio (definiciones BPM/POES/HACCP/NO CONFORMIDAD/PLAGA/TEMPERATURA), brindá explicación estándar y pasos prácticos.
4) Si el usuario pide un documento y no existe, decí que no lo encontraste; podés sugerir títulos de /docs. No inventes enlaces.
`;

const buildUserPrompt = (q, retrieved) => {
  const ctx = retrieved.chunks.map((c)=>`● [${c.title || c.source || "fragmento"}] → ${c.chunk.trim()}`).join("\n");
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

// ---------- handler ----------
export async function handler(event) {
  try {
    // GET: ping y debug RAG
    if (event.httpMethod === "GET") {
      const qp = event.queryStringParameters || {};
      if (qp.ping === "1") { ensureLoaded(); return respond(200, { ok:true, version: VERSION, docs: cache.docsList?.length || 0, embeddings: cache.embeddings?.length || 0 }); }
      if (qp.q) { // debug
        ensureLoaded();
        const q = qp.q;
        const retrieved = await retrieve(q);
        const tops = (retrieved.chunks || []).map((c) => ({
          title: c.title || c.source, source: c.source,
          score: Number(c.score?.toFixed(4)), preview: (c.chunk || "").slice(0,160)
        }));
        return respond(200, { ok:true, version: VERSION, docs: cache.docsList?.length || 0, embeddings: cache.embeddings?.length || 0, query: q, maxScore: Number(retrieved.maxScore?.toFixed(4) || 0), tops });
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

    // A) Definiciones → SIEMPRE responder (primero playbook; si no hay, LLM; y nunca doc-intent)
    if (isDefinitionPhrase(userQuery)) {
      if (!hasDomainTerm(userQuery)) {
        return respond(200, buildText("⚠️ Puedo definir conceptos de **seguridad alimentaria** (BPM, POES, CAA, plagas, temperaturas, HACCP, etc.). Pedime términos del rubro."));
      }
      const key = defKeyFromQuery(userQuery);
      if (key && DEF_PLAYBOOK[key]) {
        return respond(200, buildText(DEF_PLAYBOOK[key]));
      }
      // fallback a LLM + RAG si no hay playbook
      const retrieved = await retrieve(userQuery);
      try {
        const answer = await openAIChat([
          { role:"system", content: buildSystemPrompt() },
          { role:"user", content: buildUserPrompt(userQuery, retrieved) }
        ]);
        if (answer) return respond(200, buildText(answer));
      } catch {}
      // último recurso
      return respond(200, buildText("Es un concepto del área de **seguridad alimentaria**. Si querés, decime el contexto (BPM/POES/HACCP/CAA) y te doy la definición específica."));
    }

    // B) Entrega directa de PDFs
    if (wantsDocument(userQuery)) {
      const matches = findDirectDocMatches(userQuery, docsList);
      if (matches.length > 0) return respond(200, buildText(buildDirectLinksResponse(matches)));
      const sug = suggestDocs(userQuery, docsList);
      const msg = sug.length
        ? `No encontré un PDF en /docs que coincida con lo que pediste. Sugerencias: ${sug.map(s=>`“${s}”`).join(", ")}.`
        : `No encontré un PDF en /docs con esas palabras. Decime el nombre exacto del archivo y lo busco.`;
      return respond(200, buildText(msg));
    }

    // C) Razonamiento con RAG
    const retrieved = await retrieve(userQuery);
    const inDomain = hasDomainTerm(userQuery);
    const lowScore = retrieved.maxScore < MIN_SIMILARITY_ANY;

    if (!inDomain && lowScore) {
      return respond(200, buildText("⚠️ Solo puedo ayudarte con **seguridad alimentaria** (BPM, POES, CAA, HACCP, sanitización, temperaturas, cadena de frío, etc.). Reformulá tu consulta dentro de ese alcance."));
    }

    const answer = await openAIChat([
      { role:"system", content: buildSystemPrompt() },
      { role:"user", content: buildUserPrompt(userQuery, retrieved) }
    ]);

    return respond(200, buildText(answer));

  } catch (err) {
    console.error(err);
    return respond(500, `Error interno: ${err.message}`);
  }
}

// ---------- helpers ----------
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