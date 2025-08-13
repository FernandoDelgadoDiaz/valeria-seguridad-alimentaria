// functions/chat.js
// ─────────────────────────────────────────────────────────────
// Valeria · Seguridad Alimentaria (AR) · RAG + Ingeniería Conversacional
// Endpoints:
// - GET  ?ping=1                               → estado (docs/embeddings/version)
// - GET  ?debug=1&q=...                        → top-k del RAG (scores + preview)
// - POST { message }                           → respuesta usando RAG + fuentes
//
// Notas:
// - Usa data/embeddings.json (generado por scripts/build-embeddings.mjs).
// - Espera objetos con: { text|content, source, title, embedding:number[] }.
// - Node 18. fetch nativo; fallback a node-fetch si hiciera falta.

const fs = require("fs");
const path = require("path");
let fetchFn = global.fetch;
if (typeof fetchFn !== "function") fetchFn = require("node-fetch");

// ───────── Config ─────────
const VERSION = "v2025-08-13-conv2.0";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const CHAT_MODEL = process.env.CHAT_MODEL || "gpt-4o-mini";
const EMB_MODEL  = process.env.EMB_MODEL  || "text-embedding-3-large";

const RAG_TOP_K = Number(process.env.RAG_TOP_K || 8);
const RAG_MIN_SCORE_STRONG = Number(process.env.RAG_MIN_SCORE_STRONG || 0.25);
const RAG_MIN_SCORE_WEAK   = Number(process.env.RAG_MIN_SCORE_WEAK   || 0.18);
const MAX_SNIPPET_LEN = Number(process.env.MAX_SNIPPET_LEN || 480);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Content-Type": "application/json; charset=utf-8",
};

// ───────── Identidad / Ingeniería Conversacional ─────────
const SYSTEM_PROMPT = `
Eres **Valeria**, especialista en **Seguridad Alimentaria** para retail de alimentos en Argentina.
Objetivo: dar respuestas **prácticas, breves y accionables** basadas en los documentos internos (BPM/POES/MIP, manuales de la empresa) y en el **Código Alimentario Argentino (CAA)**. Siempre en español rioplatense.

ALCANCE
- Temas: BPM, POES, MIP, CAA (cap. II, V, rotulación, inocuidad), procedimientos internos y registros.
- Si la pregunta está fuera de alcance: dilo en 1 línea y redirígela a temas válidos.

ESTILO
- Formato: pasos numerados o viñetas, con **números concretos** (ppm, °C, tiempos), riesgos y verificación.
- Tono: claro, directo, operativo. Nada de relleno.
- Cierra con **Fuentes** listando 1–3 PDFs/títulos relevantes de la base.

INGENIERÍA CONVERSACIONAL
- Si faltan datos críticos para una instrucción segura, **pregunta primero** (producto/sector, etapa del proceso, temperatura/tiempo, concentración, equipo/superficie).
- Si la consulta es ambigua, ofrece **2–3 opciones** de interpretación.
- Si el top de recuperación es débil, da una **guía mínima segura** + pide precisión + muestra fuentes más cercanas.

REGLAS
- Prioriza documentos **internos** sobre externos cuando haya conflicto.
- No inventes números. Si no hay dato, dilo y sugiere cómo medirlo o dónde buscarlo.
- Nunca respondas fuera de seguridad alimentaria.

Responde siempre en este formato:
1) **Respuesta operativa** (pasos con cifras)
2) **Notas/Riesgos**
3) **Fuentes:** Título (ruta/al/pdf)
`.trim();

// Sinónimos / expansiones que ayudan al embedding
const EXPANSIONS = {
  "bpm": ["buenas prácticas de manufactura", "manual bpm", "poes"],
  "código alimentario argentino": ["caa", "capítulo v rotulación", "información nutricional", "capítulo ii establecimientos"],
  "poes": ["limpieza y desinfección", "sanitización", "saneamiento"],
  "mip": ["manejo integrado de plagas", "control de plagas", "desinsectación"],
  "quesos": ["fraccionamiento de quesos", "sobrante de quesos", "rotulado de quesos"],
  "carnicería": ["poes carnicería", "carne vacuna", "pollo", "molida", "sierras"],
  "sanitizante": ["amonios cuaternarios", "qac", "desinfección"],
  "rotulación": ["etiquetado", "información nutricional complementaria", "capítulo v"]
};

// ───────── Carga del índice (una vez por frío) ─────────
let VDB = null; // [{text, source, title, embedding}]
let META = { docs: 0, embeddings: 0 };

function loadVDB() {
  if (VDB) return VDB;
  const p = path.join(process.cwd(), "data", "embeddings.json");
  const raw = fs.readFileSync(p, "utf8");
  const arr = JSON.parse(raw);
  VDB = arr
    .filter(x => Array.isArray(x.embedding) && (x.text || x.content))
    .map(x => ({
      text: (x.text || x.content || "").toString(),
      source: (x.source || x.doc || x.file || "").toString(),
      title: (x.title  || x.source || x.doc || "fragmento").toString(),
      embedding: x.embedding
    }));
  META.embeddings = VDB.length;
  META.docs = new Set(VDB.map(x => x.source)).size;
  return VDB;
}

const uniqueDocs = () => Array.from(new Set(loadVDB().map(x => x.source)));

// ───────── Utilidades RAG ─────────
const cos = (a,b) => {
  let dot=0, na=0, nb=0, L=Math.min(a.length,b.length);
  for (let i=0;i<L;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) + 1e-12);
};

const embedMany = async (inputs) => {
  const res = await fetchFn("https://api.openai.com/v1/embeddings",{
    method:"POST",
    headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
    body: JSON.stringify({ model: EMB_MODEL, input: inputs })
  });
  if (!res.ok) throw new Error(`Embeddings ${res.status}: ${await res.text()}`);
  const j = await res.json();
  return j.data.map(d => d.embedding);
};

const expandQuery = (q) => {
  const t = (q||"").toLowerCase();
  const set = new Set([t]);
  for (const [k,vals] of Object.entries(EXPANSIONS)) {
    if (t.includes(k)) vals.forEach(v => set.add(v));
    vals.forEach(v => { if (t.includes(v)) vals.forEach(vv => set.add(vv)); });
  }
  // plus contexto mínimo si es muy corta
  if (t.length < 18) set.add(t + " retail alimentos argentina bpm caa poes mip temperaturas rotulación");
  return Array.from(set).slice(0,6);
};

const rankByQueries = async (queries) => {
  const db = loadVDB();
  const qVecs = await embedMany(queries);
  const scored = db.map(ch => {
    let s = -1;
    for (const v of qVecs) s = Math.max(s, cos(v, ch.embedding));
    // pequeño boost por coincidencia en título/fuente
    const tf = (ch.title + " " + ch.source).toLowerCase();
    if (queries.some(q => tf.includes(q))) s += 0.03;
    return { ...ch, score: s };
  }).sort((a,b) => b.score - a.score);

  // diversificar por fuente
  const seen = new Set(), out = [];
  for (const it of scored) {
    if (!seen.has(it.source)) { out.push(it); seen.add(it.source); }
    if (out.length >= RAG_TOP_K) break;
  }
  const maxScore = scored[0]?.score || 0;
  return { candidates: out, maxScore };
};

// ───────── Heurísticas de conversación ─────────
const CRITICAL = {
  producto: /(queso|carne|pollo|cerdo|pescad|fiambre|pan|ensalada|salsa|lácte|leche|yogur|crema|dulce|huevo)/i,
  sector: /(carnicer[ií]a|fiambrer[ií]a|panader[ií]a|pasteler[ií]a|l[aá]cteos|c[áa]mara|dep[oó]sito|g[óo]ndola|cocina|comedor)/i,
  etapa: /(recepci[oó]n|almacenamiento|exhibici[oó]n|fraccionamiento|preparaci[oó]n|cocci[oó]n|enfriado|recalentado|limpieza|sanitizaci[oó]n|desinfecci[oó]n)/i,
  parametro: /(°c|grados|ppm|partes por mill[oó]n|minutos|horas|tiempo|temperatura|concentraci[oó]n)/i,
  equipo: /(tabla|cuchillo|sierras?|mostrador|balanza|c[aá]mara|heladera|freezer|horno|m[áa]quina)/i
};

function detectClarificationNeed(q, maxScore, snippets) {
  const t = (q||"").toLowerCase();
  const missing =
    (!CRITICAL.producto.test(t)) +
    (!CRITICAL.sector.test(t)) +
    (!CRITICAL.etapa.test(t)) +
    (!CRITICAL.parametro.test(t));
  // pedir aclaración si la pregunta es genérica y la señal no es fuerte
  const weak = maxScore < 0.28 || (snippets||[]).length < 2;
  return missing >= 2 && weak;
}

function buildClarifying(q){
  const qs = [];
  if (!CRITICAL.sector.test(q)) qs.push("¿En qué **sector**? (carnicería, fiambrería, panadería, lácteos, depósito)");
  if (!CRITICAL.producto.test(q)) qs.push("¿Sobre qué **producto/superficie**?");
  if (!CRITICAL.etapa.test(q)) qs.push("¿En qué **etapa** (recepción, almacenamiento, fraccionamiento, limpieza/sanitización, etc.)?");
  if (!CRITICAL.parametro.test(q)) qs.push("¿Tenés algún **límite** que cumplir (°C, ppm, tiempos)?");
  return qs.slice(0,3);
}

// Intento de “quiero el PDF” (abrir/descargar/ver doc…)
const wantsDocument = (q) => /\b(pdf|archivo|descarg|abrir|mostrar|ver doc|procedimiento|formulario|registro|planilla)\b/i.test(q||"");
const normalizeName = (s) => (s||"").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"").replace(/\.pdf$/i,"").replace(/[-_]+/g," ").replace(/\s+/g," ").trim();

function matchDocsByName(q){
  const qn = normalizeName(q);
  const docs = uniqueDocs().map(src => ({
    src,
    title: path.basename(src).replace(/\.pdf$/i,"").replace(/[-_]+/g," "),
    norm: normalizeName(path.basename(src))
  }));
  let hits = docs.filter(d => qn.includes(d.norm) || d.norm.includes(qn));
  if (!hits.length) {
    const tokens = qn.split(" ").filter(t => t.length>=4);
    hits = docs.filter(d => tokens.some(t => d.norm.includes(t)));
  }
  return hits.slice(0,3).map(h => ({
    title: h.title.charAt(0).toUpperCase()+h.title.slice(1),
    url: `/${h.src}`
  }));
}

// ───────── LLM calls ─────────
async function chat(system, user, temperature=0.2){
  const res = await fetchFn("https://api.openai.com/v1/chat/completions",{
    method:"POST",
    headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
    body: JSON.stringify({ model: CHAT_MODEL, temperature, messages:[ {role:"system", content: system}, {role:"user", content: user} ] })
  });
  if (!res.ok) throw new Error(`Chat ${res.status}: ${await res.text()}`);
  const j = await res.json();
  return j.choices?.[0]?.message?.content?.trim() || "";
}

// ───────── HTTP helpers ─────────
const respond = (code, obj) => ({ statusCode: code, headers: CORS, body: JSON.stringify(obj) });

// ───────── Handler ─────────
exports.handler = async (event) => {
  try {
    if (event.httpMethod === "OPTIONS") return respond(204, "");

    if (!OPENAI_API_KEY) return respond(500, { ok:false, error:"Falta OPENAI_API_KEY" });

    loadVDB();

    const qs = event.queryStringParameters || {};

    // Ping
    if (qs.ping) return respond(200, { ok:true, version: VERSION, docs: META.docs, embeddings: META.embeddings });

    // Debug
    if (qs.debug) {
      const rawQ = String(qs.q||"").trim();
      if (!rawQ) return respond(400, { ok:false, error:"Falta ?q=" });
      const queries = expandQuery(rawQ);
      const { candidates, maxScore } = await rankByQueries(queries);
      return respond(200, {
        ok:true, version: VERSION,
        docs: META.docs, embeddings: META.embeddings,
        query: rawQ, expanded: queries,
        maxScore: Number(maxScore.toFixed(4)),
        tops: candidates.slice(0,6).map(c=>({
          title: c.title, source: c.source, score: Number(c.score.toFixed(4)),
          preview: (c.text||"").replace(/\s+/g," ").slice(0,320)
        }))
      });
    }

    // Entrada del chat
    let userQ = "";
    if (event.httpMethod === "POST") {
      const body = JSON.parse(event.body||"{}");
      userQ = String(body.message || body.q || body.question || "").trim();
    } else {
      userQ = String(qs.q||"").trim();
    }
    if (!userQ) return respond(400, { ok:false, error:"Falta 'message'." });

    // Si piden directamente un PDF por nombre → entregar enlaces
    if (wantsDocument(userQ)) {
      const hits = matchDocsByName(userQ);
      if (hits.length) {
        return respond(200, { ok:true, answer: `Acá tenés lo pedido:\n\n${hits.map(h => `✅ ${h.title}\n${h.url}`).join("\n\n")}`, links: hits });
      }
      // sugerencia blanda si no hubo match
      return respond(200, { ok:true, answer: "No encontré un PDF con ese nombre. Probá con el título tal como figura en /documentos o /docs." });
    }

    // RAG: rank con queries expandidas
    const queries = expandQuery(userQ);
    const { candidates, maxScore } = await rankByQueries(queries);

    // Snippets útiles (o fallback si baja señal)
    const useful = candidates.filter(x => x.score >= RAG_MIN_SCORE_WEAK);
    const snippets = (useful.length ? useful : candidates).slice(0, RAG_TOP_K).map(x => ({
      title: x.title,
      source: x.source,
      text: (x.text||"").slice(0, MAX_SNIPPET_LEN)
    }));

    // Si la consulta es demasiado ambigua → primero preguntar
    if (detectClarificationNeed(userQ, maxScore, snippets)) {
      return respond(200, {
        ok: true,
        needs_clarification: true,
        answer: "Para darte una instrucción segura necesito un par de datos:",
        questions: buildClarifying(userQ)
      });
    }

    // Construcción de contexto y prompt
    const ctx = snippets.map((s,i)=> `[#${i+1}] ${s.title} · /${s.source}\n${s.text}`).join("\n\n");
    const userPrompt = `
Consulta: ${userQ}

Usá SOLO estos fragmentos. Si falta un dato, sé prudente y acláralo. Cita con [#n] al final de cada afirmación importante.

${ctx}
`.trim();

    // Seleccionar modo de respuesta según señal
    const temperature = maxScore >= RAG_MIN_SCORE_STRONG ? 0.15 : 0.2;
    let answer = await chat(SYSTEM_PROMPT, userPrompt, temperature);

    // Agregar Fuentes si el modelo no las listó
    if (!/Fuentes:/i.test(answer)) {
      const fuentes = Array.from(new Map(snippets.map(s => [s.source, `• ${s.title} · /${s.source}`])).values()).slice(0,4).join("\n");
      if (fuentes) answer += `\n\nFuentes:\n${fuentes}`;
    }

    return respond(200, {
      ok:true,
      version: VERSION,
      maxScore: Number(maxScore.toFixed(4)),
      answer,
      sources: Array.from(new Set(snippets.map(s => s.source))).slice(0,5)
    });

  } catch (err) {
    return respond(500, { ok:false, error: String(err.message || err), version: VERSION });
  }
};