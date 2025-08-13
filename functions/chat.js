// functions/chat.js
// Valeria · RAG con multi-query + re-ranking (razonamiento interno)
// Soporta /documentos y /docs
import fs from "fs";
import path from "path";

const VERSION = "v2025-08-12k-mq-rerank";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const OPENAI_EMBEDDINGS_MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

const ROOT = process.env.LAMBDA_TASK_ROOT || path.resolve(".");
const DATA_DIR = path.join(ROOT, "data");
const EMBEDDINGS_PATH = path.join(DATA_DIR, "embeddings.json");
const DIR_CANDIDATES = ["documentos", "docs"];

const MAX_CANDIDATE_CHUNKS = 20;
const MAX_CONTEXT_CHUNKS = 8;
const MIN_SIMILARITY_ANY = 0.62;

// ---------------- utils ----------------
const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
const normalizeName = (s) => toLowerNoAccents(s.replace(/\.pdf$/i,"").replace(/[-_]+/g," ").replace(/\s+/g," ").trim());
const dot  = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const norm = (a) => Math.sqrt(a.reduce((s,v)=>s+v*v,0));
const cosineSim = (a,b) => dot(a,b)/(norm(a)*norm(b));
const loadJSON  = (p, fb=null) => { try { return JSON.parse(fs.readFileSync(p,"utf-8")); } catch { return fb; } };

// ---------------- identidad/dominio ----------------
const DOMAIN_HINTS = ["seguridad alimentaria","inocuidad","bpm","poes","haccp","caa","plaga","mip","sanit","limpieza","temperatura","cadena de frio","etiquet","rotul","pcc","ppro","registro","procedimiento","no conformidad"];
const isDefinition = (q) => /(^(que|qué)\s+es\b)|(\bdefinici[oó]n\s+de\b)/.test(toLowerNoAccents(q));
const wantsDoc = (q) => !isDefinition(q) && /\b(pdf|archivo|descarg|abrir|mostrar|ver doc|procedimiento|formulario|registro|planilla)\b/.test(toLowerNoAccents(q));

const DEF = {
  bpm: `**BPM**: prácticas para asegurar la **inocuidad**. Claves: higiene personal, POES (limpieza + **sanitización**), **MIP**, control de **temperaturas**, infraestructura/equipos, **etiquetado/trazabilidad** y **registros**. Base de **HACCP**.`,
  temperatura: `**Temperatura**: zona de peligro **5–60 °C**. Guías: **0–5 °C** (refrigerados), **≤ −18 °C** (congelados), **≥ 72 °C** (cocción). Enfriado rápido: **60→10 °C en 2 h** y **10→5 °C en 4 h**.`,
  "no conformidad": `**No conformidad**: incumplimiento de un requisito (CAA, BPM/POES, HACCP o interno). Gestión: registrar, contener, evaluar riesgo, **corrección**, causa raíz y **acciones correctivas/preventivas**; verificar eficacia.`,
  plaga: `**Plaga**: organismo que puede contaminar. Control **MIP**: prevención, monitoreo, exclusión y medidas físico/químico/biológicas con **registros**.`,
  sanitizante: `**Sanitizante**: agente aplicado **después de limpiar**. Respetar concentración, tiempo de contacto, temperatura y verificar (p. ej., **tiras QAC**).`
};
const defKey = (q) => {
  const t = toLowerNoAccents(q);
  if (t.includes("no conformidad")) return "no conformidad";
  if (t.includes("temperatura")) return "temperatura";
  if (t.includes("sanitizante")) return "sanitizante";
  if (t.includes("plaga")) return "plaga";
  if (t.includes("bpm")) return "bpm";
  return null;
};

// ---------------- carga ----------------
let CACHE = { embeddings: null, docs: null };

function listDocDirs() {
  return DIR_CANDIDATES
    .map(d => ({ name: d, abs: path.join(ROOT, d) }))
    .filter(d => fs.existsSync(d.abs));
}
function humanize(filename){
  const base = filename.replace(/\.pdf$/i,"").replace(/[-_]+/g, " ");
  return base.charAt(0).toUpperCase() + base.slice(1);
}
function listDocs() {
  const dirs = listDocDirs();
  const items = [];
  for (const d of dirs) {
    const files = fs.readdirSync(d.abs).filter(f => f.toLowerCase().endsWith(".pdf"));
    for (const f of files) {
      items.push({
        filename: f,
        title: humanize(f),
        dir: d.name,
        url: `/${d.name}/${encodeURI(f)}`
      });
    }
  }
  return items;
}
function ensureLoaded() {
  if (!CACHE.docs) CACHE.docs = listDocs();
  if (!CACHE.embeddings) CACHE.embeddings = loadJSON(EMBEDDINGS_PATH, []);
}

// ---------------- embeddings & retrieve ----------------
async function embedQuery(q) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Authorization": `Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
    body: JSON.stringify({ model: OPENAI_EMBEDDINGS_MODEL, input: expand(q) })
  });
  if (!res.ok) throw new Error(`Embeddings API: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.data[0].embedding;
}
const expand = (q) => " " + toLowerNoAccents(q) + " "
  .replace(/\bcaa\b/g," codigo alimentario argentino ")
  .replace(/\bbpm\b/g," buenas practicas de manufactura ")
  .replace(/\bhaccp\b/g," analisis de peligros y puntos criticos de control ")
  .replace(/\bpcc\b/g," punto critico de control ")
  .replace(/\bppro\b/g," programa prerrequisito operativo ")
  .trim();

function scoreAll(queryEmb, store){
  return store.map(it => ({...it, score: cosineSim(queryEmb, it.embedding)}))
              .sort((a,b)=>b.score-a.score);
}

async function retrieveSingle(q) {
  ensureLoaded();
  const store = CACHE.embeddings || [];
  if (store.length === 0) return { chunks:[], maxScore:0 };
  const qEmb = await embedQuery(q);
  const scored = scoreAll(qEmb, store);
  const slice = scored.slice(0, MAX_CANDIDATE_CHUNKS);
  return { candidates: slice, maxScore: scored[0]?.score || 0 };
}

// ---------- Multi-query + merge ----------
async function genAltQueries(q){
  const prompt = [
    { role:"system", content:"Sos especialista en seguridad alimentaria. Generá 3 variantes concisas de la consulta del usuario para recuperar información (sin pasos). Devolvé un JSON con {queries:[...]}" },
    { role:"user", content:`Consulta original: "${q}". Usá sinónimos de: rotulación/etiquetado, fraccionamiento/troceado/rebanado, sanitizante/desinfectante/QAC, CAA/código alimentario argentino, BPM/buenas prácticas de manufactura, MIP/plagas.`}
  ];
  try{
    const r = await fetch("https://api.openai.com/v1/chat/completions",{
      method:"POST",
      headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
      body: JSON.stringify({ model: OPENAI_MODEL, temperature: 0.2, messages: prompt, response_format:{ type:"json_object" } })
    });
    const j = await r.json();
    const arr = JSON.parse(j.choices?.[0]?.message?.content || "{}").queries || [];
    const uniq = Array.from(new Set([q, ...arr])).slice(0,4);
    return uniq;
  }catch{
    return [q];
  }
}

function keyOfHit(h){ return `${h.source}::${String(h.chunk).slice(0,80)}`; }

async function retrieveMulti(q){
  const alts = await genAltQueries(q);
  let pool = [];
  let best = 0;
  for (const q2 of alts){
    const { candidates, maxScore } = await retrieveSingle(q2);
    best = Math.max(best, maxScore);
    pool = pool.concat(candidates||[]);
  }
  // de-dup
  const map = new Map();
  for (const c of pool){
    const k = keyOfHit(c);
    if (!map.has(k) || map.get(k).score < c.score) map.set(k, c);
  }
  const merged = Array.from(map.values()).sort((a,b)=>b.score-a.score).slice(0, MAX_CANDIDATE_CHUNKS);
  return { candidates: merged, maxScore: best };
}

// ---------- Re-ranking con LLM ----------
async function rerankWithLLM(q, candidates){
  if ((candidates||[]).length === 0) return [];
  const items = candidates.map((c, i)=>({
    i, source: c.source, title: c.title || c.source, snippet: String(c.chunk||"").slice(0,400)
  }));
  const sys = "Devolvé SOLO JSON {keep:[indices]} con los índices (0..N-1) de los pasajes más relevantes para responder con precisión y seguridad. Máximo 8. No expliques.";
  const usr = `Consulta: "${q}"\nPasajes:\n${items.map(it=>`[${it.i}] (${it.title}) ${it.snippet}`).join("\n\n")}`;
  try{
    const r = await fetch("https://api.openai.com/v1/chat/completions",{
      method:"POST",
      headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
      body: JSON.stringify({ model: OPENAI_MODEL, temperature: 0, messages:[{role:"system",content:sys},{role:"user",content:usr}], response_format:{ type:"json_object" } })
    });
    const j = await r.json();
    const keep = JSON.parse(j.choices?.[0]?.message?.content || "{}").keep || [];
    const picked = keep.filter(Number.isInteger).map(k => candidates[k]).filter(Boolean);
    if (picked.length) return picked.slice(0, MAX_CONTEXT_CHUNKS);
  }catch{/* fall back */}
  return (candidates||[]).slice(0, MAX_CONTEXT_CHUNKS);
}

// ---------- Direct docs ----------
function findMatches(q) {
  const qn = normalizeName(q);
  const docs = (CACHE.docs || []).map(d => ({...d, norm: normalizeName(d.title)}));
  let hits = docs.filter(d => qn.includes(d.norm) || d.norm.includes(qn));
  if (hits.length === 0) {
    const tokens = qn.split("").join("").split(" ").filter(t => t.length>=4);
    hits = docs.filter(d => tokens.some(t => d.norm.includes(t)));
  }
  const seen = new Set();
  const out = [];
  for (const h of hits) {
    const key = `${h.dir}/${h.filename}`;
    if (!seen.has(key)) { seen.add(key); out.push(h); }
  }
  return out.slice(0,3);
}
const suggest = (q) => {
  const qn = normalizeName(q);
  const tokens = qn.split(" ").filter(t => t.length >= 4);
  return (CACHE.docs || [])
    .map(d => ({...d, score: tokens.reduce((s,t)=> s + (normalizeName(d.title).includes(t)?1:0), 0)}))
    .filter(d => d.score>0)
    .sort((a,b)=>b.score-a.score)
    .slice(0,5)
    .map(d => d.title);
};

// ---------- prompting ----------
const systemPrompt = () => `
Sos **Valeria**, profesional especialista en **seguridad alimentaria** en Argentina (BPM, POES, HACCP, CAA y procedimientos internos de retail).
Pensá internamente y devolvé SOLO la respuesta final, breve y accionable.
Reglas:
1) Solo temas de inocuidad; si está fuera de alcance, rechazá en una línea.
2) Priorizá lo recuperado de la base vectorial (PDFs de /documentos o /docs). Si faltan datos y es concepto básico, usá guía estándar segura.
3) Cuando corresponda, cerrá con una línea "Fuentes:" listando Título · /ruta/al/pdf.
4) Usá tono directo, argentino; resaltá **negrita** en datos críticos (temperaturas, ppm, tiempos).
`;

const userPrompt = (q, chunks) => {
  const ctx = chunks.map(c => `● [${c.title || c.source}] ${String(c.chunk||"").trim()}`).join("\n");
  return `Consulta: "${q}"\n\nContexto:\n${ctx || "(sin contexto)"}`;
};

async function llm(messages){
  const r = await fetch("https://api.openai.com/v1/chat/completions",{
    method:"POST",
    headers:{ "Authorization":`Bearer ${OPENAI_API_KEY}`, "Content-Type":"application/json" },
    body: JSON.stringify({ model: OPENAI_MODEL, temperature: 0.15, messages })
  });
  if (!r.ok) throw new Error(`Chat API: ${r.status} ${await r.text()}`);
  const j = await r.json();
  return j.choices?.[0]?.message?.content?.trim() || "";
}

// ---------- HTTP ----------
const headers = {
  "Access-Control-Allow-Origin":"*",
  "Access-Control-Allow-Headers":"Content-Type, Authorization",
  "Access-Control-Allow-Methods":"GET, POST, OPTIONS",
  "Content-Type":"application/json",
  "X-Valeria-Version": VERSION
};
const text = (t) => ({ type:"text", content: t });

export async function handler(event){
  try{
    if (event.httpMethod === "OPTIONS") return { statusCode:204, body:"", headers };

    // GET → ping / debug
    if (event.httpMethod === "GET"){
      const q = (event.queryStringParameters||{}).q;
      if ((event.queryStringParameters||{}).ping === "1"){
        ensureLoaded();
        return { statusCode:200, headers, body: JSON.stringify({ ok:true, version:VERSION, docs:(CACHE.docs||[]).length, embeddings:(CACHE.embeddings||[]).length }) };
      }
      if (q){
        ensureLoaded();
        const { candidates, maxScore } = await retrieveMulti(q);
        const tops = candidates.slice(0,6).map(c=>({ title: c.title || c.source, source: c.source, score: Number(c.score.toFixed(4)), preview: String(c.chunk||"").slice(0,160) }));
        return { statusCode:200, headers, body: JSON.stringify({ ok:true, version:VERSION, docs:(CACHE.docs||[]).length, embeddings:(CACHE.embeddings||[]).length, query:q, maxScore:Number(maxScore.toFixed(4)), tops }) };
      }
      return { statusCode:405, headers, body: JSON.stringify({ error:"Method Not Allowed" }) };
    }

    // POST → conversación
    if (event.httpMethod !== "POST") return { statusCode:405, headers, body: JSON.stringify({ error:"Method Not Allowed" }) };
    if (!OPENAI_API_KEY) return { statusCode:500, headers, body: JSON.stringify({ error:"Falta OPENAI_API_KEY" }) };

    const { message } = JSON.parse(event.body||"{}");
    const q = String(message||"").trim();
    if (!q) return { statusCode:400, headers, body: JSON.stringify({ error:"Falta 'message'." }) };

    ensureLoaded();

    // definiciones rápidas
    if (isDefinition(q)){
      const key = defKey(q);
      if (key && DEF[key]) return { statusCode:200, headers, body: JSON.stringify(text(DEF[key])) };
    }

    // PDFs directos
    if (wantsDoc(q)){
      const matches = findMatches(q);
      if (matches.length){
        const lines = matches.map(m => `✅ ${m.title}\n${m.url}`).join("\n\n");
        return { statusCode:200, headers, body: JSON.stringify(text(`Acá tenés lo pedido:\n\n${lines}`)) };
      }
      const sug = suggest(q);
      const msg = sug.length
        ? `No encontré un PDF que coincida. Probá: ${sug.map(s=>`“${s}”`).join(", ")}.`
        : `No encontré un PDF con esas palabras. Decime el nombre exacto y lo busco.`;
      return { statusCode:200, headers, body: JSON.stringify(text(msg)) };
    }

    // RAG con razonamiento (multi-query + re-rank)
    const { candidates, maxScore } = await retrieveMulti(q);
    const inDomain = DOMAIN_HINTS.some(k => toLowerNoAccents(q).includes(k));
    if (!inDomain && maxScore < MIN_SIMILARITY_ANY){
      return { statusCode:200, headers, body: JSON.stringify(text("⚠️ Solo atiendo **seguridad alimentaria** (BPM, POES, CAA, HACCP, sanitización, temperaturas, plagas, etc.). Reformulá dentro del alcance.")) };
    }
    const picked = await rerankWithLLM(q, candidates);
    const answer = await llm([
      { role:"system", content: systemPrompt() },
      { role:"user", content: userPrompt(q, picked) }
    ]);

    // agregar “Fuentes” con enlaces reales
    const byDoc = new Map();
    for (const p of picked){
      const title = (p.title || p.source);
      byDoc.set(title, (p.source||""));
    }
    const fuentes = Array.from(byDoc.entries()).slice(0,4)
      .map(([t,s]) => `• ${t} · /${s.replace(/^\/+/,"")}`)
      .join("\n");

    const final = fuentes ? `${answer}\n\nFuentes:\n${fuentes}` : answer;
    return { statusCode:200, headers, body: JSON.stringify(text(final)) };

  }catch(err){
    console.error(err);
    return { statusCode:500, headers, body: JSON.stringify({ error: `Error interno: ${err.message}`, version: VERSION }) };
  }
}