// /netlify/functions/chat.js
// Valeria · Backend RAG sobre /data/embeddings.json + entrega directa de PDFs /docs
// Versión: v2025-08-12b

import fs from "fs";
import path from "path";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const OPENAI_EMBEDDINGS_MODEL = process.env.OPENAI_EMBEDDINGS_MODEL || "text-embedding-3-large";

const DATA_DIR = path.resolve("./data");
const DOCS_DIR = path.resolve("./docs");
const EMBEDDINGS_PATH = path.join(DATA_DIR, "embeddings.json");
const MAX_CONTEXT_CHUNKS = 6;

// Umbrales ajustados
const MIN_SIMILARITY_GOOD = 0.80;
const MIN_SIMILARITY_ANY = 0.65; // ↓ levemente para definiciones (BPM/POES/HACCP)

const DOMAIN_HINTS = [
  // Términos generales
  "seguridad alimentaria","inocuidad","bpm","buenas prácticas de manufactura","buenas practicas de manufactura",
  "poes","poe","procedimientos operativos estandarizados de saneamiento","haccp","appcc","pcc","ppro",
  "sanitizante","sanitizacion","sanitización","limpieza","desinfeccion","desinfección","higiene","alérgenos","alergenos",
  "cadena de frío","cadena de frio","freezer","refrigeracion","refrigeración","coccion","cocción","temperatura","trazabilidad",
  "recepcion de perecederos","recepción de perecederos","fraccionado","fraccionamiento","etiquetado",
  // Documentos y marcos normativos
  "caa","código alimentario argentino","codigo alimentario argentino","manual de bpm","manual 5s","5s",
  "mip","manejo integrado de plagas","poes comedor","procedimiento","registro","planilla","instructivo"
];

const DIRECT_DOC_ALIASES = [
  { keys: ["sanitizante","sanitizantes","medicion de sanitizante","medición de sanitizante","g-pg-007","j-512"], file: "procedimiento sanitizantes.pdf", title:"Procedimiento de Sanitizantes" },
  { keys: ["queso","quesos","fraccionado de queso","fraccionado de quesos"], file: "procedimiento de quesos.pdf", title:"Procedimiento de Fraccionado de Quesos" },
  { keys: ["dulce","dulces","fraccionado de dulce","fraccionado de dulces"], file: "procedimiento de dulces fraccionados.pdf", title:"Procedimiento de Fraccionado de Dulces" },
  { keys: ["caa","codigo alimentario argentino","código alimentario argentino"], file: "codigo alimentario argentino.pdf", title:"Código Alimentario Argentino (CAA)" },
  { keys: ["bpm","manual de bpm"], file: "manual de bpm.pdf", title:"Manual de Buenas Prácticas de Manufactura" },
  { keys: ["5s","manual 5s","cinco s"], file: "manual 5s.pdf", title:"Manual 5S" },
  { keys: ["plagas","mip","manejo integrado de plagas"], file: "manejo integrado de plagas.pdf", title:"Manejo Integrado de Plagas (MIP)" },
  { keys: ["poes comedor","comedor"], file: "poes comedor.pdf", title:"POES del Comedor" }
];

let cache = { embeddings: null, docsList: null };

const jsonExists = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const loadJSON = (p, fb=null) => { try { return JSON.parse(fs.readFileSync(p,"utf-8")); } catch { return fb; } };
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const norm = (a) => Math.sqrt(a.reduce((s,v)=>s+v*v,0));
const cosineSim = (a,b) => dot(a,b)/(norm(a)*norm(b));
const toLowerNoAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
const containsAny = (text, arr) => { const t=toLowerNoAccents(text); return arr.some(k=>t.includes(toLowerNoAccents(k))); };

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

const looksInDomain = (q) => containsAny(q, DOMAIN_HINTS);
const wantsDocument = (q) => containsAny(q, ["pdf","procedimiento","documento","registro","poes","manual","instructivo","planilla"]);

const findDirectDocMatches = (q, docsList) => {
  const qn = toLowerNoAccents(q);
  const hits = [];

  for (const alias of DIRECT_DOC_ALIASES) {
    if (alias.keys.some(k => qn.includes(toLowerNoAccents(k)))) {
      const found = docsList.find(d =>
        toLowerNoAccents(d.filename) === toLowerNoAccents(alias.file) ||
        toLowerNoAccents(d.title||"") === toLowerNoAccents(alias.title)
      );
      if (found) hits.push({ filename: found.filename, title: found.title || alias.title });
    }
  }

  if (hits.length===0) {
    const flex = docsList.filter(d =>
      qn.includes(toLowerNoAccents(d.title||"")) || qn.includes(toLowerNoAccents(d.filename))
    ).slice(0,3);
    for (const d of flex) hits.push({ filename:d.filename, title:d.title || d.filename });
  }

  // únicos
  const uniq=[], seen=new Set();
  for (const h of hits) { const k=`${h.filename}|${h.title}`; if(!seen.has(k)){seen.add(k); uniq.push(h);} }
  return uniq;
};

const buildSystemPrompt = () => `
Sos **Valeria**, especialista en seguridad alimentaria en Argentina (BPM, POES, CAA, HACCP y procedimientos internos de retail).
Reglas:
1) Respondé SOLO sobre seguridad alimentaria. Si está fuera de alcance, rechazá con cortesía.
2) Tono profesional, pedagógico y claro, con modismos AR (freezer, carne vacuna). Usá **negritas** moderadas y emojis sobrios.
3) Priorizá lo recuperado de la base vectorial. Si la evidencia es insuficiente pero la consulta es del dominio, podés dar una definición estándar y pasos prácticos.
4) Si el usuario pide un documento existente, devolvé enlaces directos con formato "✅ Título" + URL. No ofrezcas archivos inexistentes.
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

export async function handler(event) {
  try {
    if (event.httpMethod === "OPTIONS") {
      return { statusCode:204, headers: corsHeaders() };
    }
    if (event.httpMethod !== "POST") {
      return { statusCode:405, body:"Method Not Allowed", headers: corsHeaders() };
    }
    if (!OPENAI_API_KEY) return { statusCode:500, body:"Falta OPENAI_API_KEY en Netlify.", headers: corsHeaders() };

    const { message } = JSON.parse(event.body || "{}");
    const userQuery = (message || "").trim();
    if (!userQuery) return { statusCode:400, body:"Falta 'message'.", headers: corsHeaders() };

    ensureLoaded();

    // 1) Entrega directa de documentos
    const direct = wantsDocument(userQuery) ? findDirectDocMatches(userQuery, cache.docsList) : [];
    if (direct.length > 0) return ok(buildText(buildDirectLinksResponse(direct)));

    // 2) Recuperación semántica
    const retrieved = await retrieve(userQuery);

    // 3) Política de dominio y umbrales
    const inDomain = looksInDomain(userQuery);
    const lowScore = retrieved.maxScore < MIN_SIMILARITY_ANY;

    // Si es del dominio (BPM/POES/HACCP/etc.), respondemos aunque el score sea bajo (definiciones, marco conceptual).
    // Si NO es del dominio y además el score es bajo, rechazamos.
    if (!inDomain && lowScore) {
      const msg = "Solo respondo temas de **seguridad alimentaria** (BPM, POES, CAA, HACCP, sanitización, temperaturas, cadena de frío, etc.). Reformulá tu consulta dentro de ese alcance. 🙏";
      return ok(buildText(msg));
    }

    const sys = buildSystemPrompt();
    const usr = buildUserPrompt(userQuery, retrieved);
    const answer = await openAIChat([{ role:"system", content: sys }, { role:"user", content: usr }]);

    const sanitized = answer
      .replace(/¿Querés el PDF\??/gi, "")
      .replace(/Quieres el PDF\??/gi, "");

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
  const spaced = base.replace(/[-_]+/g," ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}