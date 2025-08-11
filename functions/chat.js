// functions/chat.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

// localizar /data según empaquetado
function dataDir(){
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const cands = [
    path.join(__dirname, "data"),
    path.join(__dirname, "..", "data"),
    path.join(process.cwd(), "data"),
    path.resolve("data")
  ];
  for(const d of cands){
    const need = ["chunks.jsonl","embeddings.jsonl","files_index.json"].map(f=>path.join(d,f));
    if(need.every(p=>fs.existsSync(p))) return d;
  }
  return null;
}
function readJsonlOrArray(p){
  const raw = fs.readFileSync(p,"utf-8").trim();
  if(!raw) return [];
  if(raw[0]==="["){ try{ return JSON.parse(raw);}catch{ return [];} }
  return raw.split("\n").filter(Boolean).map(l=>{try{return JSON.parse(l);}catch{return null;}}).filter(Boolean);
}
function norm(s){
  return (s||"")
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g,"")
    .replace(/[_\-]+/g," ")
    .replace(/\s+/g," ")
    .trim();
}
function cosine(a,b){ let d=0,na=0,nb=0; const n=Math.min(a.length,b.length);
  for(let i=0;i<n;i++){ const x=a[i],y=b[i]; d+=x*y; na+=x*x; nb+=y*y; }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);
}

// carga única
let LOADED=false, DIR=null, CHUNKS=[], EMBS=[], FILES=[];
function loadOnce(){
  if(LOADED) return;
  DIR = dataDir();
  if(DIR){
    CHUNKS = readJsonlOrArray(path.join(DIR,"chunks.jsonl"));
    EMBS   = readJsonlOrArray(path.join(DIR,"embeddings.jsonl"));
    FILES  = JSON.parse(fs.readFileSync(path.join(DIR,"files_index.json"),"utf-8"));
  }
  LOADED = true;
}

// búsqueda por nombre (para “dame pdf de …”)
function findPdfByName(q){
  const m = norm(q).match(/(?:dame\s+)?(?:el\s+|la\s+)?pdf\s+de\s+(.+)$/i);
  const target = m ? m[1] : "";
  if(!target) return null;
  const qTokens = new Set(target.split(" ").filter(Boolean));
  let best=null;
  for(const f of FILES){
    const name = f.norm || norm(f.file||"");
    const tokens = new Set((f.tokens?.length ? f.tokens : name.split(" ")).filter(Boolean));
    const overlap = [...qTokens].filter(t=>tokens.has(t)).length;
    const inc = name.includes(target) ? 1 : 0;
    const score = overlap + inc;
    if(score>0 && (!best || score>best.score)) best={...f, score};
  }
  return best;
}

export async function handler(event){
  try{
    loadOnce();
    const { query="" } = event.body ? JSON.parse(event.body) : {};
    if(!query.trim()) return { statusCode:400, body:JSON.stringify({error:"missing query"}) };

    if(!DIR || !EMBS.length || !CHUNKS.length || !FILES.length){
      return { statusCode:200, body:JSON.stringify({answer:
`⚠️ Índice no listo.
Verificá:
1) /docs con PDFs.
2) Build generó /data/chunks.jsonl, /data/embeddings.jsonl, /data/files_index.json.
3) netlify.toml incluye included_files = ["data/*"].`})};
    }

    // 1) si pide PDF → nombre
    if(/pdf/i.test(query)){
      const hit = findPdfByName(query);
      if(hit){
        return { statusCode:200, body:JSON.stringify({answer:`📄 <b>Documento:</b> <a href="/${hit.path}">${hit.file}</a>`}) };
      }
    }

    // 2) RAG (razonamiento con toda la base vectorial)
    const er = await client.embeddings.create({ model: EMB_MODEL, input: query });
    const qv = er.data[0].embedding;

    const scored = EMBS.map((r,i)=>({i, s:cosine(qv, r.embedding)}))
                       .sort((a,b)=>b.s-a.s).slice(0,7);

    // umbral suave para conceptos cortos (bpm, temperatura, mip, caa, etc.)
    const MIN = 0.12; // si bajás mucho, puede dar ruido
    const topValid = scored.filter(x=>x.s>=MIN);
    const ctx = (topValid.length?topValid:scored).map(({i})=>{
      const r = EMBS[i];
      return CHUNKS.find(c=>c.doc===r.doc && c.chunk===r.chunk);
    }).filter(Boolean);

    const bullets = ctx.slice(0,5).map(c=>'• '+(c.text||'').split(/\n+/)[0].slice(0,240)).join("\n");
    const links   = [...new Set(ctx.map(c=>c.path))].slice(0,3).map(p=>`<a href="/${p}">PDF</a>`).join(" · ");

    const answer = bullets
      ? `📌 <b>Resumen basado en documentación interna</b>:\n${bullets}${links ? `\n\n🔗 ${links}` : ""}`
      : `No encontré suficiente contexto. Probá con: "manual de bpm", "codigo alimentario argentino", "poes sector pollos".`;

    return { statusCode:200, body:JSON.stringify({answer}) };
  }catch(err){
    return { statusCode:500, body:JSON.stringify({error:String(err)}) };
  }
}