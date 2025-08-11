// functions/chat.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

function findDataDir(){
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  const candidates = [
    path.join(__dirname,"data"),
    path.join(__dirname,"..","data"),
    path.join(__dirname,"..","..","data"),
    path.resolve("data"),
    path.join(process.cwd(),"data"),
  ];
  for(const d of candidates){
    const need = ["chunks.jsonl","embeddings.jsonl","files_index.json"].map(f=>path.join(d,f));
    if (need.every(p=>fs.existsSync(p))) return d;
  }
  return null;
}

function readJsonlOrArray(p){
  const raw = fs.readFileSync(p, "utf-8").trim();
  if(!raw) return [];
  if(raw[0]==="[") { try{return JSON.parse(raw);}catch{return [];} }
  return raw.split("\n").filter(Boolean).map(l=>{try{return JSON.parse(l);}catch{return null;}}).filter(Boolean);
}

function normalize(s){
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

let LOADED=false, DATADIR=null, CHUNKS=[], EMBS=[], FILES=[];
function loadOnce(){
  if(LOADED) return;
  DATADIR = findDataDir();
  if(DATADIR){
    CHUNKS = readJsonlOrArray(path.join(DATADIR,"chunks.jsonl"));
    EMBS   = readJsonlOrArray(path.join(DATADIR,"embeddings.jsonl"));
    FILES  = JSON.parse(fs.readFileSync(path.join(DATADIR,"files_index.json"),"utf-8"));
  }
  LOADED=true;
}

// --- búsqueda por nombre/alias ---
const ALIASES = [
  { key:"bpm", tokens:["bpm","buenas practicas de manufactura","manual bpm","manual de bpm"] },
  { key:"pollo", tokens:["pollo","pollos"] },
  { key:"queso", tokens:["queso","quesos","fraccionamiento de quesos"] },
  { key:"poes", tokens:["poes","procedimientos operativos estandarizados de saneamiento"] },
  { key:"sanitizante", tokens:["sanitizante","j512","amonios cuaternarios","quats"] },
];

function findByNameOrAlias(query){
  const qn = normalize(query);
  const qtokens = new Set(qn.split(" ").filter(Boolean));

  // mapear alias
  const hitsAlias = ALIASES.flatMap(a => (
    a.tokens.some(t => qn.includes(normalize(t))) ? [a.key] : []
  ));
  const want = new Set([...qtokens, ...hitsAlias]);

  // scoring simple: intersección de tokens
  let best=null;
  for(const f of FILES){
    const ftokens = new Set(f.tokens || normalize(f.file||"").split(" "));
    const overlap = [...want].filter(t => ftokens.has(t)).length;
    if(overlap>0){
      const score = overlap / (ftokens.size+1);
      if(!best || score>best.score) best = { ...f, score };
    }
  }
  return best; // {file, path, norm, tokens, score}
}

export default async (req,res)=>{
  try{
    loadOnce();
    const body = req.body ? JSON.parse(req.body) : {};
    const q = (body.query||"").trim();
    if(!q) return res.status(400).json({error:"missing query"});

    if(!DATADIR || !EMBS.length || !CHUNKS.length || !FILES.length){
      return res.status(200).json({answer:
`⚠️ Índice no listo.
Chequeá:
1) /docs con PDFs.
2) Deploy generó /data/chunks.jsonl, /data/embeddings.jsonl y /data/files_index.json.
3) netlify.toml incluye included_files = ["data/*"].`
      });
    }

    // 1) INTENCIÓN DIRECTA: "dame pdf de X", "¿qué es bpm?"
    const m = q.toLowerCase().match(/(?:dame\s+)?pdf\s+de\s+(.+)$|^que\s+es\s+(.+)$/i);
    const target = m ? (m[1] || m[2] || "").trim() : "";

    const direct = target ? findByNameOrAlias(target) : findByNameOrAlias(q);
    if (direct) {
      return res.status(200).json({
        answer: `📄 <b>Documento solicitado</b>: <a href="/${direct.path}">${direct.file}</a>`
      });
    }

    // 2) FALLBACK: RAG
    const e = await client.embeddings.create({ model: EMB_MODEL, input: q });
    const qv = e.data[0].embedding;

    const top = EMBS.map((r,i)=>({i, s:cosine(qv, r.embedding)}))
                    .sort((a,b)=>b.s-a.s).slice(0,5);

    const ctx = top.map(({i})=>{
      const r = EMBS[i];
      return CHUNKS.find(c => c.doc===r.doc && c.chunk===r.chunk);
    }).filter(Boolean);

    const bullets = ctx.map(c=>'• '+((c.text||'').split(/\n+/)[0].slice(0,240)||c.doc)).join("\n");
    const links = [...new Set(ctx.map(c=>c.path))].slice(0,3).map(p=>`<a href="/${p}">PDF</a>`).join(" · ");

    return res.status(200).json({
      answer: `📌 <b>Resumen basado en documentación interna</b>:\n${bullets}\n\n🔗 ${links}`
    });
  }catch(err){
    return res.status(500).json({error:String(err)});
  }
};