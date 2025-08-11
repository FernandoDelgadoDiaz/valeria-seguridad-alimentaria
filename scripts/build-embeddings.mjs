import fs from "fs";
import path from "path";
import pdf from "pdf-parse/lib/pdf-parse.js"; // fix ruta en Netlify
import OpenAI from "openai";

const DOCS_DIR = "docs";
const DATA_DIR = "data";
const MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
const chunksPath = path.join(DATA_DIR, "chunks.jsonl");
const embsPath   = path.join(DATA_DIR, "embeddings.jsonl");
const filesIdx   = path.join(DATA_DIR, "files_index.json");

fs.writeFileSync(chunksPath, "");
fs.writeFileSync(embsPath,   "");

const pdfs = fs.existsSync(DOCS_DIR)
  ? fs.readdirSync(DOCS_DIR).filter(f => f.toLowerCase().endsWith(".pdf")).sort()
  : [];

console.log(`Encontrados ${pdfs.length} PDF(s) en /${DOCS_DIR}`);

function normalize(s){
  return (s||"")
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g,"")
    .replace(/[_\-]+/g," ")
    .replace(/\s+/g," ")
    .trim();
}

function chunk(text, size=1200, overlap=200){
  const out=[]; let i=0, n=text.length;
  while(i<n){
    let j=Math.min(n,i+size), cut=j;
    for(const sep of [". ","\n"," "]){ const k=text.lastIndexOf(sep,j); if(k>i+200){ cut=k+sep.length; break; } }
    const piece=text.slice(i,cut).trim();
    if(piece) out.push(piece);
    i=Math.max(cut-overlap,cut);
  }
  return out;
}

async function extractText(fp){
  try{
    const data = await pdf(fs.readFileSync(fp));
    return (data.text||"").replace(/\s+/g," ").trim();
  }catch{
    return "";
  }
}

const ALL = [];
const filesIndex = [];

for(const name of pdfs){
  const fp = path.join(DOCS_DIR, name);
  const text = await extractText(fp);
  const parts = chunk(text || name); // si no hay texto (escaneado), uso el nombre
  parts.forEach((t,idx)=>{
    const rec = { doc:name, chunk:idx, text:t, path:`docs/${name}` };
    ALL.push(rec);
    fs.appendFileSync(chunksPath, JSON.stringify(rec)+"\n");
  });

  // índice por nombre (para “Dame PDF de…” y alias)
  const base = path.basename(name, ".pdf");
  const norm = normalize(base);
  const tokens = Array.from(new Set(norm.split(" "))).filter(Boolean);
  filesIndex.push({ file:name, path:`docs/${name}`, norm, tokens });
}

console.log(`Generando embeddings para ${ALL.length} fragmentos...`);
const BATCH = 64;
for(let i=0;i<ALL.length;i+=BATCH){
  const batch = ALL.slice(i,i+BATCH);
  const resp = await client.embeddings.create({
    model: MODEL,
    input: batch.map(r => r.text)
  });
  resp.data.forEach((e,k)=>{
    const r = batch[k];
    const row = { doc:r.doc, chunk:r.chunk, path:r.path, embedding:e.embedding };
    fs.appendFileSync(embsPath, JSON.stringify(row)+"\n");
  });
  console.log(`Embeddings ${Math.min(i+BATCH,ALL.length)}/${ALL.length}`);
}

fs.writeFileSync(filesIdx, JSON.stringify(filesIndex, null, 2));
console.log("Listo:", chunksPath, embsPath, filesIdx);