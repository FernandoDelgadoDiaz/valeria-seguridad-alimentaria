import fs from "fs";
import path from "path";
import pdf from "pdf-parse";
import OpenAI from "openai";

const DOCS_DIR="docs", DATA_DIR="data";
const MODEL=process.env.EMB_MODEL||"text-embedding-3-small";
const client=new OpenAI({apiKey:process.env.OPENAI_API_KEY});

if(!fs.existsSync(DOCS_DIR)){ console.error("No existe /docs con los PDFs."); process.exit(1); }
if(!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

const chunksPath=path.join(DATA_DIR,"chunks.jsonl");
const embsPath=path.join(DATA_DIR,"embeddings.jsonl");
fs.writeFileSync(chunksPath,""); fs.writeFileSync(embsPath,"");

function chunk(t,size=1200,overlap=200){
  const out=[]; let i=0,n=t.length;
  while(i<n){ let j=Math.min(n,i+size),cut=j;
    for(const sep of [". ","\n"," "]){ const k=t.lastIndexOf(sep,j); if(k>i+200){cut=k+sep.length; break;} }
    const piece=t.slice(i,cut).trim(); if(piece) out.push(piece);
    i=Math.max(cut-overlap,cut);
  } return out;
}
async function extractText(fp){ try{ const d=await pdf(fs.readFileSync(fp)); return (d.text||"").trim(); } catch { return ""; } }

const pdfs=fs.readdirSync(DOCS_DIR).filter(f=>f.toLowerCase().endsWith(".pdf")).sort();
const all=[];
for(const name of pdfs){
  const fp=path.join(DOCS_DIR,name);
  const text=await extractText(fp);
  const parts=chunk(text||name);
  parts.forEach((t,idx)=>{ const rec={doc:name,chunk:idx,text:t,path:`docs/${name}`}; all.push(rec); fs.appendFileSync(chunksPath,JSON.stringify(rec)+"\n"); });
}
const BATCH=64;
for(let i=0;i<all.length;i+=BATCH){
  const batch=all.slice(i,i+BATCH);
  const resp=await client.embeddings.create({model:MODEL,input:batch.map(r=>r.text)});
  resp.data.forEach((e,k)=>{ const r=batch[k]; fs.appendFileSync(embsPath,JSON.stringify({doc:r.doc,chunk:r.chunk,path:r.path,embedding:e.embedding})+"\n"); });
  console.log(`Embeddings ${Math.min(i+BATCH,all.length)}/${all.length}`);
}
console.log("Listo:",chunksPath,embsPath);
