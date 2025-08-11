// functions/chat.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

let LOADED=false, CHUNKS=[], EMBS=[];
function loadOnce(){
  if(LOADED) return;
  const __filename=fileURLToPath(import.meta.url);
  const __dirname=path.dirname(__filename);
  const root=path.resolve(__dirname, "..");
  const dataDir=path.join(root, "data");
  const chunksPath=path.join(dataDir,"chunks.jsonl");
  const embsPath=path.join(dataDir,"embeddings.jsonl");
  if (fs.existsSync(chunksPath) && fs.existsSync(embsPath)) {
    CHUNKS=fs.readFileSync(chunksPath,"utf-8").trim().split("\n").map(JSON.parse);
    EMBS=fs.readFileSync(embsPath,"utf-8").trim().split("\n").map(JSON.parse);
  }
  LOADED=true;
}
function cosine(a,b){let d=0,na=0,nb=0;for(let i=0;i<Math.min(a.length,b.length);i++){const x=a[i],y=b[i];d+=x*y;na+=x*x;nb+=y*y;}return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);}

export default async (req,res)=>{
  try{
    loadOnce();
    const {query}=JSON.parse(req.body||"{}");
    if(!query || !query.trim()) return res.status(400).json({error:"missing query"});
    if(!EMBS.length||!CHUNKS.length){
      return res.status(200).json({answer:"⚠️ Aún no está listo el índice. Terminá el deploy para generar /data/chunks.jsonl y /data/embeddings.jsonl."});
    }
    const e=await client.embeddings.create({model:EMB_MODEL,input:query});
    const qv=e.data[0].embedding;
    const top=EMBS.map((r,i)=>({i,s:cosine(qv,r.embedding)})).sort((a,b)=>b.s-a.s).slice(0,5);
    const ctx=top.map(({i})=>{const r=EMBS[i];return CHUNKS.find(c=>c.doc===r.doc&&c.chunk===r.chunk)}).filter(Boolean);
    const bullets=ctx.map(c=>'• '+((c.text||'').split(/\n+/)[0].slice(0,240)||c.doc)).join("\n");
    const links=[...new Set(ctx.map(c=>c.path))].slice(0,3).map(p=>`<a href="/${p}">PDF</a>`).join(" · ");
    return res.status(200).json({answer:`📌 <b>Resumen basado en documentación interna</b>:\n${bullets}\n\n🔗 ${links}`});
  }catch(err){ return res.status(500).json({error:String(err)});}
};
