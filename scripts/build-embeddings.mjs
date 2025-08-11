// scripts/build-embeddings.mjs
import fs from "fs";
import fsp from "fs/promises";
import path from "path";
import fg from "fast-glob";
import OpenAI from "openai";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse"); // build estable

const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const ROOT = process.cwd();
const DOCS = path.join(ROOT, "docs");
const DATA = path.join(ROOT, "data");

function norm(s){
  return (s||"")
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g,"")
    .replace(/[_\-]+/g," ")
    .replace(/\s+/g," ")
    .trim();
}
function chunkText(t, max=1200, overlap=200){
  const out=[]; let i=0;
  while(i<t.length){
    let j=Math.min(t.length, i+max), cut=j;
    for(const sep of [". ","\n"," "]){ const k=t.lastIndexOf(sep, j); if(k>i+200){ cut=k+sep.length; break; } }
    const piece=t.slice(i,cut).trim();
    if(piece) out.push(piece);
    i=Math.max(cut-overlap, cut);
  }
  return out;
}

async function main(){
  if(!fs.existsSync(DOCS)) throw new Error("No existe /docs.");
  await fsp.mkdir(DATA, { recursive:true });

  const chunksPath = path.join(DATA, "chunks.jsonl");
  const embsPath   = path.join(DATA, "embeddings.jsonl");
  const filesIdx   = path.join(DATA, "files_index.json");
  await fsp.writeFile(chunksPath, "");
  await fsp.writeFile(embsPath,   "");

  const pdfs = await fg("**/*.pdf", { cwd:DOCS });
  console.log(`Encontrados ${pdfs.length} PDF(s) en /docs`);

  const ALL = [];
  const FILES = [];

  for(const rel of pdfs){
    const abs  = path.join(DOCS, rel);
    const file = path.basename(rel);
    const buf  = await fsp.readFile(abs);

    let text = "";
    try {
      const res = await pdfParse(buf);
      text = (res.text||"").replace(/\r/g," ").replace(/\s+/g," ").trim();
    } catch {
      text = "";
    }

    // índice por nombre (para “dame pdf de …”)
    const base = file.replace(/\.pdf$/i,"");
    const normName = norm(base);
    const tokens = Array.from(new Set(normName.split(" ").filter(Boolean)));
    FILES.push({ file, path:`docs/${rel}`, norm:normName, tokens });

    if(text.length < 30){ // escaneado o vacío → no chunkear, igual queda en FILES
      console.log(`(sin texto utilizable) ${file}`);
      continue;
    }

    const parts = chunkText(text);
    parts.forEach((t, idx)=>{
      const rec = { doc:file, chunk:idx, text:t, path:`docs/${rel}` };
      ALL.push(rec);
      fs.appendFileSync(chunksPath, JSON.stringify(rec)+"\n");
    });
  }

  console.log(`Generando embeddings para ${ALL.length} fragmentos…`);
  const BATCH = 64;
  for(let i=0;i<ALL.length;i+=BATCH){
    const batch = ALL.slice(i, i+BATCH);
    const resp = await client.embeddings.create({
      model: EMB_MODEL,
      input: batch.map(r=>r.text)
    });
    resp.data.forEach((e,k)=>{
      const r = batch[k];
      const row = { doc:r.doc, chunk:r.chunk, path:r.path, embedding:e.embedding };
      fs.appendFileSync(embsPath, JSON.stringify(row)+"\n");
    });
    console.log(`Embeddings ${Math.min(i+BATCH,ALL.length)}/${ALL.length}`);
  }

  await fsp.writeFile(filesIdx, JSON.stringify(FILES, null, 2));
  console.log("Listo:", chunksPath, embsPath, filesIdx);
}

main().catch(err=>{ console.error(err); process.exit(1); });