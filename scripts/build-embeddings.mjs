// scripts/build-embeddings.mjs
// Extrae texto de /docs con pdfjs-dist LEGACY (Node 18/20), filtra vacíos y genera data/embeddings.json

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import OpenAI from "openai";
import { createRequire } from "node:module";

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, "docs");
const OUT_DIR  = path.join(ROOT, "data");
const OUT_FILE = path.join(OUT_DIR, "embeddings.json");

const EMB_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// --- pdfjs-dist LEGACY ---
let pdfjsLib;
try { pdfjsLib = await import("pdfjs-dist/legacy/build/pdf.mjs"); }
catch { const require = createRequire(import.meta.url); pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js"); }
// NO seteamos workerSrc en Node.

const clean = (s="") => s.replace(/\u0000/g," ").replace(/\s+/g," ").trim();

async function extractPdfText(absPath) {
  const data = fs.readFileSync(absPath);
  const getDocument = pdfjsLib.getDocument || pdfjsLib.default?.getDocument;
  if (!getDocument) throw new Error("pdfjs getDocument no disponible");
  const pdf = await getDocument({ data }).promise;
  let out = "";
  for (let i=1;i<=pdf.numPages;i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    out += " " + content.items.map(it => (typeof it.str==="string"? it.str:"")).join(" ");
  }
  return clean(out);
}

function chunk(text,size=1100,overlap=150){
  const chunks=[]; let i=0;
  while(i<text.length){
    const s=clean(text.slice(i,i+size));
    if (s.length>=120) chunks.push(s);
    i+=Math.max(1,size-overlap);
  }
  return chunks;
}

async function* walkPdfs(dir){
  const entries=await fsp.readdir(dir,{withFileTypes:true});
  for(const e of entries){
    const p=path.join(dir,e.name);
    if(e.isDirectory()) yield* walkPdfs(p);
    else if(e.isFile() && /\.pdf$/i.test(e.name)) yield p;
  }
}

async function embedOne(text){
  const r=await client.embeddings.create({ model: EMB_MODEL, input: text });
  return r.data[0].embedding;
}

(async ()=>{
  if(!process.env.OPENAI_API_KEY){ console.error("Falta OPENAI_API_KEY"); process.exit(1); }
  if(!fs.existsSync(DOCS_DIR)){ console.error("No existe docs/:", DOCS_DIR); process.exit(1); }
  await fsp.mkdir(OUT_DIR,{recursive:true});

  const pdfs=[]; for await (const p of walkPdfs(DOCS_DIR)) pdfs.push(p);
  if(pdfs.length===0) console.warn("No se encontraron PDFs en docs/");

  const out=[];
  for(let fi=0; fi<pdfs.length; fi++){
    const abs=pdfs[fi]; const name=path.basename(abs); const title=name.replace(/\.pdf$/i,"");
    console.log(`[${fi+1}/${pdfs.length}] ${name} → extrayendo texto…`);
    const text=await extractPdfText(abs);
    const parts=chunk(text);
    if(parts.length===0){ console.warn(`   ⚠️  ${name}: sin texto utilizable (¿escaneado?). Omitido.`); continue; }
    for(let i=0;i<parts.length;i++){
      const vec=await embedOne(parts[i]);
      out.push({ source:name, title, text:parts[i], embedding:vec });
      if((i+1)%10===0) console.log(`   → ${i+1}/${parts.length} chunks`);
    }
  }

  await fsp.writeFile(OUT_FILE, JSON.stringify(out));
  const size=fs.statSync(OUT_FILE).size;
  console.log(`✅ Listo: ${OUT_FILE} (${size} bytes) con ${out.length} chunks de ${pdfs.length} PDF(s).`);
})();