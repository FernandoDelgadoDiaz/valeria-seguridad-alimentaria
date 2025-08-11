// functions/chat.js
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMB_MODEL = process.env.EMB_MODEL || "text-embedding-3-small";

// --- localizar /data en cualquier escenario de empaquetado ---
function findDataDir() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname  = path.dirname(__filename);
  const candidates = [
    path.join(__dirname, "data"),
    path.join(__dirname, "..", "data"),
    path.join(__dirname, "..", "..", "data"),
    path.join(process.cwd(), "data"),
    path.resolve("data"),
  ];
  for (const d of candidates) {
    const c = path.join(d, "chunks.jsonl");
    const e = path.join(d, "embeddings.jsonl");
    if (fs.existsSync(c) && fs.existsSync(e)) return d;
  }
  return null;
}

let LOADED = false, CHUNKS = [], EMBS = [], DATADIR = null;
function loadOnce() {
  if (LOADED) return;
  DATADIR = findDataDir();
  if (DATADIR) {
    const chunksPath = path.join(DATADIR, "chunks.jsonl");
    const embsPath   = path.join(DATADIR, "embeddings.jsonl");
    CHUNKS = fs.readFileSync(chunksPath, "utf-8").trim().split("\n").map(JSON.parse);
    EMBS   = fs.readFileSync(embsPath,   "utf-8").trim().split("\n").map(JSON.parse);
  }
  LOADED = true;
}

function cosine(a,b){ let d=0,na=0,nb=0; const n=Math.min(a.length,b.length);
  for (let i=0;i<n;i++){ const x=a[i],y=b[i]; d+=x*y; na+=x*x; nb+=y*y; }
  return d/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);
}

export default async (req, res) => {
  try {
    loadOnce();
    const { query } = JSON.parse(req.body || "{}");
    if (!query || !query.trim()) return res.status(400).json({ error:"missing query" });

    if (!DATADIR || !EMBS.length || !CHUNKS.length) {
      // Índice no generado en el build
      return res.status(200).json({ answer:
`⚠️ Aún no está listo el índice semántico.
Verificá:
1) /docs tiene PDFs (no ZIP).
2) netlify.toml usa:
   [functions]
   directory = "functions"
   node_bundler = "esbuild"
   included_files = ["data/*"]
3) Deploy con cache limpio (Clear cache and deploy site).`
      });
    }

    const e = await client.embeddings.create({ model: EMB_MODEL, input: query });
    const qv = e.data[0].embedding;

    const scored = EMBS.map((r, i)=>({ i, s: cosine(qv, r.embedding) }))
                       .sort((a,b)=>b.s-a.s).slice(0,5);

    const ctx = scored.map(({i})=>{
      const r = EMBS[i];
      return CHUNKS.find(c => c.doc===r.doc && c.chunk===r.chunk);
    }).filter(Boolean);

    const bullets = ctx.map(c => '• ' + ((c.text||'').split(/\n+/)[0].slice(0,240) || c.doc)).join("\n");
    const links = [...new Set(ctx.map(c => c.path))].slice(0,3)
      .map(p => `<a href="/${p}">PDF</a>`).join(" · ");

    return res.status(200).json({
      answer: `📌 <b>Resumen basado en documentación interna</b>:\n${bullets}\n\n🔗 ${links}`
    });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
};