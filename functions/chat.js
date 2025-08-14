// functions/chat.js  (v2025-08-13-fix5)
import fs from "fs";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DOCS_DIR = "/var/task/docs";
const DATA_FILE = "/var/task/data/embeddings.json";
const VERSION = "v2025-08-13-fix5";

const json = (o, status = 200) => ({
  statusCode: status,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(o),
});

const normalize = (s = "") =>
  s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");

const dot = (a, b) => {
  const n = Math.min(a?.length || 0, b?.length || 0);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += (a[i] || 0) * (b[i] || 0);
  return sum;
};
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a), nb = norm(b);
  return !na || !nb ? 0 : dot(a, b) / (na * nb);
};

async function embed(text) {
  const { data } = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return data[0].embedding;
}

function safeReadText(p) { try { return fs.readFileSync(p, "utf8"); } catch { return ""; } }
function tryJSON(t) { try { return JSON.parse(t); } catch { return null; } }

function looksNumberArray(x) {
  return Array.isArray(x) && x.length > 8 && x.every((v) => typeof v === "number");
}
function coerceNumberArray(x) {
  if (looksNumberArray(x)) return x;
  if (Array.isArray(x) && x.length > 8) {
    const n = x.map(Number);
    if (n.every((v) => Number.isFinite(v))) return n;
  }
  if (typeof x === "string" && x.includes(",")) {
    const n = x.split(",").map((v) => Number(v.trim()));
    if (n.every((v) => Number.isFinite(v)) && n.length > 8) return n;
  }
  return null;
}

// Posibles claves (inglés + español con/sin tilde)
const TEXT_KEYS   = ["text","content","body","chunk","pageText","raw","t","texto","frase","contenido","fragmento","resumen","snippet"];
const TITLE_KEYS  = ["title","heading","h","titulo","título"];
const SOURCE_KEYS = ["source","src","file","path","doc","s","fuente","archivo","ruta"];
const VEC_KEYS    = ["vec","embedding","emb","v","e","vector","incrustacion","incrustación"];

// Posibles arrays paralelos en raíz
const TOP_TEXT_ARRS   = ["texts","contents","bodies","snippets","pages","chunksText","textos","frases","fragmentos","contenido"];
const TOP_TITLE_ARRS  = ["titles","headings","titulos","títulos"];
const TOP_SOURCE_ARRS = ["sources","files","paths","docs","fuentes","archivos","rutas"];

function* deepArrays(node, seen = new Set()) {
  if (!node || typeof node !== "object") return;
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node) && node.length && typeof node[0] === "object") yield node;
  for (const k of Object.keys(node)) {
    const v = node[k];
    if (Array.isArray(v) || (v && typeof v === "object")) yield* deepArrays(v, seen);
  }
}

function getFirstArrayOfStrings(obj, names) {
  for (const name of names) {
    const arr = obj?.[name];
    if (Array.isArray(arr) && arr.length && arr.every((x) => typeof x === "string")) {
      return arr;
    }
  }
  // también por claves normalizadas (sin tildes)
  const map = Object.fromEntries(Object.keys(obj || {}).map(k => [normalize(k), k]));
  for (const name of names) {
    const k = map[normalize(name)];
    const arr = obj?.[k];
    if (Array.isArray(arr) && arr.length && arr.every((x) => typeof x === "string")) {
      return arr;
    }
  }
  return null;
}

function longestStringInObject(o) {
  let best = "";
  for (const [k, v] of Object.entries(o)) {
    if (typeof v === "string" && v.length > best.length && !/\.(pdf|docx?|xlsx?)$/i.test(v)) {
      best = v;
    }
  }
  return best;
}

function mapArrayToChunks(arr, rootForIndexed) {
  const out = [];
  // Arrays paralelos en raíz (para usar idx)
  const topTexts   = getFirstArrayOfStrings(rootForIndexed, TOP_TEXT_ARRS) || [];
  const topTitles  = getFirstArrayOfStrings(rootForIndexed, TOP_TITLE_ARRS) || [];
  const topSources = getFirstArrayOfStrings(rootForIndexed, TOP_SOURCE_ARRS) || [];

  for (let i = 0; i < arr.length; i++) {
    const o = arr[i];
    if (!o || typeof o !== "object") continue;

    // texto direct
    let textKey = TEXT_KEYS.find(k => typeof o[k] === "string");
    let text = textKey ? o[textKey] : "";

    // título/fuente direct
    const titleKey  = TITLE_KEYS.find(k => typeof o[k] === "string");
    const sourceKey = SOURCE_KEYS.find(k => typeof o[k] === "string");
    let title = titleKey ? o[titleKey] : "";
    let source = sourceKey ? o[sourceKey] : "";

    // vector directo o anidado
    let vec = null;
    for (const vk of VEC_KEYS) { if (vk in o) { vec = coerceNumberArray(o[vk]); if (vec) break; } }
    if (!vec) {
      for (const k of Object.keys(o)) {
        const v = o[k];
        if (v && typeof v === "object") {
          for (const vk of VEC_KEYS) { if (vk in v) { vec = coerceNumberArray(v[vk]); if (vec) break; } }
          if (vec) break;
        }
      }
    }

    // si viene con idx, tomar de arrays paralelos
    const idx = Number.isInteger(o.idx) ? o.idx : (Number.isInteger(o.i) ? o.i : null);
    if (!text && idx != null && topTexts[idx]) text = topTexts[idx];
    if (!title && idx != null && topTitles[idx]) title = topTitles[idx];
    if (!source && idx != null && topSources[idx]) source = topSources[idx];

    // último recurso: cadena más larga del objeto
    if (!text) text = longestStringInObject(o);

    if (text && vec) {
      out.push({
        id: o.id ?? o.i ?? idx ?? i,
        title,
        source,
        text,
        vec,
      });
    }
  }
  return out;
}

function mapParallel(obj) {
  const emb = obj.embeddings || obj.vectors || obj.vecs || obj.incrustaciones;
  const texts = obj.texts || obj.contents || obj.bodies || obj.textos || obj.frases || obj.fragmentos;
  const titles = obj.titles || obj.titulos || obj["títulos"];
  const sources = obj.sources || obj.fuentes || obj.archivos || obj.rutas;

  if (Array.isArray(emb) && Array.isArray(texts) && emb.length === texts.length && emb.length) {
    const out = [];
    for (let i = 0; i < emb.length; i++) {
      const v = coerceNumberArray(emb[i]);
      const t = texts[i];
      if (v && typeof t === "string") {
        out.push({
          id: i,
          title: titles?.[i] || "",
          source: sources?.[i] || "",
          text: t,
          vec: v,
        });
      }
    }
    return out;
  }
  return [];
}

function parseNDJSON(text) {
  const lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const objs = [];
  for (const ln of lines) { try { objs.push(JSON.parse(ln)); } catch {} }
  return objs.length ? objs : null;
}

function pickChunksFromAny(root) {
  if (Array.isArray(root)) return mapArrayToChunks(root, {});

  const par = mapParallel(root || {});
  if (par.length) return par;

  // buscar arrays candidatas en raíz
  let arr =
    root?.chunks || root?.items || root?.entries || root?.data ||
    root?.fragments || root?.fragmentos || root?.incrustaciones;

  if (arr && !Array.isArray(arr) && typeof arr === "object") arr = Object.values(arr);

  if (Array.isArray(arr)) {
    const a2 = mapArrayToChunks(arr, root); // <- pasa root para usar idx
    if (a2.length) return a2;
  }

  // buscar profundo
  if (root && typeof root === "object") {
    for (const arr2 of deepArrays(root)) {
      const a3 = mapArrayToChunks(arr2, root);
      if (a3.length) return a3;
    }
  }
  return [];
}

let CACHE = null;

async function loadData(peekOnly = false) {
  if (CACHE && !peekOnly) return CACHE;

  const exists = fs.existsSync(DATA_FILE);
  const stats = exists ? fs.statSync(DATA_FILE) : null;

  let chunks = [];
  let dims = 0;
  const peek = { parsed: false };

  if (exists) {
    const text = safeReadText(DATA_FILE);
    let parsed = tryJSON(text);
    if (!parsed) parsed = parseNDJSON(text);

    if (parsed != null) {
      peek.parsed = true;
      if (Array.isArray(parsed)) {
        chunks = mapArrayToChunks(parsed, {});
        peek.mode = "array";
      } else {
        // guardar algunas pistas para debug si fallara
        peek.rootKeys = Object.keys(parsed).slice(0, 30);
        if (Array.isArray(parsed.incrustaciones) && parsed.incrustaciones[0]) {
          peek.firstItemKeys = Object.keys(parsed.incrustaciones[0]).slice(0, 30);
        }
        const parallel = mapParallel(parsed);
        chunks = parallel.length ? parallel : pickChunksFromAny(parsed);
        peek.mode = parallel.length ? "parallel" : "object";
      }
    }

    if (chunks.length) {
      dims = Array.isArray(chunks[0].vec) ? chunks[0].vec.length : 0;
      for (const c of chunks) c._norm = normalize(c.text);
      peek.first = {
        keys: Object.keys(chunks[0]).slice(0, 8),
        vecLen: dims,
        title: chunks[0].title,
        source: chunks[0].source,
        textSample: chunks[0].text.slice(0, 120),
      };
    } else {
      peek.error = "No se encontraron vectores/textos en el archivo.";
      peek.head = text.slice(0, 2000);
    }
  }

  const payload = {
    version: VERSION,
    docsCount: fs.existsSync(DOCS_DIR)
      ? fs.readdirSync(DOCS_DIR).filter(f => f.toLowerCase().endsWith(".pdf")).length
      : 0,
    dataFileExists: !!exists,
    dataFileSize: stats?.size || 0,
    chunks,
    dims,
    peek,
  };

  if (!peekOnly) CACHE = payload;
  return payload;
}

function topK(chunks, qVec, k = 5) {
  const scored = [];
  for (const c of chunks) {
    const s = cosSim(qVec, c.vec);
    if (s > 0) scored.push({ score: s, title: c.title, source: c.source });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

export async function handler(event) {
  const rawUrl = event.rawUrl || "";
  const method = event.httpMethod || "GET";
  const url = new URL(rawUrl);

  const isPing = rawUrl.includes("/api/ping");
  const isDiag = rawUrl.includes("/api/diag");

  if (isPing) return json({ ok: true, version: VERSION });

  if (isDiag) {
    const q = url.searchParams.get("q") || "";
    const k = Math.max(1, Math.min(20, Number(url.searchParams.get("k")) || 5));
    const peek = url.searchParams.get("peek");

    const data = await loadData(!!peek);

    if (peek) {
      const d = await loadData(true);
      return json({
        ok: true,
        version: d.version,
        docs: d.docsCount,
        dataFileExists: d.dataFileExists,
        dataFileSize: d.dataFileSize,
        embeddings: d.chunks?.length || 0,
        dims: d.dims || 0,
        peek: d.peek,
        hint: "Quitá &peek=1 para ver coincidencias con ?q=texto&k=5",
      });
    }

    if (!q) {
      return json({
        ok: true,
        version: data.version,
        docs: data.docsCount,
        dataFileExists: data.dataFileExists,
        dataFileSize: data.dataFileSize,
        embeddings: data.chunks.length,
        dims: data.dims,
        hint: "Agregue ?q=texto&k=5 para probar coincidencias. Ejemplo: /api/diag?q=BPM&k=5",
      });
    }

    try {
      const qVec = await embed(q);
      const tops = topK(data.chunks, qVec, k);
      const maxScore = tops.length ? tops[0].score : 0;
      return json({
        ok: true,
        version: data.version,
        docs: data.docsCount,
        dataFileExists: data.dataFileExists,
        dataFileSize: data.dataFileSize,
        consulta: q,
        máximos: tops,
        "puntuación máxima": maxScore,
      });
    } catch (e) {
      return json({ ok: false, error: String(e?.message || e) }, 500);
    }
  }

  if (method === "POST") {
    try {
      const body = JSON.parse(event.body || "{}");
      const userQuery = body.query || body.q || "";
      const k = Math.max(1, Math.min(10, Number(body.k) || 5));

      const data = await loadData();
      const q = userQuery ||
        (body.messages || []).map((m) => m?.content).join(" ").slice(0, 2000) || "";

      const qVec = await embed(q);
      const tops = topK(data.chunks, qVec, k);

      return json({
        ok: true,
        answer: tops.length
          ? "Encontré referencias útiles en tus documentos."
          : "No encontré coincidencias claras en los documentos.",
        refs: tops,
      });
    } catch (e) {
      return json({ ok: false, error: String(e?.message || e) }, 500);
    }
  }

  return json({ ok: false, error: "Método no permitido. Usá POST." }, 405);
}