// functions/valeria.js
// Valeria Profesional — Function completa (GET y POST) con:
// - Identidad (CAA/BPM, AR) y guardas de dominio.
// - RAG simple leyendo data/embeddings.json (incluido en el deploy).
// - Fallbacks si no hay docs o falla el modelo.
// - Diagnóstico: !!ping, !!info y !!debug <pregunta>.
// - Prueba rápida por URL: GET ?msg=...

const fs = require("fs");
const path = require("path");

// ---------- Utilidades ----------
function chunkText(text, maxLen = 1200) {
  if (!text) return [];
  const out = [];
  for (let i = 0; i < text.length; i += maxLen) out.push(text.slice(i, i + maxLen));
  return out;
}
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
function loadLocalDocs() {
  try {
    const file = path.join(__dirname, "../data/embeddings.json");
    const raw = fs.readFileSync(file, "utf8");
    const arr = JSON.parse(raw);
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

// ---------- Cache ----------
let CACHE = {
  ready: false,
  chunks: [],
  vectors: [],
  modelEmb: "text-embedding-3-small" // económico y suficiente para MVP
};

// Limitamos chunking para evitar timeouts de arranque
const MAX_CHUNKS = 150;
const BATCH = 50;

async function ensureEmbeddings(OPENAI_API_KEY) {
  if (CACHE.ready) return;

  const docs = loadLocalDocs(); // [{id,title,source,content} o {id,title,chunks:[...]}, ...]
  const chunks = [];

  for (const d of docs) {
    if (Array.isArray(d.chunks) && d.chunks.length) {
      d.chunks.forEach((c, idx) => {
        chunks.push({ id: `${d.id || "doc"}__${idx}`, title: d.title || d.id || "doc", source: d.source || "", content: c });
      });
    } else if (typeof d.content === "string" && d.content.trim()) {
      const parts = chunkText(d.content, 1200);
      parts.forEach((c, idx) => {
        chunks.push({ id: `${d.id || "doc"}__${idx}`, title: d.title || d.id || "doc", source: d.source || "", content: c });
      });
    }
  }

  const working = chunks.slice(0, MAX_CHUNKS);

  // Si no hay chunks, marcamos listo sin embeddings para que el flujo igual responda.
  if (working.length === 0) {
    CACHE.chunks = [];
    CACHE.vectors = [];
    CACHE.ready = true;
    return;
  }

  async function embedBatch(texts) {
    const resp = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: CACHE.modelEmb, input: texts })
    });
    if (!resp.ok) {
      const t = await resp.text().catch(() => "");
      throw new Error(`Fallo embeddings (${resp.status}): ${t}`);
    }
    const data = await resp.json();
    return data.data.map(d => d.embedding);
  }

  const vectors = [];
  for (let i = 0; i < working.length; i += BATCH) {
    const slice = working.slice(i, i + BATCH);
    const embs = await embedBatch(slice.map(x => x.content));
    vectors.push(...embs);
  }

  CACHE.chunks = working;
  CACHE.vectors = vectors;
  CACHE.ready = true;
}

// ---------- Núcleo de respuesta ----------
async function answerWithRAG(message, OPENAI_API_KEY, chatModel) {
  const lower = message.toLowerCase();

  // Guardas de dominio (desvía temas fuera de seguridad alimentaria)
  if (/(precio|clima|geograf|deporte|polític|politic|chisme)/i.test(lower)) {
    return "Mi alcance es seguridad alimentaria (CAA/BPM, POES, rotulado interno, no conformidades). ¿Querés que lo enfoquemos por ahí?";
  }

  // Reglas rápidas: Ishikawa / 5 porqués
  const pideIshikawa = lower.includes("espina de pescado") || lower.includes("ishikawa");
  const pide5Porques  = lower.includes("5 porqués") || lower.includes("5 porques") || lower.includes("cinco porqués");
  if (pideIshikawa || pide5Porques) {
    const problema = (lower.includes("vencid") && (lower.includes("góndola") || lower.includes("gondola")))
      ? "Productos vencidos en góndola"
      : message.trim();

    const ishikawa =
`🔎 *Análisis de causa raíz — Diagrama de Ishikawa (Espina de pescado)*
**Problema:** ${problema}

**Métodos / Procedimientos**
- Falta de retiro anticipado según norma interna (10 días masivos/congelados; 2 días perecederos).
- Rotación FEFO no aplicada en reposición.

**Mano de obra**
- Falta de capacitación en control de fechas / rotación.
- Altas cargas de trabajo en punta (se omiten controles).

**Materiales**
- Rotulado interno incompleto o ilegible.
- Etiquetas sin fecha de apertura / fraccionado.

**Maquinaria / Equipos**
- Scanner o app de control de fechas no utilizada o con fallos.
- Heladeras con stock sobrecargado que dificulta la revisión.

**Medio ambiente**
- Exhibidores desordenados, visibilidad pobre de fechas.
- Señalización de “retiro anticipado” ausente.

**Medición / Control**
- Falta de checklist diario de vencimientos por góndola/sector.
- Registros incompletos o no verificados por un responsable.`;

    const cincoPorques =
`\n\n🧠 *5 porqués (ejemplo para “${problema}”)*  
1. ¿Por qué hay productos vencidos en góndola?  
   → Porque no se retiraron a tiempo.  
2. ¿Por qué no se retiraron a tiempo?  
   → Porque no se realizó el control diario de vencimientos.  
3. ¿Por qué no se realizó el control diario?  
   → Porque el personal no tenía asignada la tarea con horario y responsable.  
4. ¿Por qué no estaba asignada?  
   → Porque el procedimiento no lo define claramente y no hay checklist operativo.  
5. ¿Por qué no hay checklist y definición?  
   → Porque no se integró la norma de retiro anticipado (10 días/2 días) al procedimiento y la capacitación.

✅ *Acciones inmediatas*  
- Barrido de góndolas por sector y retiro anticipado (según norma).  
- Señalizar y registrar el retiro (RG 01-SUC-ML-002).

🛠️ *Acciones preventivas*  
- Actualizar el procedimiento con retiro anticipado y responsable.  
- Implementar checklist diario y verificación del supervisor.  
- Capacitación en CAA/BPM, FEFO y retiro anticipado.  
- Auditoría semanal de cumplimiento.`;

    return ishikawa + cincoPorques;
  }

  // Asegurar embeddings (si falla, seguimos sin contexto)
  let haveDocs = false, context = "", topKMeta = [];
  try {
    await ensureEmbeddings(OPENAI_API_KEY);
    haveDocs = CACHE.chunks.length > 0;

    if (haveDocs) {
      // Embedding de la consulta
      const embResp = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: CACHE.modelEmb, input: message })
      });
      if (embResp.ok) {
        const embData = await embResp.json();
        const qVec = embData.data?.[0]?.embedding || null;

        if (qVec) {
          const scored = CACHE.chunks.map((c, i) => ({ c, s: cosineSim(qVec, CACHE.vectors[i]) || 0 }));
          scored.sort((a,b) => b.s - a.s);
          const topK = scored.slice(0, 5).map(x => x.c);
          topKMeta = topK.map(t => ({ id: t.id, title: t.title }));
          context = topK.map(d => `• ${d.title}: ${d.content}`).join("\n");
        }
      }
    }
  } catch {
    // Ignoramos, seguiremos con fallback
  }

  const systemPrompt = `
Sos **Valeria**, Licenciada en Tecnología de los Alimentos (Argentina), especialista en seguridad alimentaria.
Usá terminología local (carne vacuna, freezer, heladera, franco), fundamento en CAA/BPM y procedimientos internos.
Tono profesional y pedagógico; pasos concretos; ofrecé adjuntos PDF cuando corresponda (✔️).
NO copies conversaciones previas ni formatees como transcripción. Respondé directo a la consulta.`;

  const contextPrompt = context
    ? `Contexto interno (resumido):\n${context}\n\nRestringí tus afirmaciones técnicas a este contexto cuando aplique, citando títulos.`
    : `No se encontraron coincidencias en contexto interno. Respondé desde criterios generales de CAA/BPM (aclaralo).`;

  // Si no hay API Key, devolvemos respuesta general sin modelo
  if (!OPENAI_API_KEY) {
    return `No tengo la API Key configurada en el servidor, pero te dejo una guía general (CAA/BPM):\n\n• Separación de crudos y listos para consumo.\n• Cadena de frío: 0–4 °C (refrigeración) y ≤ −18 °C (congelación).\n• Rotación FEFO y retiro anticipado (10 días masivos/congelados; 2 días perecederos).\n• Limpieza y desinfección planificadas; verificación y registros.\n\nPodés pedir: “¿Cómo preparo J-512 a 200 ppm?” o “¿Cómo rotulo fracciones de queso?”`;
  }

  // Llamada al modelo
  const chatResp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: chatModel || "gpt-4o-mini",
      temperature: 0.35,
      max_tokens: 700,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "system", content: contextPrompt },
        { role: "user", content: message }
      ]
    })
  });

  if (!chatResp.ok) {
    const t = await chatResp.text().catch(() => "");
    // Fallback si el modelo falla
    return `No pude obtener respuesta del modelo (HTTP ${chatResp.status}). Guía general CAA/BPM:\n\n• Separación de crudos y listos para consumo.\n• Cadena de frío: 0–4 °C y ≤ −18 °C.\n• Rotación FEFO y retiro anticipado (10 días/2 días).\n• Limpieza y desinfección planificadas.\n\nDetalle técnico: ${t.slice(0, 500)}`;
  }

  const data = await chatResp.json();
  const reply = data?.choices?.[0]?.message?.content || "";

  if (reply && reply.trim()) return reply;

  // Fallback si vino vacío
  return context
    ? `No obtuve texto del modelo en este intento. Basado en el contexto:\n\n${context.slice(0, 1200)}`
    : `No obtuve texto del modelo. Guía general CAA/BPM:\n\n• Separación de crudos y listos para consumo.\n• Cadena de frío: 0–4 °C y ≤ −18 °C.\n• Rotación FEFO y retiro anticipado (10 días/2 días).\n• Limpieza y desinfección planificadas.`;
}

// ---------- Handler ----------
exports.handler = async (event) => {
  try {
    // Preflight CORS
    if (event.httpMethod === "OPTIONS") {
      return { statusCode: 204, headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type, Authorization", "Access-Control-Allow-Methods": "GET, POST, OPTIONS" }, body: "" };
    }

    const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
    const CHAT_MODEL = process.env.VALERIA_CHAT_MODEL || "gpt-4o-mini";

    // GET con ?msg= para pruebas rápidas pegando la URL
    if (event.httpMethod === "GET") {
      const url = new URL(event.rawUrl || (event.headers.host ? `https://${event.headers.host}${event.path}` : ""));
      const msg = url.searchParams.get("msg") || "";
      if (!msg) {
        return { statusCode: 200, body: JSON.stringify({ reply: "DIAG GET OK — Enviá un POST con { message } o agregá ?msg=..." }) };
      }
      // Comandos rápidos por GET
      const m = msg.trim();
      if (m.toLowerCase().startsWith("!!ping")) {
        return { statusCode: 200, body: JSON.stringify({ reply: `PONG ${new Date().toISOString()}` }) };
      }
      if (m.toLowerCase().startsWith("!!info")) {
        return { statusCode: 200, body: JSON.stringify({ reply: `INFO:: function viva. OPENAI_API_KEY=${OPENAI_API_KEY ? "presente" : "ausente"}.` }) };
      }
      if (m.toLowerCase().startsWith("!!debug")) {
        // Estado mínimo de embeddings
        let status = { ready: CACHE.ready, chunks: CACHE.chunks.length, vectors: CACHE.vectors.length };
        return { statusCode: 200, body: JSON.stringify({ reply: `DEBUG:: ${JSON.stringify(status)}` }) };
      }
      const ans = await answerWithRAG(m, OPENAI_API_KEY, CHAT_MODEL);
      return { statusCode: 200, body: JSON.stringify({ reply: ans }) };
    }

    // POST (flujo normal del front)
    if (event.httpMethod !== "POST") {
      return { statusCode: 405, body: JSON.stringify({ reply: "Usá POST con JSON { message } o GET ?msg=..." }) };
    }

    let body = {};
    try { body = JSON.parse(event.body || "{}"); } catch {}
    const message = (body.message || "").trim();
    if (!message) {
      return { statusCode: 400, body: JSON.stringify({ reply: "No recibí tu mensaje." }) };
    }

    // Atajos por POST
    const lower = message.toLowerCase();
    if (lower.startsWith("!!ping")) {
      return { statusCode: 200, body: JSON.stringify({ reply: `PONG ${new Date().toISOString()}` }) };
    }
    if (lower.startsWith("!!info")) {
      return { statusCode: 200, body: JSON.stringify({ reply: `INFO:: function viva. OPENAI_API_KEY=${OPENAI_API_KEY ? "presente" : "ausente"}.` }) };
    }
    if (lower.startsWith("!!debug")) {
      let status = { ready: CACHE.ready, chunks: CACHE.chunks.length, vectors: CACHE.vectors.length };
      return { statusCode: 200, body: JSON.stringify({ reply: `DEBUG:: ${JSON.stringify(status)}` }) };
    }

    const ans = await answerWithRAG(message, OPENAI_API_KEY, CHAT_MODEL);
    return { statusCode: 200, body: JSON.stringify({ reply: ans }) };

  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ reply: "Error inesperado en el servidor." }) };
  }
};
