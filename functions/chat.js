import fs from "fs";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_FILE = "/var/task/data/embeddings.json";
const VERSION = "v2026-Inocuo-Final";

// --- PROMPT MAESTRO DE INOCUO ---
const INOCUO_SYSTEM_PROMPT = `
Eres INOCUO, un experto en seguridad alimentaria para PyMEs. 
Tu misión es asesorar basándote en el Código Alimentario Argentino (CAA) y Manuales de Buenas Prácticas (BPM).

REGLAS CRÍTICAS:
1. NUNCA menciones nombres de empresas de retail o supermercados (ej. Prohibido decir 'La Anónima').
2. Si los documentos fuente mencionan una marca o empresa, reemplázala por "esta normativa" o "el protocolo profesional".
3. SIEMPRE di: "Según las Buenas Prácticas..." o "De acuerdo al CAA...".
4. Si te preguntan algo fuera de la seguridad alimentaria, di que no es tu competencia.
5. Responde de forma técnica pero accesible para un dueño de un pequeño comercio.
`;

const json = (o, status = 200) => ({
  statusCode: status,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(o),
});

// --- LÓGICA DE VECTORES (Tuya, optimizada) ---
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => dot(a, b) / (norm(a) * norm(b));

async function embed(text) {
  const { data } = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return data[0].embedding;
}

// Carga de datos con caché
let CACHE_CHUNKS = null;
async function getChunks() {
  if (CACHE_CHUNKS) return CACHE_CHUNKS;
  if (!fs.existsSync(DATA_FILE)) return [];
  try {
    const raw = fs.readFileSync(DATA_FILE, "utf8");
    const data = JSON.parse(raw);
    // Asumimos estructura de chunks con vectores
    CACHE_CHUNKS = data.chunks || data; 
    return CACHE_CHUNKS;
  } catch (e) { return []; }
}

export async function handler(event) {
  if (event.httpMethod !== "POST") return json({ error: "Usar POST" }, 405);

  try {
    const { query } = JSON.parse(event.body || "{}");
    if (!query) return json({ error: "Falta consulta" }, 400);

    const chunks = await getChunks();
    if (chunks.length === 0) return json({ answer: "Inocuo está cargando su base de conocimientos. Reintenta en breve." });

    // 1. Vectorizar la pregunta del usuario
    const qVec = await embed(query);

    // 2. Buscar los 5 fragmentos más parecidos (RAG)
    const context = chunks
      .map(c => ({ ...c, score: cosSim(qVec, c.vec || c.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(c => c.text)
      .join("\n\n---\n\n");

    // 3. Generar respuesta con la IA usando el Prompt de Inocuo
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini", // O gpt-3.5-turbo
      messages: [
        { role: "system", content: INOCUO_SYSTEM_PROMPT },
        { role: "system", content: "CONTEXTO TÉCNICO:\n" + context },
        { role: "user", content: query }
      ],
      temperature: 0.3 // Baja temperatura para que no invente (sea preciso)
    });

    return json({
      ok: true,
      answer: completion.choices[0].message.content,
      version: VERSION
    });

  } catch (e) {
    console.error(e);
    return json({ error: "Error interno en Inocuo" }, 500);
  }
}
