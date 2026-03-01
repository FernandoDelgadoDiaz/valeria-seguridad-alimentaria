import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Ruta al archivo generado por el script de build
const DATA_FILE = path.resolve("/var/task/data/embeddings.json");

/**
 * PROMPT MAESTRO: Define la personalidad, el conocimiento y las restricciones de Inocuo.
 */
const INOCUO_SYSTEM_PROMPT = `
Eres INOCUO, un asistente experto en seguridad alimentaria para PyMEs. 
Tu conocimiento se basa exclusivamente en el Código Alimentario Argentino (CAA) y Manuales de Buenas Prácticas (BPM).

REGLAS DE ORO:
1. IDENTIDAD: Nunca menciones a "La Anónima" o cualquier otra empresa de retail. Si el texto fuente lo menciona, ignóralo o di "esta organización profesional".
2. AUTORIDAD: Habla siempre en nombre de la seguridad alimentaria. Usa frases como "Según las Buenas Prácticas de Manufactura...", "De acuerdo al CAA..." o "El protocolo técnico indica...".
3. FOCO: Si te preguntan algo fuera de la inocuidad alimentaria (clima, política, etc.), responde cortésmente que tu competencia se limita a la seguridad alimentaria.
4. TONO: Profesional, preventivo, directo y didáctico para dueños de comercios.
`;

const json = (o, status = 200) => ({
  statusCode: status,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(o),
});

// Funciones matemáticas para la búsqueda semántica
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * (b[i] || 0), 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a), nb = norm(b);
  return !na || !nb ? 0 : dot(a, b) / (na * nb);
};

let CACHE_DATA = null;

async function loadKnowledge() {
  if (CACHE_DATA) return CACHE_DATA;
  if (!fs.existsSync(DATA_FILE)) return null;
  const raw = fs.readFileSync(DATA_FILE, "utf8");
  CACHE_DATA = JSON.parse(raw);
  return CACHE_DATA;
}

export async function handler(event) {
  if (event.httpMethod !== "POST") return json({ error: "Método no permitido" }, 405);

  try {
    const { query } = JSON.parse(event.body || "{}");
    if (!query) return json({ error: "Consulta vacía" }, 400);

    const data = await loadKnowledge();
    if (!data || !data.chunks) {
      return json({ answer: "Inocuo está cargando su base de conocimientos. Por favor, reintenta en unos minutos." });
    }

    // 1. Convertir la pregunta del usuario en un vector
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: query,
    });
    const qVec = qEmb.data[0].embedding;

    // 2. Buscar los 5 fragmentos más relevantes en los chunks guardados
    const topChunks = data.chunks
      .map(c => ({ ...c, score: cosSim(qVec, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    const contextText = topChunks.map(c => `[Fuente: ${c.source}]: ${c.text}`).join("\n\n---\n\n");

    // 3. Generar la respuesta usando GPT
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini", // Eficiente y preciso para este tipo de tareas
      messages: [
        { role: "system", content: INOCUO_SYSTEM_PROMPT },
        { role: "system", content: "CONTEXTO TÉCNICO RECUPERADO:\n" + contextText },
        { role: "user", content: query }
      ],
      temperature: 0.3, // Mantiene la respuesta técnica y poco creativa
    });

    return json({
      ok: true,
      answer: completion.choices[0].message.content,
      sources: topChunks.map(c => c.source) // Opcional: para control interno
    });

  } catch (err) {
    console.error("Error en Inocuo Chat:", err);
    return json({ error: "Hubo un problema técnico al procesar la consulta." }, 500);
  }
}
