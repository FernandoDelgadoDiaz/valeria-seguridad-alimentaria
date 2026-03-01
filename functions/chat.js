import fs from "fs";
import path from "path";
import OpenAI from "openai";

// Inicialización de OpenAI
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Ruta al archivo de datos generado por el build (en Netlify)
const DATA_FILE = path.resolve("/var/task/data/embeddings.json");

/**
 * CONFIGURACIÓN DE IDENTIDAD Y COMPORTAMIENTO
 */
const INOCUO_SYSTEM_PROMPT = `
Eres INOCUO, el asistente técnico definitivo en seguridad alimentaria para PyMEs.
Tu conocimiento se basa exclusivamente en el Código Alimentario Argentino (CAA) y protocolos de Buenas Prácticas (BPM).

REGLAS CRÍTICAS:
1. IDENTIDAD: Nunca menciones a "La Anónima" o empresas específicas. Si los documentos lo mencionan, tú di "esta organización" o "el protocolo técnico".
2. CONTEXTO: Mantén el hilo de la charla. Si el usuario hace una pregunta de seguimiento (ej: "¿y cómo se clasifican?"), entiende que se refiere al tema anterior.
3. ALCANCE: Si te preguntan algo fuera de la seguridad alimentaria, responde que tu competencia es la inocuidad.
4. TONO: Profesional, preventivo, directo y experto.
`;

/**
 * UTILIDADES MATEMÁTICAS PARA COMPARAR VECTORES (Similitud de Coseno)
 */
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * (b[i] || 0), 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a), nb = norm(b);
  return !na || !nb ? 0 : dot(a, b) / (na * nb);
};

// Caché para no leer el archivo del disco en cada mensaje
let CACHE_DATA = null;

const jsonResponse = (data, status = 200) => ({
  statusCode: status,
  headers: { "Content-Type": "application/json; charset=utf-8" },
  body: JSON.stringify(data),
});

export async function handler(event) {
  // Solo aceptamos peticiones POST
  if (event.httpMethod !== "POST") return jsonResponse({ error: "Method Not Allowed" }, 405);

  try {
    const { query, history = [] } = JSON.parse(event.body || "{}");
    if (!query) return jsonResponse({ error: "Consulta vacía" }, 400);

    // 1. Cargar la base de conocimientos
    if (!CACHE_DATA) {
      if (!fs.existsSync(DATA_FILE)) {
        return jsonResponse({ ok: false, answer: "Estoy terminando de cargar mis manuales. Por favor, espera un minuto y reintenta." });
      }
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    // 2. Preparar el contexto de búsqueda (Pregunta actual + última del historial)
    const lastContext = history.length > 0 ? history[history.length - 1].content : "";
    const searchString = `${lastContext} ${query}`.slice(0, 1500);

    // 3. Generar embedding de la consulta
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: searchString,
    });
    const qVec = qEmb.data[0].embedding;

    // 4. Buscar los 4 fragmentos más relevantes en el JSON
    const topChunks = CACHE_DATA.chunks
      .map(c => ({ ...c, score: cosSim(qVec, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const contextText = topChunks.map(c => `[Fuente: ${c.source}]: ${c.text}`).join("\n\n");

    // 5. Construir los mensajes para la IA incluyendo el HISTORIAL
    const messages = [
      { role: "system", content: INOCUO_SYSTEM_PROMPT },
      { role: "system", content: "INFORMACIÓN TÉCNICA DE RESPALDO:\n" + contextText },
      ...history.slice(-6), // Enviamos los últimos 6 mensajes (3 idas y vueltas)
      { role: "user", content: query }
    ];

    // 6. Generar la respuesta final
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages,
      temperature: 0.3, // Baja creatividad para mayor precisión técnica
    });

    const answer = completion.choices[0].message.content;

    // 7. Devolver respuesta e historial actualizado
    return jsonResponse({
      ok: true,
      answer: answer,
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    console.error("Error en Inocuo:", err);
    return jsonResponse({ error: "Hubo un error al procesar tu consulta técnica." }, 500);
  }
}
