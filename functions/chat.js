import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_FILE = path.resolve("/var/task/data/embeddings.json");

// Funciones matemáticas para la búsqueda semántica
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * (b[i] || 0), 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a), nb = norm(b);
  return !na || !nb ? 0 : dot(a, b) / (na * nb);
};

let CACHE_DATA = null;

const json = (o) => ({
  statusCode: 200,
  headers: { "Content-Type": "application/json; charset=utf-8" },
  body: JSON.stringify(o),
});

export async function handler(event) {
  if (event.httpMethod !== "POST") return { statusCode: 405 };

  try {
    const { query, history = [] } = JSON.parse(event.body || "{}");
    
    if (!CACHE_DATA) {
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    // 1. Buscar fragmentos relevantes
    const lastMsg = history.length > 0 ? history[history.length - 1].content : "";
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: `${lastMsg} ${query}`.slice(0, 1000),
    });

    // Recuperamos más fragmentos (8 en lugar de 4) para tener contexto amplio
    const topChunks = CACHE_DATA.chunks
      .map(c => ({ ...c, score: cosSim(qEmb.data[0].embedding, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);

    // Construimos el contexto con los textos y sus fuentes
    const contextText = topChunks
      .map(c => `[FUENTE: ${c.source}]\n${c.text}`)
      .join("\n\n---\n\n");

    // 2. Llamada a la IA con instrucciones explícitas
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM), basado exclusivamente en el Código Alimentario Argentino (CAA) y en los manuales técnicos proporcionados.

INSTRUCCIONES ESTRICTAS:
- Debes responder ÚNICAMENTE utilizando la información del CONTEXTO que se te proporciona a continuación.
- Si la pregunta del usuario solicita un artículo específico del CAA y ese artículo aparece en el CONTEXTO, debes copiarlo TEXTUALMENTE, incluyendo el número de artículo y capítulo.
- Si el CONTEXTO contiene la información necesaria, responde con ella, citando la fuente (nombre del archivo).
- Si el CONTEXTO no contiene la información solicitada, debes indicar claramente que no tienes esa información en tus manuales y ofrecer ayuda con otro tema relacionado.
- NO uses tu conocimiento general para responder preguntas sobre artículos específicos del CAA a menos que el CONTEXTO lo respalde.
- Cuando proporciones un fragmento textual, indícalo con comillas o formato claro.
- Si el usuario pregunta por algo fuera de seguridad alimentaria, responde amablemente que solo puedes ayudar con temas del CAA y BPM.

CONTEXTO:
${contextText}`
        },
        ...history.slice(-6),
        { role: "user", content: query }
      ],
      temperature: 0.1, // Muy baja para respuestas literales
    });

    const answer = completion.choices[0].message.content;

    return json({
      ok: true,
      answer: answer,
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    console.error("Error en handler:", err);
    return json({ error: err.message });
  }
}