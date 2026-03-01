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

    // 1. Buscamos en los manuales usando la pregunta actual + el contexto previo
    const lastMsg = history.length > 0 ? history[history.length - 1].content : "";
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: `${lastMsg} ${query}`.slice(0, 1000),
    });

    const topChunks = CACHE_DATA.chunks
      .map(c => ({ ...c, score: cosSim(qEmb.data[0].embedding, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const contextText = topChunks.map(c => c.text).join("\n\n");

    // 2. Le damos a la IA la personalidad de INOCUO y la MEMORIA
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { 
          role: "system", 
          content: "Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM), basado en el Código Alimentario Argentino (CAA). Para responder, primero debes utilizar la información de los manuales técnicos que se te proporcionan en el contexto. Si esa información es suficiente, responde basándote en ella. Si no encuentras información relevante en los manuales, entonces puedes recurrir a tu conocimiento del CAA. Siempre que sea posible, combina ambas fuentes. No menciones marcas de retail. Si el usuario hace una pregunta fuera de este ámbito (por ejemplo, sobre deportes, geografía, cultura general, etc.), debes responder educadamente que solo puedes ayudar con temas de seguridad alimentaria y BPM. Si el usuario hace una pregunta de seguimiento, usa los mensajes anteriores para entender el contexto." 
        },
        { role: "system", content: "Información técnica de manuales: " + contextText },
        ...history.slice(-6), // ESTA LÍNEA ES LA MEMORIA
        { role: "user", content: query }
      ],
      temperature: 0.3,
    });

    const answer = completion.choices[0].message.content;

    return json({
      ok: true,
      answer: answer,
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    return json({ error: err.message });
  }
}