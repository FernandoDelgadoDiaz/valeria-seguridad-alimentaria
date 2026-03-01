import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_FILE = path.resolve("/var/task/data/embeddings.json");

const INOCUO_SYSTEM_PROMPT = `
Eres INOCUO, asistente experto en seguridad alimentaria para PyMEs. 
Tu conocimiento se basa en el Código Alimentario Argentino (CAA) y Buenas Prácticas (BPM).

REGLAS DE ORO:
1. No menciones marcas de retail (ej. La Anónima). Usa "esta organización" o "el protocolo".
2. MEMORIA: Mantén el hilo de la conversación. Si el usuario pregunta "qué sería eso" o "cómo se divide", refiere a lo hablado anteriormente.
3. Tono profesional y preventivo.
`;

const json = (o) => ({
  statusCode: 200,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(o),
});

// Funciones matemáticas para búsqueda semántica
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * (b[i] || 0), 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a), nb = norm(b);
  return !na || !nb ? 0 : dot(a, b) / (na * nb);
};

let CACHE_DATA = null;

export async function handler(event) {
  if (event.httpMethod !== "POST") return { statusCode: 405 };

  try {
    const { query, history = [] } = JSON.parse(event.body || "{}");
    
    if (!CACHE_DATA) {
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    // Unimos el historial para que la búsqueda de vectores sea contextual
    const lastContext = history.slice(-1).map(m => m.content).join(" ");
    const searchString = `${lastContext} ${query}`.slice(0, 1000);

    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: searchString,
    });
    const qVec = qEmb.data[0].embedding;

    const topChunks = CACHE_DATA.chunks
      .map(c => ({ ...c, score: cosSim(qVec, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const contextText = topChunks.map(c => c.text).join("\n\n");

    const messages = [
      { role: "system", content: INOCUO_SYSTEM_PROMPT },
      { role: "system", content: "CONTEXTO DE MANUALES:\n" + contextText },
      ...history.slice(-6), // Enviamos los últimos 3 pares de mensajes
      { role: "user", content: query }
    ];

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages,
      temperature: 0.3,
    });

    const answer = completion.choices[0].message.content;

    return json({
      ok: true,
      answer: answer,
      // Devolvemos el historial actualizado para que el navegador lo guarde
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    return json({ error: err.message });
  }
}
