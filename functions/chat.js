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
2. Mantén el hilo de la conversación. Si el usuario hace una pregunta de seguimiento, usa el contexto anterior.
3. Tono profesional, directo y preventivo.
`;

const json = (o, status = 200) => ({
  statusCode: status,
  headers: { "content-type": "application/json; charset=utf-8" },
  body: JSON.stringify(o),
});

const dot = (a, b) => a.reduce((sum, v, i) => sum + v * (b[i] || 0), 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosSim = (a, b) => {
  const na = norm(a), nb = norm(b);
  return !na || !nb ? 0 : dot(a, b) / (na * nb);
};

let CACHE_DATA = null;

export async function handler(event) {
  if (event.httpMethod !== "POST") return json({ error: "Method Not Allowed" }, 405);

  try {
    const { query, history = [] } = JSON.parse(event.body || "{}");
    
    if (!CACHE_DATA) {
      if (!fs.existsSync(DATA_FILE)) return json({ answer: "Cargando base de conocimientos..." });
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    // Buscamos contexto usando la pregunta actual + la anterior para mejor precisión
    const lastContext = history.length > 0 ? history[history.length - 1].content : "";
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: `${lastContext} ${query}`.slice(0, 2000),
    });
    const qVec = qEmb.data[0].embedding;

    const topChunks = CACHE_DATA.chunks
      .map(c => ({ ...c, score: cosSim(qVec, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const contextText = topChunks.map(c => c.text).join("\n\n");

    const messages = [
      { role: "system", content: INOCUO_SYSTEM_PROMPT },
      { role: "system", content: "CONTEXTO TÉCNICO:\n" + contextText },
      ...history.slice(-6), // Enviamos los últimos 6 mensajes para contexto
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
      // Devolvemos el historial actualizado para que el frontend lo guarde
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    return json({ error: "Error técnico: " + err.message }, 500);
  }
}
