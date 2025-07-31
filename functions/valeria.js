// functions/valeria.js
const documents = require("../data/embeddings.json");

exports.handler = async (event) => {
  try {
    const { message } = JSON.parse(event.body || "{}");

    if (!message) {
      return { statusCode: 400, body: JSON.stringify({ reply: "No recibí tu mensaje." }) };
    }

    const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
    if (!OPENAI_API_KEY) {
      return { statusCode: 500, body: JSON.stringify({ reply: "❌ API Key no configurada en Netlify." }) };
    }

    // Búsqueda starter por coincidencia de texto (simple)
    const relevant = documents
      .map((doc) => ({
        ...doc,
        score: doc.content.toLowerCase().includes(message.toLowerCase()) ? 1 : 0,
      }))
      .filter((d) => d.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    const context = relevant.map((d) => `• ${d.title}: ${d.content}`).join("\n");

    const prompt =
`Sos Valeria, Licenciada en Tecnología de los Alimentos especializada en seguridad alimentaria (Argentina).
Usá terminología local (carne vacuna, freezer) y fundamentos del CAA/BPM y procedimientos internos.
Respondé claro, con pasos concretos, y ofrecé adjuntos PDF si aplica (✔️).

Contexto interno:
${context || "(sin coincidencias, responder con criterios generales BPM/CAA)"}

Usuario: ${message}
Valeria:`;

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.4,
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return { statusCode: 502, body: JSON.stringify({ reply: `Error con OpenAI (${resp.status}). ${errText}` }) };
    }

    const data = await resp.json();
    const reply = data?.choices?.[0]?.message?.content || "No se encontró respuesta.";

    return { statusCode: 200, body: JSON.stringify({ reply }) };
  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ reply: "Error inesperado en el servidor." }) };
  }
};
