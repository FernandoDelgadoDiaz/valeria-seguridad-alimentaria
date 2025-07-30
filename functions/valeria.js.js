const fetch = require("node-fetch");
const documents = require("../data/embeddings.json");

exports.handler = async (event) => {
  const { message } = JSON.parse(event.body);

  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENAI_API_KEY) {
    return { statusCode: 500, body: JSON.stringify({ reply: "❌ API Key no configurada." }) };
  }

  const relevant = documents
    .map((doc) => ({
      ...doc,
      score: doc.content.toLowerCase().includes(message.toLowerCase()) ? 1 : 0
    }))
    .filter((d) => d.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  const context = relevant.map(d => d.content).join("\n");

  const prompt = `Sos Valeria, una licenciada en Tecnología de los Alimentos especializada en seguridad alimentaria. Usando los siguientes documentos internos, respondé de manera profesional y clara:\n\n${context}\n\nUsuario: ${message}\nValeria:`;

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.4,
    }),
  });

  const data = await response.json();
  const reply = data.choices?.[0]?.message?.content || "No se encontró respuesta.";

  return {
    statusCode: 200,
    body: JSON.stringify({ reply }),
  };
};
