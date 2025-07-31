// functions/valeria.js
// DIAGNÓSTICO PURO (sin OpenAI ni embeddings).
// Objetivo: confirmar que la Function se actualiza y responde lo que esperamos.

exports.handler = async (event) => {
  try {
    // Salud simple por GET
    if (event.httpMethod === "GET") {
      return {
        statusCode: 200,
        body: JSON.stringify({ reply: "DIAG GET OK — Enviá un POST con { message }." })
      };
    }

    // Parseo seguro del body
    let body = {};
    try { body = JSON.parse(event.body || "{}"); } catch {}
    const msg = (body.message || "").trim();

    // Comandos de diagnóstico
    if (msg.toLowerCase().startsWith("!!ping")) {
      return {
        statusCode: 200,
        body: JSON.stringify({ reply: `PONG ${new Date().toISOString()}` })
      };
    }

    if (msg.toLowerCase().startsWith("!!info")) {
      const hasKey = !!process.env.OPENAI_API_KEY;
      return {
        statusCode: 200,
        body: JSON.stringify({
          reply: `INFO:: function viva. OPENAI_API_KEY=${hasKey ? "presente" : "ausente"}. Ruta OK.`
        })
      };
    }

    // Respuesta por defecto (eco)
    if (msg) {
      return {
        statusCode: 200,
        body: JSON.stringify({ reply: `DIAG OK — Recibí: "${msg}"` })
      };
    }

    // Sin mensaje
    return {
      statusCode: 400,
      body: JSON.stringify({ reply: "DIAG: No recibí tu mensaje." })
    };
  } catch (err) {
    return {
      statusCode: 500,
      body: JSON.stringify({ reply: "DIAG: Error inesperado en el servidor." })
    };
  }
};
