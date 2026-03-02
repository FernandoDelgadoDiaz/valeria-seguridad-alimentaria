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
    const { query, history = [], mode = "tecnico" } = JSON.parse(event.body || "{}");

    if (!CACHE_DATA) {
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    // Detectar intenciones del usuario
    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);

    // 1. Buscar fragmentos relevantes
    const lastMsg = history.length > 0 ? history[history.length - 1].content : "";
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: `${lastMsg} ${query}`.slice(0, 1000),
    });

    const allChunks = CACHE_DATA.chunks.map(c => ({
      ...c,
      score: cosSim(qEmb.data[0].embedding, c.embedding || c.vec)
    }));

    const sorted = allChunks.sort((a, b) => b.score - a.score);

    // Separar internos y CAA (por nombre de archivo)
    const internos = sorted.filter(c => !c.source.toLowerCase().includes("capitulo") && !c.source.toLowerCase().includes("caa"));
    const caaChunks = sorted.filter(c => c.source.toLowerCase().includes("capitulo") || c.source.toLowerCase().includes("caa"));

    const topInternos = internos.slice(0, 4);
    const topCAA = caaChunks.slice(0, 4);

    // Decidir qué fragmentos usar según la intención
    let contextChunks = [];
    let usarCAA = false;

    if (pideExacto) {
      contextChunks = topCAA.length > 0 ? topCAA : topInternos;
      usarCAA = true;
    } else if (mencionaCAA) {
      contextChunks = [...topInternos, ...topCAA].slice(0, 8);
      usarCAA = true;
    } else {
      contextChunks = topInternos;
    }

    if (contextChunks.length === 0) {
      contextChunks = topCAA;
      usarCAA = true;
    }

    // Texto de contexto limpio (sin fuentes)
    const contextText = contextChunks.map(c => c.text).join("\n\n---\n\n");

    // Prompt base según el modo
    let systemPrompt = "";

    if (mode === "tecnico") {
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **TÉCNICO** – Tus respuestas deben ser **concisas, directas y técnicas**. Prioriza la información esencial. Si la respuesta proviene de documentos internos, no menciones la fuente. Si proviene del CAA, cita el capítulo y artículo (extrayéndolos del texto). No agregues explicaciones extensas ni ejemplos a menos que el usuario los pida explícitamente.

Jerarquía:
- Si el usuario pide un artículo exacto del CAA, dale el texto literal con la cita.
- Si menciona el CAA, combina internos + CAA con citas.
- En caso contrario, responde solo con internos y al final podés ofrecer: "¿Necesitas que consulte también el CAA para ampliar?"

CONTEXTO:
${contextText}`;
    } else { // modo "ensena"
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **ENSEÑA** – Tus respuestas deben ser **didácticas, estructuradas y pedagógicas**. 

**Estructura de tus respuestas:**
- Primero, da una definición clara del concepto.
- Luego, si corresponde, ofrece una clasificación o tipos.
- Incluye ejemplos prácticos (usa los documentos disponibles).
- Al final de tu explicación, añade siempre lo siguiente (en una línea separada):
  "**Para profundizar en este tema, responde '1'. Para hacer un test de aprendizaje, responde '2'.**"

**Manejo de las opciones:**
- Si el usuario responde '1', proporciona información adicional, más detalles o ejemplos avanzados sobre el tema.
- Si el usuario responde '2', debes generar un test de aprendizaje de 5 preguntas de opción múltiple sobre el tema que acabas de explicar. Las preguntas deben ser claras y las opciones (a, b, c) plausibles. Al final del test, indica: "Responde con un mensaje que contenga tus respuestas en orden, por ejemplo: '1b,2a,3c,4b,5a'. Luego te daré tu puntuación y te indicaré las respuestas correctas."

**Corrección del test:**
- Después de que el usuario envíe sus respuestas (en el formato indicado), debes evaluarlas. Compara con las respuestas correctas que generaste. Devuelve un mensaje con:
  - El puntaje obtenido (ej: "Has acertado 3 de 5 preguntas.")
  - Para cada pregunta incorrecta, muestra la pregunta, la respuesta del usuario y la respuesta correcta, con una breve explicación.
- Luego, ofrece continuar: "¿Quieres intentar otro test sobre el mismo tema o prefieres cambiar de tema?"

**Importante:**
- Siempre que generes un test, recuerda las preguntas y respuestas correctas para poder corregir después. Puedes mantenerlas en el historial.
- Si el usuario no sigue el formato esperado, guíalo amablemente.
- Utiliza la información del CONTEXTO para basar tus explicaciones y tests.

CONTEXTO:
${contextText}`;
    }

    // Añadir instrucciones comunes para casos especiales
    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario ha pedido el TEXTO EXACTO. Debes copiar el fragmento del CAA lo más literal posible, sin resumir. Indica el capítulo y artículo correspondientes antes del texto.`;
    }

    // Llamada a la API
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        ...history.slice(-6),
        { role: "user", content: query }
      ],
      temperature: pideExacto ? 0.1 : 0.3,
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