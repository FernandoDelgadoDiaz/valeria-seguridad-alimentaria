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

    // Detectar si el usuario pide explícitamente un artículo/capítulo del CAA o menciona "CAA"
    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);

    // 1. Buscamos fragmentos relevantes
    const lastMsg = history.length > 0 ? history[history.length - 1].content : "";
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: `${lastMsg} ${query}`.slice(0, 1000),
    });

    const allChunks = CACHE_DATA.chunks.map(c => ({
      ...c,
      score: cosSim(qEmb.data[0].embedding, c.embedding || c.vec)
    }));

    // Ordenar por similitud
    const sorted = allChunks.sort((a, b) => b.score - a.score);

    // Separar internos (que no sean del CAA) y CAA (por nombre de archivo)
    const internos = sorted.filter(c => !c.source.toLowerCase().includes("capitulo") && !c.source.toLowerCase().includes("caa"));
    const caaChunks = sorted.filter(c => c.source.toLowerCase().includes("capitulo") || c.source.toLowerCase().includes("caa"));

    // Tomar los mejores de cada grupo
    const topInternos = internos.slice(0, 4);
    const topCAA = caaChunks.slice(0, 4);

    // Decidir qué contexto usar según la intención del usuario
    let contextChunks = [];
    let usarCAA = false;

    if (pideExacto) {
      // Priorizar CAA
      contextChunks = topCAA.length > 0 ? topCAA : topInternos;
      usarCAA = true;
    } else if (mencionaCAA) {
      // Combinar internos + CAA
      contextChunks = [...topInternos, ...topCAA].slice(0, 8);
      usarCAA = true;
    } else {
      // Por defecto, solo internos
      contextChunks = topInternos;
    }

    // Si no hay chunks de internos, usar CAA como fallback
    if (contextChunks.length === 0) {
      contextChunks = topCAA;
      usarCAA = true;
    }

    // Construir el texto de contexto SIN incluir la fuente explícita
    const contextText = contextChunks.map(c => c.text).join("\n\n---\n\n");

    // Prompt dinámico según el modo
    let systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a:

- Documentos internos (manuales, BPM, procedimientos)
- Código Alimentario Argentino (CAA)

Debes seguir esta jerarquía:

1. Si el usuario pide EXPLÍCITAMENTE un artículo o capítulo del CAA (ej: "dame el artículo 5 del capítulo 1"), responde con el TEXTO EXACTO del fragmento del CAA, citando la fuente de manera natural: extrae del propio texto el capítulo y artículo (por ejemplo, "Según el Capítulo I, Artículo 2 del CAA: ..."). No uses el nombre del archivo.

2. Si el usuario menciona "CAA", "código alimentario", "artículo" o "capítulo" pero no pide el texto exacto, combina información de documentos internos y del CAA para dar una respuesta completa. Para el CAA, cita siempre el capítulo y artículo correspondientes (extrayéndolos del texto). Para los internos, no menciones la fuente.

3. En caso contrario, responde principalmente con los documentos internos. Si con ellos es suficiente, no agregues CAA. Al final de la respuesta, puedes preguntar: "¿Necesitas que consulte también el Código Alimentario Argentino para ampliar?" Cuando uses internos, no hagas referencia al nombre del archivo ni a la fuente.

4. Siempre que uses información del CAA, debes indicar el capítulo y artículo (por ejemplo, "Capítulo I, Artículo 2") basándote en el texto del fragmento. No uses el nombre del archivo.

5. Si la pregunta no es sobre seguridad alimentaria, responde amablemente que solo puedes ayudar con temas del CAA y BPM.

CONTEXTO ACTUAL:
${contextText}`;

    // Añadir instrucción extra si se requiere exactitud
    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario ha pedido el TEXTO EXACTO. Debes copiar el fragmento del CAA lo más literal posible, sin resumir. Asegúrate de indicar el capítulo y artículo correspondientes antes del texto.`;
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