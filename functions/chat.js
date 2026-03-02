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

    // Detectar intenciones del usuario
    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);
    const modoAprendizaje = /^(defin[ei]|qu[eé] es|explica|concepto|significa|aprender|[/]aprender)/i.test(query.trim());

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
    } else if (mencionaCAA || modoAprendizaje) {
      // En modo aprendizaje combinamos, pero si hay CAA lo incluimos
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

    // Prompt base
    let systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a:

- Documentos internos (manuales, BPM, procedimientos)
- Código Alimentario Argentino (CAA)

Debes seguir esta jerarquía:

1. Si el usuario pide EXPLÍCITAMENTE un artículo o capítulo del CAA (ej: "dame el artículo 5 del capítulo 1"), responde con el TEXTO EXACTO del fragmento del CAA, citando la fuente de manera natural: extrae del propio texto el capítulo y artículo (por ejemplo, "Según el Capítulo I, Artículo 2 del CAA: ..."). No uses el nombre del archivo.

2. Si el usuario menciona "CAA", "código alimentario", "artículo" o "capítulo" pero no pide el texto exacto, combina información de documentos internos y del CAA para dar una respuesta completa. Para el CAA, cita siempre el capítulo y artículo correspondientes (extrayéndolos del texto). Para los internos, no menciones la fuente.

3. Si el usuario activa el MODO APRENDIZAJE (preguntas como "qué es", "explica", "definición", o el comando "/aprender"), debes estructurar tu respuesta de forma pedagógica:
   - Definición clara del concepto.
   - Clasificación o tipos si corresponde.
   - Ejemplos prácticos (usa los documentos internos o el CAA según corresponda).
   - Al final, ofrece continuar: "¿Quieres que profundice en algún aspecto en particular?"

4. En caso contrario (consulta general sin mención a CAA ni modo aprendizaje), responde principalmente con los documentos internos. Si con ellos es suficiente, no agregues CAA. Al final de la respuesta, puedes preguntar: "¿Necesitas que consulte también el Código Alimentario Argentino para ampliar?"

5. Siempre que uses información del CAA, debes indicar el capítulo y artículo (por ejemplo, "Capítulo I, Artículo 2") basándote en el texto del fragmento. No uses el nombre del archivo.

6. Si la pregunta no es sobre seguridad alimentaria, responde amablemente que solo puedes ayudar con temas del CAA y BPM.

CONTEXTO ACTUAL:
${contextText}`;

    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario ha pedido el TEXTO EXACTO. Debes copiar el fragmento del CAA lo más literal posible, sin resumir. Asegúrate de indicar el capítulo y artículo correspondientes antes del texto.`;
    }

    if (modoAprendizaje) {
      systemPrompt += `\n\nIMPORTANTE: Estás en MODO APRENDIZAJE. Tu respuesta debe ser didáctica, con definiciones, ejemplos y una invitación a profundizar.`;
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