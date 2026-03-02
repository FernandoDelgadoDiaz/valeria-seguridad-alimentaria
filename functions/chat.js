import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_FILE = path.resolve("/var/task/data/embeddings.json");

// Funciones matemáticas para búsqueda semántica
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

    // Detectar intenciones
    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);

    // Buscar fragmentos relevantes (igual que antes)
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
    const internos = sorted.filter(c => !c.source.toLowerCase().includes("capitulo") && !c.source.toLowerCase().includes("caa"));
    const caaChunks = sorted.filter(c => c.source.toLowerCase().includes("capitulo") || c.source.toLowerCase().includes("caa"));

    const topInternos = internos.slice(0, 4);
    const topCAA = caaChunks.slice(0, 4);

    let contextChunks = [];
    if (pideExacto) {
      contextChunks = topCAA.length > 0 ? topCAA : topInternos;
    } else if (mencionaCAA) {
      contextChunks = [...topInternos, ...topCAA].slice(0, 8);
    } else {
      contextChunks = topInternos;
    }
    if (contextChunks.length === 0) contextChunks = topCAA;

    const contextText = contextChunks.map(c => c.text).join("\n\n---\n\n");

    // --- Lógica especial para tests interactivos en modo Enseña ---
    let testState = null;
    const lastAssistantMsg = history.filter(m => m.role === "assistant").pop();

    // Buscar si el último mensaje del asistente contiene un marcador de test
    if (lastAssistantMsg && lastAssistantMsg.content.includes("__TEST_STATE__")) {
      try {
        const match = lastAssistantMsg.content.match(/__TEST_STATE__(\{.*?\})/);
        if (match) {
          testState = JSON.parse(match[1]);
        }
      } catch (e) {}
    }

    // Si estamos en modo Enseña y hay un test en curso, procesamos la respuesta
    if (mode === "ensena" && testState) {
      const userAnswer = query.trim().toLowerCase();
      // Guardar respuesta del usuario
      testState.respuestasUsuario[testState.preguntaActual] = userAnswer;

      if (testState.preguntaActual < 5) {
        // Pasar a la siguiente pregunta
        testState.preguntaActual++;
        const siguientePregunta = testState.preguntas[testState.preguntaActual - 1]; // -1 porque el array empieza en 0
        const mensaje = `**Pregunta ${testState.preguntaActual} de 5:**\n${siguientePregunta}\n\nResponde con la letra de la opción (ej: a)`;
        // Incluir el estado actualizado en el mensaje (oculto para el usuario)
        const mensajeConEstado = mensaje + `\n\n__TEST_STATE__${JSON.stringify(testState)}`;
        return json({
          ok: true,
          answer: mensajeConEstado,
          history: [...history, { role: "user", content: query }, { role: "assistant", content: mensajeConEstado }]
        });
      } else {
        // Terminó el test, calcular puntaje
        let aciertos = 0;
        const respuestasCorrectas = testState.respuestasCorrectas;
        const respuestasUsuario = testState.respuestasUsuario;
        const resultados = [];

        for (let i = 1; i <= 5; i++) {
          const correcta = respuestasCorrectas[i];
          const usuario = respuestasUsuario[i] || '';
          const esCorrecta = (usuario === correcta);
          if (esCorrecta) aciertos++;
          resultados.push({
            pregunta: i,
            usuario: usuario,
            correcta: correcta,
            esCorrecta: esCorrecta
          });
        }

        // Construir mensaje de resultados
        let mensajeResultado = `**Resultados del test**\n\nHas acertado ${aciertos} de 5.\n\n`;
        resultados.forEach(r => {
          if (!r.esCorrecta) {
            mensajeResultado += `❌ Pregunta ${r.pregunta}: tu respuesta fue "${r.usuario}", la correcta es "${r.correcta}".\n`;
          }
        });
        mensajeResultado += "\n¿Quieres hacer otro test sobre el mismo tema o prefieres cambiar de tema?";

        return json({
          ok: true,
          answer: mensajeResultado,
          history: [...history, { role: "user", content: query }, { role: "assistant", content: mensajeResultado }]
        });
      }
    }

    // --- Prompt normal (sin test en curso) ---
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
    } else {
      // Modo ENSEÑA
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **ENSEÑA** – Tus respuestas deben ser **didácticas, estructuradas y pedagógicas**.

**Estructura de tus respuestas:**
- Primero, da una definición clara del concepto.
- Luego, si corresponde, ofrece una clasificación o tipos.
- Incluye ejemplos prácticos (usa los documentos disponibles).
- Al final de tu explicación, añade siempre:
  "**Para profundizar en este tema, responde '1'. Para hacer un test de aprendizaje, responde '2'.**"

**Manejo de las opciones:**
- Si el usuario responde '1', proporciona información adicional, más detalles o ejemplos avanzados.
- Si el usuario responde '2', debes iniciar un test interactivo de 5 preguntas. Para ello, genera las 5 preguntas con sus opciones (a, b, c) y guárdalas internamente. Luego muestra solo la primera pregunta y espera la respuesta. Para indicar que se inicia un test, debes incluir en tu mensaje un marcador especial con el siguiente formato exacto (sin espacios adicionales):
  "__TEST_STATE__{\"preguntaActual\":1,\"preguntas\":[\"Pregunta 1...\",\"Pregunta 2...\",...],\"respuestasCorrectas\":{\"1\":\"a\",\"2\":\"b\",...},\"respuestasUsuario\":{}}"
  Este marcador debe ir al final del mensaje, después de la primera pregunta. No debe ser visible para el usuario, pero el sistema lo usará para continuar el test.

**Importante:**
- Cuando el usuario responda a una pregunta (ej: "a"), el sistema automáticamente procesará la respuesta y mostrará la siguiente pregunta o el resultado final.
- Al final, se mostrará el puntaje y las respuestas incorrectas con las correctas.

CONTEXTO:
${contextText}`;
    }

    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario ha pedido el TEXTO EXACTO. Debes copiar el fragmento del CAA lo más literal posible, sin resumir. Indica el capítulo y artículo correspondientes antes del texto.`;
    }

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