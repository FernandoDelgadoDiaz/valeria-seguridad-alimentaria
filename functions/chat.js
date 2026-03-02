import fs from "fs";
import path from "path";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_FILE = path.resolve("/var/task/data/embeddings.json");

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
    const { query, history = [], mode = "tecnico", testState: incomingTestState } = JSON.parse(event.body || "{}");

    if (!CACHE_DATA) {
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    // Detectar intenciones
    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);

    // Búsqueda semántica
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

    // --- Lógica para test interactivo (modo Enseña) ---
    if (mode === "ensena" && incomingTestState) {
      // Estamos en medio de un test
      const testState = incomingTestState;
      const preguntaActual = testState.preguntaActual;
      const respuestasUsuario = testState.respuestasUsuario || {};

      // Guardar respuesta del usuario (si viene de una pregunta anterior)
      if (query && preguntaActual > 1) {
        respuestasUsuario[preguntaActual - 1] = query.trim().toLowerCase();
      }

      if (preguntaActual <= testState.preguntas.length) {
        // Enviar siguiente pregunta
        const pregunta = testState.preguntas[preguntaActual - 1];
        const mensaje = `**Pregunta ${preguntaActual} de ${testState.preguntas.length}:**\n${pregunta}\n\nResponde con la letra de la opción (ej: a)`;
        return json({
          ok: true,
          answer: mensaje,
          testState: {
            ...testState,
            preguntaActual: preguntaActual + 1,
            respuestasUsuario
          },
          history: [...history, { role: "assistant", content: mensaje }]
        });
      } else {
        // Test finalizado, calcular puntaje
        const correctas = testState.respuestasCorrectas;
        let aciertos = 0;
        const detalles = [];

        for (let i = 1; i <= testState.preguntas.length; i++) {
          const userAns = respuestasUsuario[i] || "";
          const correctAns = correctas[i];
          const esCorrecta = (userAns === correctAns);
          if (esCorrecta) aciertos++;
          detalles.push({ pregunta: i, usuario: userAns, correcta: correctAns, esCorrecta });
        }

        let mensajeResultado = `**Resultados del test**\n\nHas acertado ${aciertos} de ${testState.preguntas.length}.\n\n`;
        detalles.forEach(d => {
          if (!d.esCorrecta) {
            mensajeResultado += `❌ Pregunta ${d.pregunta}: tu respuesta fue "${d.usuario}", la correcta es "${d.correcta}".\n`;
          }
        });
        mensajeResultado += "\n¿Quieres hacer otro test sobre el mismo tema o prefieres cambiar de tema?";

        return json({
          ok: true,
          answer: mensajeResultado,
          testState: null, // Finaliza el test
          history: [...history, { role: "assistant", content: mensajeResultado }]
        });
      }
    }

    // --- Si estamos en modo enseña y el usuario quiere un test (responde '2') ---
    if (mode === "ensena" && (query.trim() === "2" || query.toLowerCase().includes("test"))) {
      // Generar test con IA
      const testGenPrompt = `Basado en el siguiente contexto, genera un test de 5 preguntas de opción múltiple sobre el tema tratado. Cada pregunta debe tener 3 opciones (a, b, c). Devuelve exclusivamente un objeto JSON con dos campos: "preguntas" (un array de strings, cada uno con la pregunta y las opciones en formato "Pregunta? a) ... b) ... c) ...") y "respuestasCorrectas" (un objeto con claves "1","2","3","4","5" y valores "a","b","c" correspondientes a la opción correcta). No incluyas texto adicional, solo el JSON.

Contexto:
${contextText}`;

      try {
        const testGenCompletion = await client.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: "Eres un generador de tests. Devuelve solo JSON." },
            { role: "user", content: testGenPrompt }
          ],
          temperature: 0.3,
        });

        const testGenAnswer = testGenCompletion.choices[0].message.content;
        let testData;
        try {
          testData = JSON.parse(testGenAnswer);
        } catch (e) {
          const match = testGenAnswer.match(/\{.*\}/s);
          if (match) testData = JSON.parse(match[0]);
          else throw new Error("No se pudo generar el test");
        }

        if (!testData.preguntas || !testData.respuestasCorrectas || testData.preguntas.length !== 5) {
          throw new Error("Formato de test inválido");
        }

        const testState = {
          preguntas: testData.preguntas,
          respuestasCorrectas: testData.respuestasCorrectas,
          preguntaActual: 1,
          respuestasUsuario: {}
        };

        const primeraPregunta = testData.preguntas[0];
        const mensaje = `**Pregunta 1 de 5:**\n${primeraPregunta}\n\nResponde con la letra de la opción (ej: a)`;
        return json({
          ok: true,
          answer: mensaje,
          testState: testState,
          history: [...history, { role: "assistant", content: mensaje }]
        });

      } catch (err) {
        console.error("Error generando test:", err);
        return json({
          ok: true,
          answer: "Lo siento, tuve un problema al generar el test. ¿Puedes intentarlo de nuevo?",
          testState: null,
          history: [...history, { role: "assistant", content: "Lo siento, tuve un problema al generar el test. ¿Puedes intentarlo de nuevo?" }]
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
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **ENSEÑA** – Tus respuestas deben ser **didácticas, estructuradas y pedagógicas**.

**Estructura de tus respuestas:**
- Primero, da una definición clara del concepto.
- Luego, si corresponde, ofrece una clasificación o tipos.
- Incluye ejemplos prácticos (usa los documentos disponibles).
- Al final de tu explicación, añade siempre:
  "**Para profundizar en este tema, responde '1'. Para hacer un test de aprendizaje, responde '2'.**"

**Importante:**
- Si el usuario responde '1', proporciona información adicional.
- Si el usuario responde '2', el sistema se encargará de generar el test. Tú no debes generar el test directamente, solo responder con la explicación y las opciones.
- Siempre utiliza el CONTEXTO para basar tus explicaciones.

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
      testState: null, // Por defecto no hay test en curso
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    console.error("Error en handler:", err);
    return json({ error: err.message });
  }
}