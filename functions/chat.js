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

function validateTest(testData) {
  if (!testData || !testData.preguntas || !testData.respuestasCorrectas) return false;
  if (testData.preguntas.length !== 5) return false;
  const preguntasNormalizadas = testData.preguntas.map(p => p.replace(/\s+/g, ' ').trim().toLowerCase());
  const unique = new Set(preguntasNormalizadas);
  if (unique.size !== 5) return false;
  for (const p of testData.preguntas) {
    if (!p.includes('a)') || !p.includes('b)') || !p.includes('c)')) return false;
  }
  const correctas = testData.respuestasCorrectas;
  for (let i = 1; i <= 5; i++) {
    if (!correctas[i] || !['a','b','c'].includes(correctas[i])) return false;
  }
  return true;
}

export async function handler(event) {
  if (event.httpMethod !== "POST") return { statusCode: 405 };

  try {
    const { query, history = [], mode = "tecnico", testState: incomingTestState } = JSON.parse(event.body || "{}");

    if (!CACHE_DATA) {
      CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    }

    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);

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
      contextChunks = topInternos; // Prioridad internos
    }
    if (contextChunks.length === 0) contextChunks = topCAA;

    const contextText = contextChunks.map(c => c.text).join("\n\n---\n\n");

    // --- Lógica para test interactivo (modo Enseña) ---
    if (mode === "ensena" && incomingTestState) {
      const testState = incomingTestState;
      const preguntaActual = testState.preguntaActual;
      const respuestasUsuario = testState.respuestasUsuario || {};

      // Guardar respuesta actual (si corresponde)
      if (query && preguntaActual > 1) {
        respuestasUsuario[preguntaActual - 1] = query.trim().toLowerCase();
      }

      if (preguntaActual <= testState.preguntas.length) {
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
        // Test finalizado
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
          testState: null,
          history: [...history, { role: "assistant", content: mensajeResultado }]
        });
      }
    }

    // --- Iniciar test (respuesta '2') ---
    if (mode === "ensena" && (query.trim() === "2" || query.toLowerCase().includes("test"))) {
      const lastAssistantMsg = history.filter(m => m.role === "assistant").pop();
      const lastExplanation = lastAssistantMsg ? lastAssistantMsg.content : "";

      let testData = null;
      let attempts = 0;
      const maxAttempts = 3;

      while (attempts < maxAttempts) {
        attempts++;
        const testGenPrompt = `Basado en el siguiente contexto y en la explicación reciente, genera un test de 5 preguntas de opción múltiple **exclusivamente sobre el mismo tema tratado en la explicación**. No incluyas preguntas de otros temas aunque aparezcan en el contexto. Cada pregunta debe tener 3 opciones (a, b, c) claramente diferenciadas y no repetidas entre preguntas. Devuelve exclusivamente un objeto JSON con dos campos: "preguntas" (un array de strings, cada uno con la pregunta y las opciones en formato "Pregunta? a) ... b) ... c) ...") y "respuestasCorrectas" (un objeto con claves "1","2","3","4","5" y valores "a","b","c" correspondientes a la opción correcta). No incluyas texto adicional, solo el JSON.

Contexto general:
${contextText}

Explicación reciente del asistente (tema a evaluar):
${lastExplanation}`;

        try {
          const testGenCompletion = await client.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
              { role: "system", content: "Eres un generador de tests. Devuelve solo JSON." },
              { role: "user", content: testGenPrompt }
            ],
            temperature: 0.3 + (attempts * 0.1),
          });

          const testGenAnswer = testGenCompletion.choices[0].message.content;
          let parsed;
          try {
            parsed = JSON.parse(testGenAnswer);
          } catch (e) {
            const match = testGenAnswer.match(/\{.*\}/s);
            if (match) parsed = JSON.parse(match[0]);
            else throw new Error("No se pudo parsear JSON");
          }

          if (validateTest(parsed)) {
            testData = parsed;
            break;
          }
        } catch (err) {
          console.log(`Error en intento ${attempts}:`, err.message);
        }
      }

      if (!testData) {
        return json({
          ok: true,
          answer: "Lo siento, no pude generar un test válido en este momento. ¿Puedes intentarlo de nuevo más tarde?",
          testState: null,
          history: [...history, { role: "assistant", content: "Lo siento, no pude generar un test válido en este momento. ¿Puedes intentarlo de nuevo más tarde?" }]
        });
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
    }

    // --- Prompt normal (sin test) ---
    let systemPrompt = "";

    if (mode === "tecnico") {
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **TÉCNICO** – Tus respuestas deben ser **concisas, directas y técnicas**. 

**Reglas estrictas:**
1. **Prioridad absoluta**: Siempre debes basar tu respuesta en los documentos internos (manuales, BPM) que se te proporcionan en el CONTEXTO. Si hay información relevante en los internos, úsala primero.
2. Si el usuario menciona "CAA", "código", "artículo" o "capítulo", puedes complementar con información del CAA.
3. Si el usuario pide un artículo exacto, dale el texto literal con la cita.
4. Si no hay información en internos, puedes usar el CAA.
5. **Mantén el hilo de la conversación**: Ten en cuenta los mensajes anteriores del historial para responder coherentemente.
6. Al final, si solo usaste internos, ofrece: "¿Necesitas que consulte también el CAA para ampliar?"
7. **Restricción de dominio**: Solo responde sobre seguridad alimentaria, BPM o CAA. Si la pregunta es fuera de tema, recházala amablemente.

CONTEXTO:
${contextText}

Historial reciente (para mantener coherencia): ${history.slice(-3).map(m => m.content).join(' | ')}`;
    } else {
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tienes acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **ENSEÑA** – Tus respuestas deben ser **didácticas, estructuradas y pedagógicas**.

**Estructura:**
- Definición clara del concepto.
- Clasificación o tipos si corresponde.
- Ejemplos prácticos (usa los documentos).
- Al final, añade: "**Para profundizar, responde '1'. Para hacer un test, responde '2'.**"

**Importante:**
- Siempre usa el CONTEXTO para basar tus explicaciones.
- Mantén el hilo de la conversación.
- Restricción de dominio: solo temas de seguridad alimentaria.

CONTEXTO:
${contextText}

Historial reciente: ${history.slice(-3).map(m => m.content).join(' | ')}`;
    }

    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario pide texto exacto. Dale el fragmento del CAA literal con cita.`;
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
      testState: null,
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    console.error("Error:", err);
    return json({ error: err.message });
  }
}