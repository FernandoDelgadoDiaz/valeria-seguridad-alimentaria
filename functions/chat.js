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
    const parsed = JSON.parse(event.body || "{}");
    const query = (parsed.query || "").slice(0, 2000);
    const history = (Array.isArray(parsed.history) ? parsed.history : []).slice(-20);
    const mode = parsed.mode || "tecnico";
    const incomingTestState = parsed.testState || null;

    if (!CACHE_DATA) {
      try {
        CACHE_DATA = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
      } catch (e) {
        console.error("Error cargando embeddings.json:", e.message);
        return json({ ok: false, error: "Base de conocimiento no disponible. Intentá más tarde." });
      }
      if (!CACHE_DATA?.chunks?.length) {
        CACHE_DATA = null;
        return json({ ok: false, error: "Base de conocimiento vacía. Contactá al administrador." });
      }
    }

    const mencionaCAA = /caa|código alimentario|art[ií]culo|cap[ií]tulo/i.test(query);
    const pideExacto = /texto exacto|literal|textualmente|dame el art[ií]culo|copia el art[ií]culo/i.test(query);

    const lastMsg = history.length > 0 ? history[history.length - 1].content : "";
    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: `${query} ${lastMsg}`.slice(0, 1000),
    });

    const allChunks = CACHE_DATA.chunks.map(c => ({
      ...c,
      score: cosSim(qEmb.data[0].embedding, c.embedding || c.vec)
    }));

    const sorted = allChunks.sort((a, b) => b.score - a.score);
    const internos = sorted.filter(c =>
      !c.source.toLowerCase().includes("capitulo") &&
      !c.source.toLowerCase().includes("caa")
    );
    const caaChunks = sorted.filter(c =>
      c.source.toLowerCase().includes("capitulo") ||
      c.source.toLowerCase().includes("caa")
    );

    const topInternos = internos.slice(0, 4);
    const topCAA = caaChunks.slice(0, 4);

    let contextChunks = [];
    if (pideExacto) {
      contextChunks = topCAA.length > 0 ? topCAA : topInternos;
    } else if (mencionaCAA) {
      contextChunks = [...topInternos, ...topCAA];
    } else {
      const internosRelevantes = topInternos.filter(c => c.score > 0.3);
      if (internosRelevantes.length >= 2) {
        contextChunks = topInternos;
      } else {
        contextChunks = [...topInternos, ...topCAA.slice(0, 4 - topInternos.length)];
      }
    }
    if (contextChunks.length === 0) contextChunks = topCAA;

    const contextText = contextChunks.map(c => c.text).join("\n\n---\n\n");

    // Test interactivo en curso
    if (mode === "ensena" && incomingTestState) {
      const testState = incomingTestState;
      const preguntaActual = testState.preguntaActual;
      const respuestasUsuario = testState.respuestasUsuario || {};

      if (query && preguntaActual >= 2) {
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
        let mensajeResultado = `**Resultados del test**\n\nAcertaste ${aciertos} de ${testState.preguntas.length}.\n\n`;
        detalles.forEach(d => {
          if (!d.esCorrecta) {
            mensajeResultado += `Pregunta ${d.pregunta}: respondiste "${d.usuario || '(sin respuesta)'}", la correcta era "${d.correcta}".\n`;
          }
        });
        mensajeResultado += "\n¿Querés hacer otro test sobre el mismo tema o preferís cambiar de tema?";
        return json({
          ok: true, answer: mensajeResultado, testState: null,
          history: [...history, { role: "assistant", content: mensajeResultado }]
        });
      }
    }

    // Generar test
    if (mode === "ensena" && (query.trim() === "2" || query.toLowerCase().includes("test"))) {
      const lastAssistantMsg = history.filter(m => m.role === "assistant").pop();
      const lastExplanation = lastAssistantMsg ? lastAssistantMsg.content : "";
      let testData = null;
      let attempts = 0;

      while (attempts < 3) {
        attempts++;
        const testGenPrompt = `Basado en el siguiente contexto y en la explicación reciente, genera un test de 5 preguntas de opción múltiple **exclusivamente sobre el mismo tema tratado en la explicación**. Cada pregunta con 3 opciones (a, b, c) diferenciadas. Devuelve solo JSON con: "preguntas" (array de strings, formato "Pregunta? a) ... b) ... c) ...") y "respuestasCorrectas" (objeto con claves "1" a "5", valores "a","b","c"). Sin texto adicional.

Contexto:
${contextText}

Explicación del asistente (tema a evaluar):
${lastExplanation}`;

        try {
          const res = await client.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
              { role: "system", content: "Generador de tests. Devuelve solo JSON válido." },
              { role: "user", content: testGenPrompt }
            ],
            temperature: 0.3 + (attempts * 0.1),
          });
          const raw = res.choices[0].message.content;
          let parsedTest;
          try { parsedTest = JSON.parse(raw); }
          catch (e) {
            const m = raw.match(/\{.*\}/s);
            if (m) parsedTest = JSON.parse(m[0]);
            else throw new Error("JSON inválido");
          }
          if (validateTest(parsedTest)) { testData = parsedTest; break; }
          else console.log(`Intento ${attempts}: test inválido, reintentando...`);
        } catch (err) {
          console.log(`Error intento ${attempts}:`, err.message);
        }
      }

      if (!testData) {
        return json({ ok: true, answer: "No pude generar un test válido. ¿Podés intentarlo de nuevo?", testState: null, history });
      }

      // preguntaActual = 2: la pregunta 1 ya se muestra en este response
      const testState = {
        preguntas: testData.preguntas,
        respuestasCorrectas: testData.respuestasCorrectas,
        preguntaActual: 2,
        respuestasUsuario: {}
      };
      const mensaje = `**Pregunta 1 de 5:**\n${testData.preguntas[0]}\n\nResponde con la letra de la opción (ej: a)`;
      return json({
        ok: true, answer: mensaje, testState,
        history: [...history, { role: "assistant", content: mensaje }]
      });
    }

    // Respuesta normal
    let systemPrompt = "";

    if (mode === "tecnico") {
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tenés acceso a documentos internos (procedimientos, manuales, instructivos) y al Código Alimentario Argentino (CAA).

Modo actual: **TÉCNICO** — Respuestas concisas, directas y técnicas.

**JERARQUÍA DE FUENTES (seguir estrictamente):**
1. Los documentos internos son tu PRIMERA fuente. Respondé desde ahí cuando estén disponibles en el CONTEXTO.
2. Incorporás el CAA solo cuando: el usuario lo pide explícitamente, o los internos no tienen la respuesta.
3. No menciones la fuente interna. Si citás el CAA, indicá capítulo y artículo.
4. Al terminar respuestas basadas en internos, podés ofrecer: "¿Necesitás que consulte también el CAA para ampliar?"

**SEGUIMIENTO DE CONVERSACIÓN:**
- Leé el historial completo antes de responder. No repitas información ya dada.
- Si el usuario valida o comenta algo sobre lo que dijiste, reconocelo y avanzá desde ahí.
- Si pide aclaraciones, profundizá en el mismo tema sin resetear la conversación.
- Si el usuario dice "eso es lo que dice el procedimiento" u otra afirmación sobre el contexto, procesala y respondé en consecuencia.

**RESTRICCIÓN:** Solo respondés sobre seguridad alimentaria, BPM y CAA.

CONTEXTO:
${contextText}`;
    } else {
      systemPrompt = `Eres INOCUO, un asistente experto en seguridad alimentaria y Buenas Prácticas de Manufactura (BPM). Tenés acceso a documentos internos y al Código Alimentario Argentino (CAA).

Modo actual: **ENSEÑA** — Respuestas didácticas, estructuradas y pedagógicas.

**Estructura de tus respuestas:**
- Primero, definición clara del concepto.
- Luego, clasificación o tipos si corresponde.
- Ejemplos prácticos tomados de los documentos.
- Al final, siempre: "**Para profundizar en este tema, respondé '1'. Para hacer un test de aprendizaje, respondé '2'.**"

**Importante:**
- Los documentos internos tienen prioridad como fuente.
- Si responde '1', profundizá. Si responde '2', el sistema genera el test.

**RESTRICCIÓN:** Solo respondés sobre seguridad alimentaria, BPM y CAA.

CONTEXTO:
${contextText}`;
    }

    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario pidió el TEXTO EXACTO del CAA. Copialo lo más literal posible, sin resumir. Indicá capítulo y artículo antes del texto.`;
    }

    const esContinuacion = query.trim().length <= 3 ||
      /^(si|sí|no|ok|yes|dale|bueno|claro|1|2|a|b|c|gracias|entendido|correcto)$/i.test(query.trim());

    if (!esContinuacion) {
      const guardCheck = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: `Clasificador para asistente de seguridad alimentaria. Responde SOLO "SI" o "NO".
SI si está relacionado con: seguridad alimentaria, BPM, higiene, conservación, contaminación, CAA, normativas, etiquetado, procesos o ingredientes alimentarios.
NO solo si es claramente ajeno: deportes, geografía, entretenimiento, matemáticas.
Ante la duda: "SI".`
          },
          { role: "user", content: query }
        ],
        temperature: 0,
        max_tokens: 5,
      });

      const esRelevante = guardCheck.choices[0].message.content.trim().toUpperCase().startsWith("SI");
      if (!esRelevante) {
        const rechazo = "Soy INOCUO, especializado en seguridad alimentaria y BPM. Esta consulta está fuera de mi área. Si tenés dudas sobre inocuidad, normativas del CAA o manipulación de alimentos, ¡con gusto te ayudo!";
        return json({
          ok: true, answer: rechazo, testState: null,
          history: [...history, { role: "user", content: query }, { role: "assistant", content: rechazo }]
        });
      }
    }

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        ...history.slice(-8),
        { role: "user", content: query }
      ],
      temperature: pideExacto ? 0.1 : 0.3,
    });

    const answer = completion.choices[0].message.content;
    return json({
      ok: true, answer, testState: null,
      history: [...history, { role: "user", content: query }, { role: "assistant", content: answer }]
    });

  } catch (err) {
    console.error("Error en handler:", err);
    return json({ ok: false, error: "Error interno. Intentá de nuevo." });
  }
}
