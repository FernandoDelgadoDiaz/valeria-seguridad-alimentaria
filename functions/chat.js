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
  const norm2 = testData.preguntas.map(p => p.replace(/\s+/g, ' ').trim().toLowerCase());
  if (new Set(norm2).size !== 5) return false;
  for (const p of testData.preguntas) {
    if (!p.includes('a)') || !p.includes('b)') || !p.includes('c)')) return false;
  }
  for (let i = 1; i <= 5; i++) {
    if (!testData.respuestasCorrectas[i] || !['a','b','c'].includes(testData.respuestasCorrectas[i])) return false;
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

    const q = query.toLowerCase();

    // FIX: detectar pedidos de texto exacto del CAA — regex ampliado
    // Antes solo matcheaba "texto exacto", "literal", "dame el artículo"
    // Ahora también matchea "textual", "artículo X del capítulo Y", etc.
    const pideExacto =
      /texto exacto|textual|literal|textualmente/i.test(query) ||
      /dame el art[ií]culo|copia el art[ií]culo|transcrib/i.test(query) ||
      (/art[ií]culo\s*\d+/.test(q) && /cap[ií]tulo\s*\d+/.test(q));

    const mencionaCAA = /caa|c[oó]digo alimentario|art[ií]culo|cap[ií]tulo/i.test(query);

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

    // Jerarquía: documentos internos siempre primero
    let contextChunks = [];
    if (pideExacto) {
      contextChunks = topCAA.length > 0 ? topCAA : topInternos;
    } else if (mencionaCAA) {
      contextChunks = [...topInternos, ...topCAA];
    } else {
      const relevantes = topInternos.filter(c => c.score > 0.3);
      if (relevantes.length >= 2) {
        contextChunks = topInternos;
      } else {
        contextChunks = [...topInternos, ...topCAA.slice(0, 4 - topInternos.length)];
      }
    }
    if (contextChunks.length === 0) contextChunks = topCAA;

    const contextText = contextChunks.map(c => c.text).join("\n\n---\n\n");

    // ── Test interactivo en curso ──
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
          testState: { ...testState, preguntaActual: preguntaActual + 1, respuestasUsuario },
          history: [...history, { role: "assistant", content: mensaje }]
        });
      } else {
        const correctas = testState.respuestasCorrectas;
        let aciertos = 0;
        const detalles = [];
        for (let i = 1; i <= testState.preguntas.length; i++) {
          const userAns = respuestasUsuario[i] || "";
          const correctAns = correctas[i];
          const ok2 = userAns === correctAns;
          if (ok2) aciertos++;
          detalles.push({ pregunta: i, usuario: userAns, correcta: correctAns, esCorrecta: ok2 });
        }
        let resultado = `**Resultados del test**\n\nAcertaste ${aciertos} de ${testState.preguntas.length}.\n\n`;
        detalles.forEach(d => {
          if (!d.esCorrecta) {
            resultado += `❌ Pregunta ${d.pregunta}: respondiste "${d.usuario || '(sin respuesta)'}", la correcta era "${d.correcta}".\n`;
          }
        });
        resultado += "\n¿Querés hacer otro test o cambiar de tema?";
        return json({
          ok: true, answer: resultado, testState: null,
          history: [...history, { role: "assistant", content: resultado }]
        });
      }
    }

    // ── Generar test ──
    if (mode === "ensena" && (query.trim() === "2" || query.toLowerCase().includes("test"))) {
      const lastAI = history.filter(m => m.role === "assistant").pop();
      const lastExplanation = lastAI ? lastAI.content : "";
      let testData = null;

      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          const res = await client.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
              { role: "system", content: "Generador de tests. Devuelve solo JSON válido sin markdown." },
              { role: "user", content: `Generá un test de 5 preguntas de opción múltiple sobre el tema de la siguiente explicación, pensado para evaluar a profesionales del sector alimentario.

REGLAS PARA LAS PREGUNTAS:
- Preguntá sobre conceptos específicos, no definiciones obvias.
- Las 3 opciones (a, b, c) deben ser plausibles — evitá opciones absurdas o claramente incorrectas.
- Al menos 2 preguntas deben requerir razonamiento, no solo memoria.
- Variá el tipo: algunas sobre "qué hacer", otras sobre "por qué", otras sobre valores/límites concretos.

Devolvé solo JSON con: "preguntas" (array de strings, formato "Pregunta? a) ... b) ... c) ...") y "respuestasCorrectas" (objeto con claves "1" a "5", valores "a","b","c"). Sin markdown.

Contexto:
${contextText}

Explicación:
${lastExplanation}` }
            ],
            temperature: 0.3 + (attempt * 0.1),
          });
          const raw = res.choices[0].message.content;
          let parsed2;
          try { parsed2 = JSON.parse(raw); }
          catch { const m = raw.match(/\{[\s\S]*\}/); if (m) parsed2 = JSON.parse(m[0]); else throw new Error("no JSON"); }
          if (validateTest(parsed2)) { testData = parsed2; break; }
        } catch (e) { console.log(`Test attempt ${attempt}:`, e.message); }
      }

      if (!testData) {
        return json({ ok: true, answer: "No pude generar el test. Intentá de nuevo.", testState: null, history });
      }

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

    // ── Respuesta normal ──
    let systemPrompt = "";
    if (mode === "tecnico") {
      systemPrompt = `Eres INOCUO, un experto en seguridad alimentaria, BPM y normativas del CAA. Respondés como un especialista experimentado que habla directo, sin rodeos.

FORMATO — MUY IMPORTANTE:
- Respondé en texto corrido, sin headers (###), sin listas con guiones ni bullets.
- Si necesitás enumerar, usá números dentro del párrafo: "Los requisitos son: 1) ... 2) ... 3) ..."
- Máximo 4 párrafos cortos. En mobile se lee mejor así.
- Tono: técnico pero conversacional, como un colega experto.

FUENTES:
- Si la info viene de tus documentos internos: respondé directo, sin mencionar la fuente.
- Si la info viene del CAA: terminá con → *Fuente: CAA, Cap. [X], Art. [Y]*
- Si no encontrás el dato exacto en el CONTEXTO: decilo claramente, no inventes.
- Podés ofrecer: "¿Querés que busque también en el CAA?"

FUERA DE DOMINIO: Si la pregunta no es de seguridad alimentaria, BPM o CAA, respondé: "Soy INOCUO, especializado en seguridad alimentaria y BPM. Esta consulta está fuera de mi área. Si tenés dudas sobre inocuidad, normativas del CAA o manipulación de alimentos, ¡con gusto te ayudo!"

SEGUIMIENTO: Leé el historial. No repitas lo ya dicho. Si el usuario confirma algo, avanzá.

CONTEXTO:
${contextText}`;
    } else {
      systemPrompt = `Eres INOCUO, experto en seguridad alimentaria y BPM. En Modo Enseña explicás como un buen docente: claro, progresivo y con ejemplos reales de la industria.

ESTRUCTURA DE RESPUESTA:
1. Definición simple en 2-3 oraciones.
2. Desarrollo en párrafos cortos (no listas con guiones). Máximo 3 párrafos.
3. 1 o 2 ejemplos concretos de la industria alimentaria argentina.
4. Siempre al final: "**Para profundizar respondé '1'. Para hacer un test respondé '2'.**"

FORMATO:
- Podés usar negritas para conceptos clave.
- Evitá listas largas con guiones. Preferí párrafos fluidos.
- Tono: didáctico pero no infantil. Como un capacitador experimentado.

FUENTES:
- Info de documentos internos: respondé sin mencionar la fuente.
- Info del CAA: citá al final → *Fuente: CAA, Cap. [X], Art. [Y]*

FUERA DE DOMINIO: Si la pregunta no es de seguridad alimentaria o CAA, rechazala: "Soy INOCUO, especializado en seguridad alimentaria y BPM. Esta consulta está fuera de mi área."

RESTRICCIÓN: Solo seguridad alimentaria, BPM y CAA.

CONTEXTO:
${contextText}`;
    }

    if (pideExacto) {
      systemPrompt += `\n\nIMPORTANTE: El usuario pidió el TEXTO EXACTO o TEXTUAL del CAA. Buscá en el CONTEXTO el artículo mencionado y copialo de forma literal, sin resumir ni parafrasear. Indicá capítulo y artículo antes del texto. Si el fragmento exacto no está en el CONTEXTO disponible, indicalo claramente.`;
    }

    // Guardia de dominio
    // FIX: bypass ampliado — cualquier mensaje que mencione artículo+capítulo es claramente del dominio
    const esContinuacion =
      query.trim().length <= 3 ||
      history.length > 2 ||  // FIX: con historial activo es continuacion de sesion
      /^(si|sí|no|ok|yes|dale|bueno|claro|1|2|a|b|c|gracias|entendido|correcto|otro tema|mismo tema|otro|cambiar|continuar|seguir)$/i.test(query.trim()) ||
      pideExacto ||
      mencionaCAA;

    if (!esContinuacion) {
      const guardCheck = await client.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: `Clasificador para asistente de seguridad alimentaria. Responde SOLO "SI" o "NO".
SI si está relacionado con: seguridad alimentaria, BPM, higiene, conservación, contaminación, CAA, normativas, etiquetado, procesos, ingredientes alimentarios, habilitaciones.
NO solo si es claramente ajeno: deportes, geografía, entretenimiento, matemáticas, política.
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
        ...history.slice(-12),
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
