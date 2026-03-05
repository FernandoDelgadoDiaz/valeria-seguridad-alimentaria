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
      /dame|mostr[aá]|copi[aá]|transcrib|pas[aá]me|qu[eé] dice|extra[eé]|obten[eé]|traeme|dime|necesito|busca|encontr[aá]|consult[aá]|ver|ve[aá]|lee|le[eé]me/i.test(query) ||
      /art[ií]culo\s*\d+/i.test(query) ||
      /cap[ií]tulo\s*(completo|entero|todo)/i.test(query) ||
      (/cap[ií]tulo/i.test(query) && /art[ií]culo/i.test(query));

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

    // Boost: si la query menciona un tema específico, priorizar ese capítulo
    let topCAA;
    topCAA = caaChunks.slice(0, 5);

    // Mapa de capítulos a keywords de archivo
    const CAP_FILE_MAP = {
      'i':    ['capitulo_i','capitulo_1','disp_grales'],
      'ii':   ['capitulo_ii','establec'],
      'iii':  ['capitulo_iii','prod_alimenticios'],
      'iv':   ['capitulo_iv','envases'],
      'v':    ['capitulo_v','rotulacion'],
      'vi':   ['capitulo_vi','carneos'],
      'vii':  ['capitulo_vii','alimentos_grasos','caa_cap_alimentos'],
      'viii': ['capitulo_viii','lacteos'],
      'ix':   ['capitulo_ix','harinas'],
      'x':    ['capitulo_x','azucarados'],
      'xi':   ['capitulo_xi','vegetales'],
      'xii':  ['capitulo_xii','aguas'],
      'xiii': ['capitulo_xiii','beb_fermentadas'],
      'xiv':  ['capitulo_xiv','caa_capitulo_xiv'],
      'xv':   ['capitulo_xv','estimulantes'],
      'xvi':  ['capitulo_xvi','correctivos'],
      'xvii': ['capitulo_xvii','dieteticos'],
      'xviii':['capitulo_xviii','aditivos'],
      'xix':  ['capitulo_xix','aislados_prot'],
      'xx':   ['capitulo_xx','metodologia'],
      'xxi':  ['capitulo_xxi','procedimientos'],
      'xxii': ['capitulo_xxii','miscelaneos'],
    };
    const ARAB_TO_ROMAN = {'1':'i','2':'ii','3':'iii','4':'iv','5':'v','6':'vi','7':'vii',
      '8':'viii','9':'ix','10':'x','11':'xi','12':'xii','13':'xiii','14':'xiv','15':'xv',
      '16':'xvi','17':'xvii','18':'xviii','19':'xix','20':'xx','21':'xxi','22':'xxii'};

    // Búsqueda exacta por artículo/capítulo
    let exactMatches = [];
    if (pideExacto) {
      const artMatch = query.match(/art[ií]culo\s*(?:n[°º]?\s*)?(\d+)/i);
      const capMatch = query.match(/cap[ií]tulo\s*(?:n[°º]?\s*)?(\d+|[IVXivx]+)/i);
      const artNum = artMatch ? artMatch[1] : null;
      let capRoman = capMatch ? capMatch[1].toLowerCase() : null;
      if (capRoman && ARAB_TO_ROMAN[capRoman]) capRoman = ARAB_TO_ROMAN[capRoman];
      const capKeywords = capRoman ? (CAP_FILE_MAP[capRoman] || [capRoman]) : null;

      const isFromCap = (source) => {
        const s = source.toLowerCase();
        const isCAA = s.includes("capitulo") || s.includes("caa") || s.includes("anmat");
        if (!capKeywords) return isCAA;
        return capKeywords.some(k => s.includes(k));
      };

      if (artNum || capRoman) {
        exactMatches = CACHE_DATA.chunks.filter(c => {
          if (!isFromCap(c.source)) return false;
          const t = c.text || '';
          const hasArt = artNum ? (['artículo ' + artNum, 'articulo ' + artNum, 'Artículo ' + artNum, 'ARTÍCULO ' + artNum, 'Art. ' + artNum, 'Art ' + artNum, 'ART ' + artNum, 'ART. ' + artNum, 'art ' + artNum].some(p => t.includes(p))) : true;
          return hasArt;
        });
        if (exactMatches.length === 0 && artNum) {
          exactMatches = CACHE_DATA.chunks.filter(c => {
            const s = c.source.toLowerCase();
            const isCAA = s.includes("capitulo") || s.includes("caa") || s.includes("anmat");
            return isCAA && ['artículo ' + artNum, 'articulo ' + artNum, 'Artículo ' + artNum, 'ARTÍCULO ' + artNum, 'Art. ' + artNum, 'Art ' + artNum, 'ART ' + artNum, 'ART. ' + artNum, 'art ' + artNum].some(p => (c.text || '').includes(p));
          });
        }
      }
    }

    // Jerarquía de contexto
    let contextChunks = [];
    if (pideExacto) {
      const exactSet = new Set(exactMatches.map(c => c.text));
      const semCAA = topCAA.filter(c => !exactSet.has(c.text));
      contextChunks = [...exactMatches.slice(0, 6), ...semCAA.slice(0, 2)];
      if (contextChunks.length === 0) contextChunks = topCAA;
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

    // Construir contexto con metadata de citas
    const contextText = contextChunks.map(c => {
      let header = '';
      if (c.capitulo) {
        header = `[CAA — Cap. ${c.capitulo}`;
        if (c.articulos && c.articulos.length > 0) {
          header += `, Art. ${c.articulos.join(', ')}`;
        }
        header += ']\n';
      }
      return header + c.text;
    }).join("\n\n---\n\n");

    // Citas disponibles para que el modelo las use
    const citasDisponibles = contextChunks
      .filter(c => c.capitulo)
      .map(c => {
        if (c.articulos && c.articulos.length > 0) {
          return `CAA, Cap. ${c.capitulo}, Art. ${c.articulos[0]}`;
        }
        return `CAA, Cap. ${c.capitulo}`;
      });

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
              { role: "user", content: `Genera un test de 5 preguntas de opción múltiple exclusivamente sobre el tema de la siguiente explicación. Cada pregunta con 3 opciones (a, b, c). Devuelve solo JSON con: "preguntas" (array de strings, formato "Pregunta? a) ... b) ... c) ...") y "respuestasCorrectas" (objeto con claves "1" a "5", valores "a","b","c").\n\nContexto:\n${contextText}\n\nExplicación:\n${lastExplanation}` }
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

IDIOMA: Siempre español rioplatense. "vos", "tenés", "querés", "podés", "necesitás". Nunca "tú", "tienes", "quieres", "puedes".

FORMATO: Respondé en texto corrido, sin headers ni listas. Si enumerás, usá: "1) ... 2) ... 3) ...". Máximo 4 párrafos cortos. Tono de colega experto.

FUENTES: Info interna: respondé directo sin mencionar fuente. Info del CAA: si el CONTEXTO tiene encabezados como [CAA — Cap. X, Art. Y], usá ESE dato exacto para citar al final → *Fuente: CAA, Cap. X, Art. Y*. Si el contexto NO tiene encabezado con capítulo y artículo, NO cites — es mejor no citar que inventar.

FUERA DE DOMINIO: "Soy INOCUO, especializado en seguridad alimentaria y BPM. Esta consulta está fuera de mi área. Si tenés dudas sobre inocuidad, normativas del CAA o manipulación de alimentos, ¡con gusto te ayudo!"

SEGUIMIENTO: Leé el historial. No repitas lo ya dicho.

CITAS DISPONIBLES: ${citasDisponibles.length > 0 ? citasDisponibles.join(' | ') : 'ninguna detectada'}

CONTEXTO:
${contextText}`;
    } else {
      systemPrompt = `Eres INOCUO, experto en seguridad alimentaria y BPM. Explicás como un buen docente: claro, progresivo, con ejemplos reales de la industria argentina.

IDIOMA: Siempre español rioplatense. "vos", "tenés", "querés", "podés". Nunca "tú", "tienes", "quieres", "puedes".

ESTRUCTURA: 1) Definición simple en 2-3 oraciones. 2) Desarrollo en párrafos cortos, sin listas con guiones. 3) 1 o 2 ejemplos concretos de la industria alimentaria argentina. 4) Siempre al final: "**Para profundizar respondé '1'. Para hacer un test respondé '2'.**"

FORMATO: Negritas solo para conceptos clave. Párrafos fluidos, no bullets. Tono de capacitador experimentado, no infantil.

FUENTES: Info del CAA: citá al final → *Fuente: CAA, Cap. [X], Art. [Y]*

FUERA DE DOMINIO: "Soy INOCUO, especializado en seguridad alimentaria y BPM. Esta consulta está fuera de mi área."

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
