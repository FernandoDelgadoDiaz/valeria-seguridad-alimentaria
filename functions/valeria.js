// functions/valeria.js
// Backend de Valeria con identidad, búsqueda vectorial y reglas especiales

const fs = require("fs");
const path = require("path");

// Leer los documentos de data/embeddings.json
function loadDocs() {
  try {
    const raw = fs.readFileSync(path.join(__dirname, "../data/embeddings.json"), "utf8");
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

// Similitud coseno
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// Cache para los embeddings de los documentos
let CACHE = { ready: false, chunks: [], vectors: [], model: "text-embedding-3-small" };

async function ensureEmbeddings(OPENAI_API_KEY) {
  if (CACHE.ready) return;

  const docs = loadDocs(); // documentos ya con chunks
  const chunks = docs.map((d, i) => ({
    id: d.id || `doc${i}`,
    title: d.title || `Documento ${i + 1}`,
    content: d.content || ""
  }));

  // Hacemos embeddings en batch
  async function embedBatch(texts) {
    const resp = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ model: CACHE.model, input: texts })
    });
    if (!resp.ok) throw new Error(`Error creando embeddings (${resp.status})`);
    const data = await resp.json();
    return data.data.map(d => d.embedding);
  }

  // En lotes de 60 para no exceder límites
  const vectors = [];
  const BATCH = 60;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const slice = chunks.slice(i, i + BATCH);
    const embs = await embedBatch(slice.map(c => c.content));
    vectors.push(...embs);
  }

  CACHE.chunks = chunks;
  CACHE.vectors = vectors;
  CACHE.ready = true;
}

exports.handler = async (event) => {
  try {
    const { message } = JSON.parse(event.body || "{}");
    if (!message) {
      return { statusCode: 400, body: JSON.stringify({ reply: "No recibí tu mensaje." }) };
    }

    const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
    if (!OPENAI_API_KEY) {
      return { statusCode: 500, body: JSON.stringify({ reply: "❌ API Key no configurada en Netlify." }) };
    }

    const lower = message.toLowerCase();

    // Respuestas especiales Ishikawa/5 porqués
    const pideIshikawa = lower.includes("espina de pescado") || lower.includes("ishikawa");
    const pide5Porques = lower.includes("5 porqués") || lower.includes("5 porques") || lower.includes("cinco porqués");
    if (pideIshikawa || pide5Porques) {
      const problema = lower.includes("vencid") && lower.includes("góndola")
        ? "Productos vencidos en góndola"
        : message.trim();

      const ishikawa =
`🔎 *Análisis de causa raíz — Diagrama de Ishikawa (Espina de pescado)*
**Problema:** ${problema}

**Métodos / Procedimientos**
- Falta de retiro anticipado según norma interna (10 días masivos/congelados; 2 días perecederos).
- Rotación FEFO no aplicada en reposición.

**Mano de obra**
- Falta de capacitación en control de fechas / rotación.
- Altas cargas de trabajo en punta (se omiten controles).

**Materiales**
- Rotulado interno incompleto o ilegible.
- Etiquetas sin fecha de apertura / fraccionado.

**Maquinaria / Equipos**
- Scanner o app de control de fechas no utilizada o con fallos.
- Heladeras con stock sobrecargado que dificulta la revisión.

**Medio ambiente**
- Exhibidores desordenados, visibilidad pobre de fechas.
- Señalización de “retiro anticipado” ausente.

**Medición / Control**
- Falta de checklist diario de vencimientos por góndola/sector.
- Registros incompletos o no verificados por un responsable.`;

      const cincoPorques =
`\n\n🧠 *5 porqués (ejemplo para “${problema}”)*  
1. ¿Por qué hay productos vencidos en góndola?  
   → Porque no se retiraron a tiempo.  
2. ¿Por qué no se retiraron a tiempo?  
   → Porque no se realizó el control diario de vencimientos.  
3. ¿Por qué no se realizó el control diario?  
   → Porque el personal no tenía asignada la tarea con horario y responsable.  
4. ¿Por qué no estaba asignada?  
   → Porque el procedimiento no lo define claramente y no hay checklist operativo.  
5. ¿Por qué no hay checklist y definición?  
   → Porque no se integró la norma de retiro anticipado (10 días/2 días) al procedimiento y la capacitación.

✅ *Acciones inmediatas*  
- Barrido de góndolas por sector y retiro anticipado (según norma).  
- Señalizar y registrar el retiro (RG 01-SUC-ML-002).

🛠️ *Acciones preventivas*  
- Actualizar el procedimiento con retiro anticipado y responsable.  
- Implementar checklist diario y verificación del supervisor.  
- Capacitación en CAA/BPM, FEFO y retiro anticipado.  
- Auditoría semanal de cumplimiento.`;

      return { statusCode: 200, body: JSON.stringify({ reply: ishikawa + cincoPorques }) };
    }

    // Aseguramos embeddings de los documentos en memoria
    await ensureEmbeddings(OPENAI_API_KEY);

    // Embedding de la consulta
    const embResp = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ model: CACHE.model, input: message })
    });
    if (!embResp.ok) throw new Error(`Error creando embedding de la consulta (${embResp.status})`);
    const embData = await embResp.json();
    const qVec = embData.data[0].embedding;

    // Rankear documentos por similitud
    const scored = CACHE.chunks.map((c, i) => ({ c, s: cosineSim(qVec, CACHE.vectors[i]) }));
    scored.sort((a, b) => b.s - a.s);
    const top = scored.slice(0, 5).map(x => x.c);

    const context = top.map(c => `• ${c.title}: ${c.content}`).join("\n");

    // Prompt con identidad fuerte
    const systemPrompt = `
Sos **Valeria**, Licenciada en Tecnología de los Alimentos (Argentina), especialista en seguridad alimentaria.
Usá terminología local (carne vacuna, freezer, heladera, franco), fundamento en CAA/BPM y procedimientos internos.
Tono profesional y pedagógico; pasos concretos; ofrecé adjuntos PDF cuando corresponda (✔️).
NO copies conversaciones previas. Respondé directo a la consulta.`;

    const contextPrompt = context
      ? `Contexto interno (resumido):\n${context}\n\nRestringí tus afirmaciones técnicas a este contexto cuando aplique.`
      : `No se encontraron coincidencias en contexto interno. Usá criterios generales de CAA/BPM (aclará cuando asumís criterios generales).`;

    // Llamada a OpenAI
    const chatResp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        temperature: 0.4,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "system", content: contextPrompt },
          { role: "user", content: message }
        ]
      })
    });

    if (!chatResp.ok) {
      const txt = await chatResp.text().catch(() => "");
      return { statusCode: 502, body: JSON.stringify({ reply: `Error con OpenAI (${chatResp.status}). ${txt}` }) };
    }

    const data = await chatResp.json();
    const reply = data.choices?.[0]?.message?.content || "No se encontró respuesta.";
    return { statusCode: 200, body: JSON.stringify({ reply }) };

  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ reply: "Error inesperado en el servidor." }) };
  }
};
