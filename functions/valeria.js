// functions/valeria.js
const fs = require("fs");
const path = require("path");

let documents = [];
try {
  const raw = fs.readFileSync(path.join(__dirname, "../data/embeddings.json"), "utf8");
  documents = JSON.parse(raw);
} catch (e) {
  documents = [];
}

// util: coseno (por si ya tenés "embedding" numérico en cada doc)
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
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

    // 1) Recuperación de contexto
    // Si tus documentos ya tienen embeddings (campo "embedding": number[]),
    // podés calcular la similitud coseno contra un embedding del mensaje del usuario.
    // Mientras tanto, hacemos fallback a "includes" (starter).

    // --- Fallback simple por includes ---
    const simpleTop = documents
      .map(d => ({
        doc: d,
        score: (d.content || "").toLowerCase().includes(message.toLowerCase()) ? 1 : 0
      }))
      .filter(x => x.score > 0)
      .sort((a,b)=> b.score - a.score)
      .slice(0, 5)
      .map(x => x.doc);

    const contextText = simpleTop.map(d => `• ${d.title}: ${d.content}`).join("\n");

    // 2) Reglas especiales: Ishikawa + 5 porqués
    const lower = message.toLowerCase();
    const pideIshikawa = lower.includes("espina de pescado") || lower.includes("ishikawa");
    const pide5Porques  = lower.includes("5 porqués") || lower.includes("5 porques") || lower.includes("cinco porqués");
    if (pideIshikawa || pide5Porques) {
      const problema =
        lower.includes("vencid") && lower.includes("góndola")
          ? "Productos vencidos en góndola"
          : (message.trim() || "Problema a definir");

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

    // 3) Llamada al modelo con identidad fuerte
    const systemPrompt = `
Sos **Valeria**, Licenciada en Tecnología de los Alimentos (Argentina), especialista en seguridad alimentaria.
Usá terminología local (carne vacuna, freezer, heladera, franco), fundamento en CAA/BPM y procedimientos internos.
Tono profesional y pedagógico; pasos concretos; ofrecé adjuntos PDF cuando corresponda (✔️).
NO copies conversaciones previas ni formatees como transcripción. Respondé directo a la consulta.
Si el contexto interno no cubre la pregunta, respondé con criterios generales de CAA/BPM y decí cuándo asumís criterios generales.
`;

    const userPrompt = message;
    const contextPrompt = contextText
      ? `Contexto interno (resumido):\n${contextText}\n\nRestringí tus afirmaciones técnicas a este contexto cuando aplique.`
      : `No se encontraron coincidencias en contexto interno. Usá criterios generales de CAA/BPM, dejando claro que son generales.`;

    const resp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        temperature: 0.4,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "system", content: contextPrompt },
          { role: "user", content: userPrompt }
        ],
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      return { statusCode: 502, body: JSON.stringify({ reply: `Error con OpenAI (${resp.status}). ${errText}` }) };
    }

    const data = await resp.json();
    const reply = data?.choices?.[0]?.message?.content || "No se encontró respuesta.";
    return { statusCode: 200, body: JSON.stringify({ reply }) };

  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ reply: "Error inesperado en el servidor." }) };
  }
};

