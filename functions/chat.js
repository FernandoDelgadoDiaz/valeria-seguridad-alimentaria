// ... (mantenemos las importaciones y constantes de arriba)

export async function handler(event) {
  if (event.httpMethod !== "POST") return json({ error: "Método no permitido" }, 405);

  try {
    const { query, history = [] } = JSON.parse(event.body || "{}");
    if (!query) return json({ error: "Consulta vacía" }, 400);

    const data = await loadKnowledge();
    if (!data || !data.chunks) return json({ answer: "Cargando base de datos..." });

    // 1. Crear un 'Contexto de Búsqueda' combinando la pregunta con la anterior si existe
    const searchContext = history.length > 0 
      ? `${history[history.length - 1].content} ${query}` 
      : query;

    const qEmb = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: searchContext,
    });
    const qVec = qEmb.data[0].embedding;

    // 2. RAG: Buscar fragmentos relevantes
    const topChunks = data.chunks
      .map(c => ({ ...c, score: cosSim(qVec, c.embedding || c.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    const contextText = topChunks.map(c => `[Doc]: ${c.text}`).join("\n\n");

    // 3. ARMADO DE MENSAJES PARA LA IA (Incluye el historial)
    const messages = [
      { role: "system", content: INOCUO_SYSTEM_PROMPT },
      { role: "system", content: "CONTEXTO TÉCNICO:\n" + contextText },
      ...history.slice(-4), // Enviamos los últimos 4 mensajes para mantener el hilo
      { role: "user", content: query }
    ];

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: messages,
      temperature: 0.3,
    });

    return json({
      ok: true,
      answer: completion.choices[0].message.content,
      history: [...history, { role: "user", content: query }, { role: "assistant", content: completion.choices[0].message.content }]
    });

  } catch (err) {
    return json({ error: "Error técnico" }, 500);
  }
}
