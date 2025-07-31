// functions/chat.js
// Proxy compatible: reusa la lógica de "valeria" y devuelve ambas claves:
// - reply  (actual)
// - answer (alias para frontends que esperan esta clave)

const valeria = require('./valeria.js');

exports.handler = async (event, context) => {
  // Ejecutamos la function original
  const res = await valeria.handler(event, context);

  // Intentamos unificar el body agregando la clave "answer"
  try {
    const original = JSON.parse(res.body || '{}');
    const reply = typeof original.reply === 'string' ? original.reply : '';

    const unified = {
      // mantenemos todo lo que ya venga
      ...original,
      // garantizamos las dos claves equivalentes
      reply,
      answer: reply
    };

    return {
      ...res,
      body: JSON.stringify(unified)
    };
  } catch {
    // Si por alguna razón no se puede parsear, devolvemos tal cual
    return res;
  }
};
