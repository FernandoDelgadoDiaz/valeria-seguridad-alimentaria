// functions/chat.js
// Proxy: responde igual que /.netlify/functions/valeria

const valeria = require('./valeria.js');

exports.handler = async (event, context) => {
  return valeria.handler(event, context);
};
