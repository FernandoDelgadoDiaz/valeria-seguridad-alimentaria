import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html
.replaceAll('¿Qué afirmaciones son correctas sobre esta situación?','¿Qué situaciones de la imagen requieren corrección según 5S?')
.replaceAll('Marcá todas las afirmaciones correctas. Puede haber una o varias.','Marcá únicamente las situaciones que representan un incumplimiento o una oportunidad de mejora. Puede haber una o varias.');
fs.writeFileSync(file,html);
console.log('Visual challenge wording clarified without altering scoring or bonus game.');
