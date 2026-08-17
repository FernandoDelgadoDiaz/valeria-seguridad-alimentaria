import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html
.replaceAll('¿Qué afirmaciones son correctas sobre esta situación?','¿Cuáles de estas observaciones representan un desvío que deberías corregir?')
.replaceAll('¿Qué situaciones de la imagen requieren corrección según 5S?','¿Cuáles de estas observaciones representan un desvío que deberías corregir?')
.replaceAll('Marcá todas las afirmaciones correctas. Puede haber una o varias.','Marcá solo las observaciones que requieren una acción correctiva según 5S. Algunas describen condiciones aceptables y no deben marcarse.')
.replaceAll('Marcá únicamente las situaciones que representan un incumplimiento o una oportunidad de mejora. Puede haber una o varias.','Marcá solo las observaciones que requieren una acción correctiva según 5S. Algunas describen condiciones aceptables y no deben marcarse.');
fs.writeFileSync(file,html);
console.log('Visual challenge wording clarified: mark only corrective deviations.');
