import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html
.replaceAll('¿Qué afirmaciones son correctas sobre esta situación?','¿Cuáles de estas afirmaciones son verdaderas según el estándar 5S?')
.replaceAll('¿Qué situaciones de la imagen requieren corrección según 5S?','¿Cuáles de estas afirmaciones son verdaderas según el estándar 5S?')
.replaceAll('¿Cuáles de estas observaciones representan un desvío que deberías corregir?','¿Cuáles de estas afirmaciones son verdaderas según el estándar 5S?')
.replaceAll('Marcá todas las afirmaciones correctas. Puede haber una o varias.','Marcá todas las afirmaciones que consideres verdaderas. Puede haber una o varias.')
.replaceAll('Marcá únicamente las situaciones que representan un incumplimiento o una oportunidad de mejora. Puede haber una o varias.','Marcá todas las afirmaciones que consideres verdaderas. Puede haber una o varias.')
.replaceAll('Marcá solo las observaciones que requieren una acción correctiva según 5S. Algunas describen condiciones aceptables y no deben marcarse.','Marcá todas las afirmaciones que consideres verdaderas. Puede haber una o varias.');
fs.writeFileSync(file,html);
console.log('Visual challenge wording reframed as true statements according to 5S.');
