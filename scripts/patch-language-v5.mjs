import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');

// Final language pass for visible training copy. Keep technical retail terms that add learning value,
// but avoid unnecessarily formal wording in everyday instructions.
const replacements=[
  [/\["LIMPIAR","[^"]*","[^"]*"\]/,'["LIMPIAR","Dejar el sector limpio y detectar por qué vuelve a ensuciarse o desordenarse.","Limpiar también ayuda a encontrar la causa de los problemas."]'],
  [/\["SOSTENER","[^"]*","[^"]*"\]/,'["SOSTENER","Mantener las buenas prácticas todos los días.","La constancia convierte una mejora en hábito."]'],
  [/\["SEGREGACION","[^"]*","[^"]*"\]/,'["SEPARACION","Mantener separados los elementos según su uso, estado o destino.","Separar correctamente evita mezclas y confusiones."]'],
  [/\["DECOMISO","[^"]*","[^"]*"\]/,'["DECOMISO","Producto retirado de la venta que debe ir al lugar o proceso que corresponde.","Debe quedar identificado, separado y con un destino claro."]'],
  [/\["ANOMALIA","[^"]*","[^"]*"\]/,'["PROBLEMA","Situación fuera de lo esperado que necesita atención.","Limpiar y revisar ayuda a detectarla a tiempo."]'],
  [/\["REPOSICION","[^"]*","[^"]*"\]/,'["REPOSICION","Tarea de llevar mercadería desde la reserva hasta la exhibición.","La tarea termina cuando el sector queda como corresponde."]'],
  [/\["ELABORACION","[^"]*","[^"]*"\]/,'["ELABORACION","Proceso de preparación o transformación de un producto.","Al terminar, el puesto debe quedar como corresponde."]']
];

for(const [pattern,value] of replacements) html=html.replace(pattern,value);

fs.writeFileSync(file,html);
console.log('Language v5 applied: Argentine plain-language copy normalized.');
