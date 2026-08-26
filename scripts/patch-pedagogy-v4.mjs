import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');

html=html.replace(/<style id="pedagogy-v4-5s">[\s\S]*?<\/style>/g,'');

const oldLoad="async function loadQuestion(){try{decision=null;current=await rpc('desafio5s_pregunta',{p_evaluacion_id:session.evaluacionId,p_access_token:session.accessToken,p_orden:order});renderQuestion()}catch(e){flash(e.message)}}";

const newLoad=`const standardSeen=new Set();
const standards5s={
 S1:{name:'CLASIFICAR',lead:'En el puesto queda sólo lo necesario. Lo que no corresponde se retira, se separa o se lleva al lugar que corresponde.',items:['Sólo queda lo necesario para la tarea.','Se sabe qué es cada cosa y qué destino tiene.','No quedan sobrantes ni elementos ajenos al puesto.']},
 S2:{name:'ORDENAR',lead:'Cada elemento necesario tiene un lugar definido y vuelve ahí después de usarlo.',items:['Cada cosa tiene un lugar claro.','Pasillos y superficies quedan libres.','Después de usar algo, vuelve a su lugar.']},
 S3:{name:'LIMPIAR',lead:'Si algo se ensucia o queda fuera de lugar, se corrige y también se busca por qué pasó.',items:['Pisos y superficies limpios y en buen estado.','Se puede acceder para limpiar y revisar.','La suciedad y los problemas se corrigen a tiempo.']},
 S4:{name:'ESTANDARIZAR',lead:'Cualquiera debe poder ver rápidamente cómo tiene que quedar el puesto, sin depender de la memoria de una persona.',items:['Lugares e identificaciones visibles.','El mismo criterio para todos los turnos.','Queda claro qué va en cada lugar y para qué se usa.']},
 S5:{name:'SOSTENER',lead:'El estándar se mantiene todos los días y cada tarea termina dejando el puesto como corresponde.',items:['El estándar se mantiene durante toda la jornada.','Los problemas se corrigen antes de que se vuelvan costumbre.','La constancia convierte la mejora en hábito.']}
};
function renderStandard(s){const st=standards5s[s];if(!st)return renderQuestion();const n=Number(String(s).replace('S',''))||1;app.innerHTML=\`<section class="screen standard-screen"><article class="standard-card"><div class="standard-top"><span class="eyebrow">ANTES DE RESPONDER</span><span class="standard-count">PRINCIPIO \${n} DE 5</span></div><div class="standard-code">\${s} · \${st.name}</div><h1>Qué esperamos</h1><p class="standard-lead">\${st.lead}</p><div class="standard-list">\${st.items.map(x=>\`<div><i>✓</i><span>\${x}</span></div>\`).join('')}</div><div class="standard-note"><strong>Ahora aplicalo.</strong><span>Vas a resolver 3 situaciones distintas sobre este principio.</span></div><button id="startBlock" class="btn primary">VER 3 SITUACIONES</button></article></section>\`;document.querySelector('#startBlock').onclick=()=>{renderQuestion();window.scrollTo(0,0)}}
async function loadQuestion(){try{decision=null;current=await rpc('desafio5s_pregunta',{p_evaluacion_id:session.evaluacionId,p_access_token:session.accessToken,p_orden:order});const s=current&&current.s;const key=session.evaluacionId+':'+s;if(s&&standards5s[s]&&!standardSeen.has(key)){standardSeen.add(key);renderStandard(s)}else renderQuestion()}catch(e){flash(e.message)}}`;

if(!html.includes(oldLoad)) throw new Error('No se encontró loadQuestion base para pedagogía v4');
html=html.replace(oldLoad,newLoad);

const css=`<style id="pedagogy-v4-5s">
.standard-screen{display:flex;justify-content:center;align-items:flex-start}
.standard-card{width:min(680px,100%);background:#fff;border:1px solid #dde2e7;border-radius:12px;padding:28px;box-shadow:0 8px 24px rgba(20,43,67,.06)}
.standard-top{display:flex;align-items:center;justify-content:space-between;gap:12px}
.standard-count{font-size:10px;font-weight:900;letter-spacing:.12em;color:#65717d}
.standard-code{display:inline-flex;margin-top:18px;background:#e7f0f8;color:#0755a5;border-radius:8px;padding:8px 10px;font-size:11px;font-weight:900;letter-spacing:.08em}
.standard-card h1{font-size:36px;line-height:1.02;letter-spacing:-.03em;margin:12px 0 10px;color:#20262d}
.standard-lead{font-size:17px;line-height:1.48;color:#465463;margin:0 0 18px;max-width:590px}
.standard-list{display:grid;gap:8px}
.standard-list>div{display:flex;gap:10px;align-items:flex-start;border-top:1px solid #edf0f3;padding:10px 0 2px}
.standard-list i{font-style:normal;width:22px;height:22px;flex:0 0 22px;border-radius:50%;display:grid;place-items:center;background:#eaf6ed;color:#16892b;font-size:12px;font-weight:900}
.standard-list span{font-size:14px;line-height:1.4;color:#2f3b46;padding-top:1px}
.standard-note{display:flex;flex-direction:column;gap:3px;background:#f5f7f9;border-left:3px solid #0755a5;padding:11px 13px;margin-top:17px}
.standard-note strong{font-size:12px;color:#0755a5}
.standard-note span{font-size:12px;line-height:1.35;color:#65717d}
.standard-card .btn{margin-top:16px}
@media(max-width:800px){
 .standard-screen{padding-top:16px!important}
 .standard-card{padding:20px 18px;border-radius:10px}
 .standard-card h1{font-size:29px;margin:10px 0 8px}
 .standard-lead{font-size:15px;line-height:1.42;margin-bottom:13px}
 .standard-code{margin-top:13px;padding:7px 9px}
 .standard-list{gap:4px}
 .standard-list>div{padding:8px 0 1px}
 .standard-list span{font-size:13px}
 .standard-note{margin-top:13px;padding:9px 11px}
 .standard-card .btn{margin-top:13px}
}
</style>`;

html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Pedagogy v4 applied: Argentine plain-language 5S intro before each block; assessment logic remains backend-driven.');
