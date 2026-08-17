import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');

// Require document on the initial official registration so later access can be legajo + documento.
html=html.replace(
  '<label class="field">Legajo<input id="legajo" required inputmode="numeric"></label><label class="field">Sector',
  '<label class="field">Legajo<input id="legajo" required inputmode="numeric"></label><label class="field">Documento<input id="documento" required inputmode="numeric" autocomplete="off"></label><label class="field">Sector'
);
html=html.replace(
  "p_apellido:apellido.value.trim(),p_sector:sector.value",
  "p_apellido:apellido.value.trim(),p_sector:sector.value,p_documento:documento.value.trim()"
);

// Stronger post-evaluation access: legajo + documento, never apellido.
html=html.replace(
  '<p class="small">Ingresá tu legajo y apellido para proteger tu resultado personal.</p><label class="field">Legajo<input id="rankLegajo" inputmode="numeric"></label><label class="field">Apellido<input id="rankApellido" autocomplete="family-name"></label>',
  '<p class="small">Ingresá tu legajo y documento. Podés consultar tu resultado, revisar tus respuestas, mirar el ranking y jugar al ahorcado todas las veces que quieras.</p><label class="field">Legajo<input id="rankLegajo" inputmode="numeric"></label><label class="field">Documento<input id="rankDocumento" inputmode="numeric" autocomplete="off"></label>'
);
html=html.replace(
  "rpc('desafio5s_acceso_ranking',{p_legajo:rankLegajo.value.trim(),p_apellido:rankApellido.value.trim()})",
  "rpc('desafio5s_acceso_ranking',{p_legajo:rankLegajo.value.trim(),p_documento:rankDocumento.value.trim()})"
);

// Result becomes the permanent participant hub.
html=html.replace(
  'Evaluación finalizada. Tu legajo queda habilitado únicamente para consultar este resultado y el ranking.',
  'Evaluación finalizada. Con tu legajo y documento podés volver cuando quieras para consultar tu resultado, revisar tus respuestas, ver el ranking y jugar al ahorcado.'
);
html=html.replace(
  '<button id="bonus" class="btn primary">🎮 DESAFÍO EXTRA · AHORCADO</button><button id="rank" class="btn secondary">VER RANKING OFICIAL</button>',
  '<button id="review" class="btn secondary">🔎 REVISAR MIS RESPUESTAS</button><button id="bonus" class="btn primary">🎮 JUGAR AL AHORCADO</button><button id="rank" class="btn secondary">VER RANKING OFICIAL</button>'
);
html=html.replace(
  "document.querySelector('#bonus').onclick=startHangman;document.querySelector('#rank').onclick=showRanking;if(testMode)",
  "document.querySelector('#review').onclick=reviewAnswers;document.querySelector('#bonus').onclick=startHangman;document.querySelector('#rank').onclick=showRanking;if(testMode)"
);

// Ranking also acts as a recurrent access point to review and hangman.
html=html.replace(
  '<button id="backResult" class="btn secondary">VOLVER A MI RESULTADO</button>',
  '<button id="rankReview" class="btn secondary">🔎 REVISAR MIS RESPUESTAS</button><button id="rankHang" class="btn primary">🎮 JUGAR AL AHORCADO</button><button id="backResult" class="btn secondary">VOLVER A MI RESULTADO</button>'
);
html=html.replace(
  "document.querySelector('#backResult').onclick=showResult}catch(e)",
  "document.querySelector('#rankReview').onclick=reviewAnswers;document.querySelector('#rankHang').onclick=startHangman;document.querySelector('#backResult').onclick=showResult}catch(e)"
);

const reviewFn=String.raw`async function reviewAnswers(){
 try{
  const d=await rpc('desafio5s_revision',{p_evaluacion_id:session.evaluacionId,p_access_token:session.accessToken});
  const all=d.items||[];
  const errors=all.filter(function(x){return !x.es_correcta || (x.tipo==='fotografica' && Number(x.observacion_score||0)<Number(x.observacion_total||0));});
  const cards=errors.map(function(x){
    const visual=x.tipo==='fotografica';
    const obsUser=(x.observaciones_usuario||[]).map(function(v){return '<li>'+esc(v)+'</li>';}).join('');
    const obsCorrect=(x.observaciones_correctas||[]).map(function(v){return '<li>'+esc(v)+'</li>';}).join('');
    return '<article class="review-card"><div class="review-head"><span>Situación '+x.orden+' · '+esc(x.s)+'</span><b>'+(!x.es_correcta?'DECISIÓN':'OBSERVACIÓN')+'</b></div><h3>'+esc(x.pregunta)+'</h3>'+(!x.es_correcta?'<div class="review-answer wrong"><small>TU RESPUESTA</small><p>'+esc(x.respuesta_usuario||'')+'</p></div><div class="review-answer right"><small>RESPUESTA CORRECTA</small><p>'+esc(x.respuesta_correcta||'')+'</p></div>':'')+(visual&&Number(x.observacion_score||0)<Number(x.observacion_total||0)?'<div class="review-visual"><div><small>LO QUE MARCASTE</small><ul>'+(obsUser||'<li>Ninguna afirmación</li>')+'</ul></div><div><small>LAS AFIRMACIONES VERDADERAS</small><ul>'+obsCorrect+'</ul></div></div>':'')+(x.feedback?'<div class="review-feedback"><strong>POR QUÉ</strong><p>'+esc(x.feedback)+'</p></div>':'')+'</article>';
  }).join('');
  app.innerHTML='<section class="screen review-screen"><div class="eyebrow">TU APRENDIZAJE</div><h1>Revisá dónde te equivocaste</h1><p class="review-intro">No cambia tu puntaje. Esta revisión te muestra qué elegiste, cuál era el criterio correcto y el feedback para aprender del error.</p><div class="review-summary"><div><span>SITUACIONES CON ERROR</span><b>'+errors.length+'</b></div><div><span>RESPUESTAS REVISADAS</span><b>'+all.length+'</b></div></div>'+(cards||'<div class="review-perfect"><strong>🏆 No tenés respuestas para corregir</strong><p>Completaste la evaluación sin errores pendientes de revisión.</p></div>')+'<button id="reviewRank" class="btn secondary">VER RANKING OFICIAL</button><button id="reviewHang" class="btn primary">🎮 JUGAR AL AHORCADO</button><button id="reviewBack" class="btn ghost">VOLVER A MI RESULTADO</button></section>';
  document.querySelector('#reviewRank').onclick=showRanking;
  document.querySelector('#reviewHang').onclick=startHangman;
  document.querySelector('#reviewBack').onclick=showResult;
 }catch(e){flash(e.message)}
}`;
html=html.replace('function rankingAccess(){',reviewFn+'\nfunction rankingAccess(){');

const css=`<style id="participant-portal-v1">
.review-intro{color:#667d96;line-height:1.5;max-width:680px}.review-summary{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:18px 0}.review-summary div{background:#fff;border:1px solid #d7e3ef;border-radius:16px;padding:15px}.review-summary span{display:block;color:#667d96;font-size:10px;font-weight:900;letter-spacing:.06em}.review-summary b{display:block;color:#0755a5;font-size:28px;margin-top:3px}.review-card{background:#fff;border:1px solid #d7e3ef;border-radius:20px;padding:20px;margin:12px 0}.review-head{display:flex;justify-content:space-between;gap:10px;color:#667d96;font-size:11px;font-weight:900;letter-spacing:.06em}.review-head b{color:#d93b35}.review-card h3{font-size:20px;margin:12px 0 14px}.review-answer{border-radius:14px;padding:13px 15px;margin:8px 0}.review-answer small,.review-visual small{font-size:9px;font-weight:900;letter-spacing:.08em}.review-answer p{margin:5px 0 0;line-height:1.4}.review-answer.wrong{background:#fff1f0;border:1px solid #efb1ac}.review-answer.wrong small{color:#c43831}.review-answer.right{background:#eef9ef;border:1px solid #aedaaf}.review-answer.right small{color:#16892b}.review-visual{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:10px 0}.review-visual>div{background:#f7fafc;border:1px solid #dbe5ee;border-radius:14px;padding:13px}.review-visual ul{padding-left:18px;margin:7px 0 0}.review-visual li{margin:5px 0;line-height:1.35}.review-feedback{background:#eef5fb;border-radius:14px;padding:14px;margin-top:10px}.review-feedback strong{color:#0755a5;font-size:10px;letter-spacing:.08em}.review-feedback p{margin:6px 0 0;line-height:1.45}.review-perfect{background:#eef9ef;border:1px solid #aedaaf;border-radius:18px;padding:20px;text-align:center;margin:18px 0}.review-perfect strong{color:#16892b}@media(max-width:800px){.review-screen h1{font-size:30px}.review-card{padding:16px}.review-card h3{font-size:18px}.review-visual{grid-template-columns:1fr}.review-summary b{font-size:24px}}
</style>`;
html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Participant portal applied: legajo+documento, reusable ranking/hangman and answer review.');
