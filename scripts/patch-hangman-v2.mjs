import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');

function hangSvgV2(errors,face='neutral'){
  const show=i=>errors>=i?'visible':'hidden';
  const mouth=face==='happy'?'<path d="M110 66 Q120 74 130 66"/>':face==='sad'?'<path d="M110 70 Q120 63 130 70"/>':'<line x1="112" y1="67" x2="128" y2="67"/>';
  return `<svg class="hang-svg ${face}" viewBox="0 0 220 230" aria-label="Ahorcado"><line class="gallows" x1="20" y1="210" x2="190" y2="210"/><line class="gallows" x1="48" y1="210" x2="48" y2="18"/><line class="gallows" x1="48" y1="18" x2="120" y2="18"/><line class="gallows" x1="120" y1="18" x2="120" y2="40"/><g style="visibility:${show(1)}"><circle cx="120" cy="59" r="19"/><circle class="eye" cx="113" cy="56" r="1.8"/><circle class="eye" cx="127" cy="56" r="1.8"/>${mouth}</g><line style="visibility:${show(2)}" x1="120" y1="78" x2="120" y2="132"/><line style="visibility:${show(3)}" x1="120" y1="94" x2="90" y2="118"/><line style="visibility:${show(4)}" x1="120" y1="94" x2="150" y2="118"/><line style="visibility:${show(5)}" x1="120" y1="132" x2="98" y2="174"/><line style="visibility:${show(6)}" x1="120" y1="132" x2="142" y2="174"/></svg>`;
}
function successWalkSvgV2(){
  return `<svg class="escape-svg" viewBox="0 0 330 235" aria-label="Muñeco liberándose de la horca"><line class="escape-gallows" x1="20" y1="210" x2="305" y2="210"/><line class="escape-gallows" x1="48" y1="210" x2="48" y2="18"/><line class="escape-gallows" x1="48" y1="18" x2="120" y2="18"/><line class="escape-gallows" x1="120" y1="18" x2="120" y2="40"/><line class="escape-rope" x1="120" y1="40" x2="120" y2="47"/><g class="escaped-man"><circle cx="120" cy="62" r="19"/><circle class="eye" cx="113" cy="58" r="1.8"/><circle class="eye" cx="127" cy="58" r="1.8"/><path d="M110 66 Q120 75 130 66"/><line x1="120" y1="81" x2="120" y2="135"/><line x1="120" y1="96" x2="90" y2="118"/><line x1="120" y1="96" x2="150" y2="118"/><line class="walk-leg-a" x1="120" y1="135" x2="98" y2="176"/><line class="walk-leg-b" x1="120" y1="135" x2="142" y2="176"/></g></svg>`;
}
function hangSuccessV2(){
  const[w,d,tip]=hangSet[hangIndex];
  app.innerHTML=`<section class="screen"><div class="hang-card outcome success escape-success"><div class="outcome-face">😄</div><div class="eyebrow">¡MUY BIEN!</div><h1>${esc(w)}</h1><div class="escape-message">¡Te liberaste!</div><div class="outcome-figure escape-scene">${successWalkSvg()}</div><p>${esc(tip)}</p><button id="nextHang" class="btn primary">SIGUIENTE PALABRA</button></div></section>`;
  document.querySelector('#nextHang').onclick=()=>{hangIndex++;hangUsed=new Set();hangErrors=0;if(hangIndex>=hangSet.length)hangEnd();else drawHangman()};
}
function hangFailV2(){
  const[w,d,tip]=hangSet[hangIndex];
  app.innerHTML=`<section class="screen"><div class="hang-card outcome fail"><div class="outcome-face">😢</div><div class="eyebrow">¡UPS! TE QUEDASTE SIN INTENTOS</div><p>La palabra correcta era:</p><h1><strong>${esc(w)}</strong></h1><div class="outcome-figure zoomed">${hangSvg(6,'sad')}</div><p>${esc(tip)}</p><button id="nextHang" class="btn danger">SIGUIENTE PALABRA</button></div></section>`;
  document.querySelector('#nextHang').onclick=()=>{hangIndex++;hangUsed=new Set();hangErrors=0;if(hangIndex>=hangSet.length)hangEnd();else drawHangman()};
}

const newHangSvg=hangSvgV2.toString().replace('hangSvgV2','hangSvg');
const helper=successWalkSvgV2.toString().replace('successWalkSvgV2','successWalkSvg');
const newSuccess=hangSuccessV2.toString().replace('hangSuccessV2','hangSuccess');
const newFail=hangFailV2.toString().replace('hangFailV2','hangFail');

html=html.replace(/function hangSvg\(errors,face='neutral'\)\{[\s\S]*?\}\nfunction drawHangman/,newHangSvg+'\n'+helper+'\nfunction drawHangman');
html=html.replace(/function hangSuccess\(\)\{[\s\S]*?\}\nfunction hangFail/,newSuccess+'\nfunction hangFail');
html=html.replace(/function hangFail\(\)\{[\s\S]*?\}\nfunction hangEnd/,newFail+'\nfunction hangEnd');

const extra=`<style id="hangman-v2-style">
.escape-success{background:linear-gradient(180deg,#f2fff4,#ffffff)!important;border-color:#9bd9a2!important}.escape-message{font-weight:900;color:#16892b;font-size:18px;letter-spacing:.04em;margin:6px 0 0}.escape-scene{overflow:hidden}.escape-svg{width:100%;max-width:420px;height:auto}.escape-svg *{fill:none;stroke:#149636;stroke-width:5;stroke-linecap:round}.escape-svg .escape-gallows{stroke:#243b57;stroke-width:7}.escape-svg .eye{fill:#17365a;stroke:none}.escape-rope{animation:ropeRelease 2.8s ease-in-out both}.escaped-man{transform-box:fill-box;transform-origin:center;animation:escapeWalk 2.8s cubic-bezier(.2,.75,.25,1) both}.walk-leg-a{animation:legA .32s linear 1.15s infinite alternate}.walk-leg-b{animation:legB .32s linear 1.15s infinite alternate}.outcome.fail .hang-svg.sad path{stroke:#d93b35!important}.outcome.fail .hang-svg.sad{stroke:#d93b35!important}
@keyframes ropeRelease{0%,18%{opacity:1}28%,100%{opacity:0}}@keyframes escapeWalk{0%,18%{transform:translate(0,0)}32%{transform:translate(0,38px) rotate(5deg)}48%{transform:translate(38px,38px) rotate(0deg)}68%{transform:translate(90px,38px)}84%{transform:translate(135px,38px)}100%{transform:translate(172px,38px)}}@keyframes legA{from{transform:rotate(0deg)}to{transform:rotate(11deg)}}@keyframes legB{from{transform:rotate(0deg)}to{transform:rotate(-11deg)}}
</style>`;
html=html.replace('</head>',extra+'</head>');
fs.writeFileSync(file,html);
console.log('Hangman v2 applied: corrected sad face and animated green escape on success.');
