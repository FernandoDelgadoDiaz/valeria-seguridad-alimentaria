import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="institutional-mobile-v31-5s">[\s\S]*?<\/style>/g,'');
html=html.replace(/<script id="institutional-mobile-v31-runtime-5s">[\s\S]*?<\/script>/g,'');

const css=`<style id="institutional-mobile-v31-5s">
/* Institutional pass v3.2 — mobile-only tuning. No scoring, navigation or business-logic changes. */
@media(max-width:800px){
  /* Home: preserve the approved composition, keep the wheel readable and collision-free. */
  .hero{min-height:232px!important;padding:19px 20px 58px!important}
  .institutional-wheel{right:14px!important;top:27px!important;width:128px!important;gap:5px!important}
  .wheel-disc{width:70px!important;height:70px!important}
  .wheel-disc:after{inset:20px!important}
  .wheel-legend{width:128px!important;padding:5px 6px!important;gap:3px 5px!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr)!important;overflow:hidden!important}
  .wheel-legend b{font-size:7.2px!important;line-height:1.12!important;padding-left:8px!important;min-width:0!important;white-space:normal!important;overflow-wrap:break-word!important}
  .wheel-legend b:before{width:4px!important;height:4px!important}
  .wheel-legend b:nth-child(4){grid-column:1/-1!important;grid-row:3!important;text-align:center!important;padding-left:0!important}
  .wheel-legend b:nth-child(4):before{position:static!important;display:inline-block!important;margin-right:4px!important;vertical-align:middle!important}
  .wheel-legend b:nth-child(5){grid-column:2!important;grid-row:2!important;text-align:left!important;padding-left:8px!important}
  .wheel-legend b:nth-child(5):before{position:absolute!important;display:block!important;left:0!important;top:.34em!important;margin-right:0!important;vertical-align:initial!important;width:4px!important;height:4px!important}
  .hero-meta{bottom:12px!important;gap:6px!important}
  .hero-chip{font-size:9.5px!important;padding:6px 7px!important;line-height:1.1!important}

  /* Test-mode banner must stay fully visible below the sticky corporate header. */
  .testbar.on{position:sticky!important;top:52px!important;z-index:19!important;padding:5px 10px!important;font-size:8.5px!important;line-height:1.15!important}

  /* Visual challenges: keep the current image size so operational details remain readable. */
  .question-shell:has(.visual-layout) .visual-photo{max-height:min(34vh,280px)!important}
  .question-shell:has(.visual-layout) .visual-photo img{max-height:min(34vh,280px)!important}
  .question-shell:has(.visual-layout) .visual-content{padding-top:14px!important}

  /* Decision step: reduce reading height while preserving hierarchy. */
  .question-shell:has(.visual-layout) .decision h2{font-size:19px!important;line-height:1.22!important;letter-spacing:-.01em!important;margin:8px 0 12px!important}
  .question-shell:has(.visual-layout) .decision{padding-top:13px!important}
  .question-shell:has(.visual-layout) .decision .options{gap:7px!important}
  .question-shell:has(.visual-layout) .decision .option{padding:10px 11px!important;min-height:50px!important}
  .question-shell:has(.visual-layout) .decision .option.selected{padding:9px 10px!important}
  .question-shell:has(.visual-layout) .decision .option span{font-size:15px!important;line-height:1.3!important}
}
</style>`;

const runtime=`<script id="institutional-mobile-v31-runtime-5s">
(()=>{
  const labels=['Personas','Consistencia','Adaptabilidad','Análisis de peligros y riesgos','Misión y visión'];
  const fixWheel=()=>{
    const nodes=document.querySelectorAll('.wheel-legend b');
    if(nodes.length!==5)return;
    nodes.forEach((node,i)=>{if(node.textContent!==labels[i])node.textContent=labels[i]});
    const wheel=document.querySelector('.institutional-wheel');
    if(wheel)wheel.setAttribute('aria-label','Dimensiones institucionales');
  };
  const app=document.querySelector('#app');
  if(app)new MutationObserver(fixWheel).observe(app,{childList:true,subtree:true});
  fixWheel();
})();
</script>`;

html=html.replace('</head>',css+'</head>').replace('</body>',runtime+'</body>');
fs.writeFileSync(file,html);
console.log('Institutional mobile v3.2 applied; wheel legend contained; presentation-only tuning.');
