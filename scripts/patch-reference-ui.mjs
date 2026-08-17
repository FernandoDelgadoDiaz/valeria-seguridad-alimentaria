import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');

html=html.replace(/<style id="reference-ui-5s">[\s\S]*?<\/style>/g,'');
html=html.replace(/<script id="reference-ui-runtime-5s">[\s\S]*?<\/script>/g,'');

const css=`<style id="reference-ui-5s">
:root{--ok:#1b9c34;--ok-bg:#eef9ef;--warn:#d79a00;--warn-bg:#fff7dc;--bad:#d93b35;--bad-bg:#fff0ef;--deep:#0b4f98;--blue:#0755a5;--ink:#10233f;--muted:#667d96;--line:#d7e3ef}
body{background:#eef4fa!important}.screen{max-width:760px!important;padding-bottom:96px!important}
.result-celebration{display:flex;align-items:center;gap:18px;background:linear-gradient(135deg,#0755a5,#063f7f);color:#fff;border-radius:22px;padding:22px 24px;margin-bottom:14px;box-shadow:0 14px 30px rgba(7,85,165,.18)}
.result-celebration .cup{font-size:54px;line-height:1}.result-celebration strong{display:block;font-size:25px}.result-celebration span{display:block;margin-top:6px;font-size:14px;opacity:.95}
.result-grid{gap:12px!important;margin-top:0!important}.metric{border-radius:18px!important;padding:20px!important;position:relative}.metric b{font-size:36px!important}.metric:after{position:absolute;right:15px;bottom:14px;width:38px;height:38px;border-radius:50%;display:grid;place-items:center;color:#fff;font-weight:900;font-size:21px}.metric:first-child:after{content:'✓';background:var(--ok)}.metric:nth-child(2):after{content:'★';background:#f4bf00}
.result-hero{background:#fff!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:20px!important;padding:22px!important;margin-top:12px!important;box-shadow:none!important}.result-hero:before{content:'RESULTADO FINAL';display:block;color:var(--blue);font-size:12px;font-weight:900;letter-spacing:.08em;margin-bottom:10px}.result-index small{display:none!important}.result-index strong{font-size:64px!important;line-height:.95!important}.result-top{align-items:center!important}.result-hero .small,.result-hero p{color:var(--ink)!important}
.result-hero.status-green .result-index strong{color:var(--ok)!important}.result-hero.status-yellow .result-index strong{color:var(--warn)!important}.result-hero.status-red .result-index strong{color:var(--bad)!important}
.result-status{display:inline-flex!important;align-items:center;gap:7px;border-radius:10px;padding:8px 12px!important;font-size:17px!important;margin:0!important}.result-status:before{width:21px;height:21px;border-radius:50%;display:grid;place-items:center;color:#fff;font-size:12px;font-weight:900}.status-green .result-status{border:1px solid #64be6b;background:var(--ok-bg);color:#16892b}.status-green .result-status:before{content:'✓';background:var(--ok)}.status-yellow .result-status{border:1px solid #e6bd45;background:var(--warn-bg);color:#9a6c00}.status-yellow .result-status:before{content:'!';background:var(--warn)}.status-red .result-status{border:1px solid #e59b96;background:var(--bad-bg);color:#b62f29}.status-red .result-status:before{content:'!';background:var(--bad)}
.result-final-meter{height:8px;border-radius:99px;background:#e7edf4;overflow:hidden;margin-top:14px}.result-final-meter i{display:block;height:100%;border-radius:99px}.status-green .result-final-meter i{background:var(--ok)}.status-yellow .result-final-meter i{background:#f2b900}.status-red .result-final-meter i{background:var(--bad)}
.breakdown{grid-template-columns:1fr!important;gap:10px!important;margin-top:12px!important}.principle{position:relative;border-radius:15px!important;padding:13px 16px 13px 58px!important;min-height:82px}.principle:before{position:absolute;left:15px;top:50%;transform:translateY(-50%);font-size:28px}.principle:nth-child(1):before{content:'🏷️'}.principle:nth-child(2):before{content:'📍'}.principle:nth-child(3):before{content:'🧹'}.principle:nth-child(4):before{content:'📋'}.principle:nth-child(5):before{content:'🔄'}.principle span{font-size:15px!important;text-transform:uppercase}.principle small{font-size:11px!important}.principle b{font-size:24px!important}.principle.green{background:linear-gradient(90deg,#effaf0,#f8fff8)!important}.principle.yellow{background:linear-gradient(90deg,#fff8e4,#fffdf6)!important}.principle.red{background:linear-gradient(90deg,#fff0ef,#fff9f8)!important}
#bonus{background:linear-gradient(135deg,#0755a5,#064580)!important;border-radius:16px!important;padding:18px!important;font-size:16px!important}#rank{border-radius:16px!important;padding:17px!important}
.hang-card{border-radius:22px!important;padding:22px!important}.hang-definition{background:#fff!important;border:1px solid var(--line)!important;font-size:17px!important;margin-top:16px!important;padding:20px 18px!important}.hang-stage{grid-template-columns:1fr 54px!important}.hang-svg{height:285px!important;stroke:#17365a!important;stroke-width:5!important}.hang-svg .gallows{stroke:#243b57!important;stroke-width:7!important}.hang-errors i.bad{background:var(--bad)!important}.hang-keys button.right{background:#e7f8e8!important;color:#16892b!important;border-color:#82ce89!important;opacity:1!important}.hang-keys button.wrong{background:#fff0ef!important;color:var(--bad)!important;border-color:#eeaaa5!important;opacity:1!important}.outcome.fail .hang-svg{stroke:var(--bad)!important}.outcome.success .hang-svg{stroke:var(--ok)!important}.btn.danger{background:#df312d!important}
@media(max-width:800px){.screen{width:calc(100% - 22px)!important;padding-top:14px!important}.result-index strong{font-size:56px!important}.result-status{font-size:14px!important}.result-grid{grid-template-columns:1fr 1fr!important}.metric{padding:16px!important}.metric b{font-size:31px!important}.principle{padding-left:54px!important}.hang-card{padding:18px 14px!important}.hang-svg{height:245px!important}.hang-keys{grid-template-columns:repeat(7,1fr)!important}}
</style>`;

const runtime=`<script id="reference-ui-runtime-5s">
(()=>{
 const app=document.querySelector('#app');if(!app)return;
 function enhanceResult(){
  const hero=app.querySelector('.result-hero');if(!hero||hero.dataset.refUi)return;hero.dataset.refUi='1';
  const screen=hero.closest('.screen');if(!screen)return;
  const pct=Number((hero.querySelector('.result-index strong')?.textContent||'0').replace(/[^0-9.]/g,''))||0;
  const status=(hero.querySelector('.result-status')?.textContent||'').toUpperCase();
  let tone='red';
  if(status.includes('AFIANZADO')) tone='green';
  else if(status.includes('REFUERZO')) tone='yellow';
  else if(pct>=80) tone='green'; else if(pct>=60) tone='yellow';
  hero.classList.add('status-'+tone);
  const metrics=screen.querySelector('.result-grid');
  const banner=document.createElement('div');banner.className='result-celebration';
  banner.innerHTML=tone==='green'?'<div class="cup">🏆</div><div><strong>¡Excelente trabajo!</strong><span>Seguí así, cada acción cuenta.</span></div>':tone==='yellow'?'<div class="cup">💡</div><div><strong>Hay una buena base</strong><span>Reforzá los puntos señalados para afianzar las 5S.</span></div>':'<div class="cup">🎯</div><div><strong>Hay oportunidades importantes</strong><span>Revisá los principios con menor dominio antes de volver a intentar.</span></div>';
  screen.insertBefore(banner,screen.firstChild);if(metrics)screen.insertBefore(metrics,hero);
  const meter=document.createElement('div');meter.className='result-final-meter';meter.innerHTML='<i style="width:'+Math.min(100,Math.max(0,pct))+'%"></i>';hero.appendChild(meter);
 }
 const run=()=>enhanceResult();new MutationObserver(run).observe(app,{childList:true,subtree:true});run();
})();
</script>`;

html=html.replace('</head>',css+'</head>').replace('</body>',runtime+'</body>');
fs.writeFileSync(file,html);
console.log('Reference UI applied with semantic traffic-light result status.');
