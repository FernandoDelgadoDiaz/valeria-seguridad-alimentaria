import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="institutional-polish-5s">[\s\S]*?<\/style>/g,'');
html=html.replace(/<script id="institutional-polish-runtime-5s">[\s\S]*?<\/script>/g,'');

const css=`<style id="institutional-polish-5s">
/* Institutional pass v2 — deliberately visible, visual-only. Application logic is untouched. */
:root{--la-blue:#0755a5;--la-blue-dark:#0b3e78;--la-blue-light:#e7f0f8;--la-bg:#f5f7f9;--la-line:#dde2e7;--la-ink:#20262d;--la-muted:#65717d;--institutional-radius-card:12px;--institutional-radius-control:9px}
html,body{background:var(--la-bg)!important}body{color:var(--la-ink)!important}
.top{height:64px!important;background:#fff!important;backdrop-filter:none!important;box-shadow:none!important;border-bottom:1px solid var(--la-line)!important}
.wordmark b{font-weight:800!important;letter-spacing:.075em!important}.top-title{font-weight:800!important;letter-spacing:.045em!important;color:var(--la-blue)!important}
.home{background:var(--la-bg)!important}.hero{background:var(--la-blue)!important;box-shadow:none!important}.hero-chip{border-radius:18px!important;background:transparent!important;border-color:rgba(255,255,255,.42)!important}.home-right{background:var(--la-bg)!important}.card{border-radius:12px!important;box-shadow:none!important;border:1px solid var(--la-line)!important}.field input,.field select{border-radius:8px!important;border-color:#cfd9e2!important;background:#fff!important}
.progress-wrap,.question-shell,.panel,.metric,.rankbox,.kpi,.loginbox,.hang-card,.preview-panel,.personrow,.errorrow{border-radius:12px!important;box-shadow:none!important;border-color:var(--la-line)!important}.btn,.option,.check{border-radius:9px!important}.bar,.result-final-meter,.principle-bar{border-radius:99px!important}.question-shell{box-shadow:none!important}.option{background:#fff!important;border-color:#cfd9e2!important}.option.selected{background:var(--la-blue-light)!important;border-color:var(--la-blue)!important}.visual-photo img{border-radius:8px!important}.visual-photo{background:#eef2f5!important}
.result-celebration{background:var(--la-blue)!important;border-radius:12px!important;box-shadow:none!important;padding:17px 20px!important}.result-celebration .cup{display:none!important}.result-celebration strong{font-size:21px!important}.result-celebration span{font-size:14px!important}.result-hero{border-radius:12px!important;box-shadow:none!important;padding:20px!important;border-color:var(--la-line)!important}.result-status{border-radius:8px!important}.metric{padding:16px!important}.metric:after{display:none!important}
.breakdown{gap:8px!important}.principle{position:relative;background:#fff!important;border:1px solid var(--la-line)!important;border-radius:10px!important;box-shadow:none!important;min-height:74px!important;padding:12px 14px 12px 58px!important;overflow:hidden}.principle:after{content:''!important;position:absolute!important;left:0!important;top:0!important;bottom:0!important;width:4px!important;background:#8da0b3!important}.principle.green:after{background:#1b9c34!important}.principle.yellow:after{background:#d79a00!important}.principle.red:after{background:#d93b35!important}.principle.green,.principle.yellow,.principle.red{background:#fff!important}.principle:before{position:absolute!important;left:14px!important;top:50%!important;transform:translateY(-50%)!important;width:30px!important;height:30px!important;border-radius:6px!important;background:#eef3f7!important;color:var(--la-blue)!important;display:grid!important;place-items:center!important;font-size:11px!important;font-weight:900!important;line-height:1!important}.principle:nth-child(1):before{content:'S1'!important}.principle:nth-child(2):before{content:'S2'!important}.principle:nth-child(3):before{content:'S3'!important}.principle:nth-child(4):before{content:'S4'!important}.principle:nth-child(5):before{content:'S5'!important}.principle span{font-size:13px!important}.principle b{font-size:22px!important}
.btn.primary,#bonus{background:var(--la-blue)!important;color:#fff!important}.btn.secondary{background:var(--la-blue-light)!important;color:var(--la-blue)!important}.btn.danger{background:var(--la-blue)!important;color:#fff!important}
.rankbox{box-shadow:none!important}.rankrow{background:#fff!important;border-top:1px solid #e5ebf0!important}.preview-rank-row{background:#fff!important;border-radius:10px!important;box-shadow:none!important;border-color:var(--la-line)!important;position:relative;overflow:hidden}.preview-rank-row:before{content:''!important;position:absolute;left:0;top:0;bottom:0;width:4px;background:#8da0b3!important}.preview-rank-row.green:before{background:#1b9c34!important}.preview-rank-row.yellow:before{background:#d79a00!important}.preview-rank-row.red:before{background:#d93b35!important}.preview-summary div{border-radius:9px!important;background:#f3f6f9!important}.preview-tabs button{border-radius:9px!important}
.adminwrap{padding-top:26px!important}.kpis{gap:8px!important}.kpi{border-radius:8px!important;padding:14px!important;box-shadow:none!important}.kpi b{font-size:26px!important}.grid2{gap:10px!important}.rows>div,.people-row{padding-top:9px!important;padding-bottom:9px!important}.drillwrap{padding-top:24px!important}.personrow,.errorrow{border-radius:8px!important;padding:12px!important;box-shadow:none!important}.sgrid div,.drillnote{border-radius:8px!important}
.hang-card{border-radius:12px!important;box-shadow:none!important}.hang-definition{border-radius:9px!important}.outcome-face{display:none!important}.hang-title{gap:6px!important}.hang-svg{stroke-width:4!important}.hang-svg .gallows{stroke-width:6!important}.outcome.fail{background:#fff!important}.outcome.fail .hang-svg{stroke:#d93b35!important}
@media(max-width:800px){.top{height:58px!important}.screen{padding-top:16px!important}.hero{margin:12px 20px 0!important;padding:28px 24px 30px!important;border-radius:12px!important}.home-right{padding:16px 20px 26px!important}.card{margin-top:0!important;padding:24px 20px!important;border-radius:12px!important}.question-shell{border-radius:12px!important}.progress-wrap{border-radius:10px!important}.result-celebration{gap:0!important;padding:15px 16px!important}.result-celebration strong{font-size:19px!important}.result-hero{padding:17px!important}.result-index strong{font-size:52px!important}.principle{padding-left:54px!important;min-height:70px!important}.kpi{padding:12px!important}.kpi b{font-size:24px!important}.adminwrap{padding-top:20px!important}.hang-card{padding:18px 16px!important}}
</style>`;

const runtime=`<script id="institutional-polish-runtime-5s">
(()=>{
 const clean=s=>s.replace(/[🎮🔎📷👁️👁🎯🏆💡😢😊]/gu,'').replace(/\\s{2,}/g,' ').trim();
 function normalize(root=document.querySelector('#app')){
  if(!root)return;
  const w=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);
  const nodes=[];while(w.nextNode())nodes.push(w.currentNode);
  for(const n of nodes){const v=n.nodeValue||'';const c=clean(v);if(c!==v.trim())n.nodeValue=(/^\\s/.test(v)?' ':'')+c+(/\\s$/.test(v)?' ':'')}
 }
 const app=document.querySelector('#app');if(!app)return;
 new MutationObserver(()=>normalize(app)).observe(app,{childList:true,subtree:true});normalize(app);
})();
</script>`;

html=html.replace('</head>',css+'</head>').replace('</body>',runtime+'</body>');
fs.writeFileSync(file,html);
console.log('Institutional visual polish v2 applied; logic untouched.');
