import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="institutional-polish-5s">[\s\S]*?<\/style>/g,'');
const css=`<style id="institutional-polish-5s">
/* Final institutional polish: visual-only overrides. No application logic is changed. */
:root{--institutional-radius-card:12px;--institutional-radius-control:9px;--institutional-shadow:0 4px 14px rgba(20,54,91,.055)}

/* 1. Geometry and surfaces */
.card,.question-shell,.progress-wrap,.panel,.metric,.rankbox,.kpi,.loginbox,.hang-card,.preview-panel,.personrow,.errorrow{border-radius:var(--institutional-radius-card)!important;box-shadow:var(--institutional-shadow)!important}
.btn,.field input,.field select,.option,.check{border-radius:var(--institutional-radius-control)!important}
.question-shell,.panel,.metric,.rankbox,.kpi,.hang-card,.preview-panel,.personrow,.errorrow{border-color:#d9e2ea!important}

/* 2. Internal header: present, quieter and more operational */
.top{height:64px!important;background:#fff!important;backdrop-filter:none!important;box-shadow:none!important}
.wordmark b{font-weight:800!important;letter-spacing:.075em!important}.top-title{font-weight:800!important;letter-spacing:.045em!important}

/* 3. Results: one clear hero, solid corporate blue, restrained celebration */
.result-celebration{background:#0755a5!important;border-radius:12px!important;box-shadow:none!important;padding:18px 20px!important}
.result-celebration .cup{font-size:38px!important}.result-celebration strong{font-size:21px!important}.result-celebration span{font-size:13px!important}
.result-hero{border-radius:12px!important;box-shadow:none!important;padding:20px!important}
.result-status{border-radius:8px!important}
.metric{padding:16px!important}.metric:after{width:32px!important;height:32px!important;font-size:17px!important}

/* 4. S1-S5: white operational rows; semantic colour is an accent, never the whole surface */
.breakdown{gap:8px!important}.principle{background:#fff!important;border:1px solid #d9e2ea!important;border-radius:10px!important;box-shadow:none!important;min-height:74px!important;padding:12px 14px 12px 54px!important;overflow:hidden}
.principle:after{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:#8da0b3}
.principle.green:after{background:#1b9c34}.principle.yellow:after{background:#d79a00}.principle.red:after{background:#d93b35}
.principle.green,.principle.yellow,.principle.red{background:#fff!important}
.principle span{font-size:13px!important}.principle b{font-size:22px!important}

/* 5. Rankings: predominantly white, aligned and corporate */
.rankbox{box-shadow:none!important}.rankrow{background:#fff!important;border-top:1px solid #e5ebf0!important}
.preview-rank-row{background:#fff!important;border-radius:10px!important;box-shadow:none!important;border-color:#d9e2ea!important;position:relative;overflow:hidden}
.preview-rank-row:before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:#8da0b3}
.preview-rank-row.green:before{background:#1b9c34}.preview-rank-row.yellow:before{background:#d79a00}.preview-rank-row.red:before{background:#d93b35}
.preview-summary div{border-radius:9px!important;background:#f3f6f9!important}
.preview-tabs button{border-radius:9px!important}

/* 6. Administrator: denser, flatter, clearly operational */
.adminwrap{padding-top:26px!important}.kpis{gap:8px!important}.kpi{border-radius:8px!important;padding:14px!important;box-shadow:none!important}.kpi b{font-size:26px!important}.grid2{gap:10px!important}.rows>div,.people-row{padding-top:9px!important;padding-bottom:9px!important}.drillwrap{padding-top:24px!important}.personrow,.errorrow{border-radius:8px!important;padding:12px!important;box-shadow:none!important}.sgrid div{border-radius:8px!important}.drillnote{border-radius:8px!important}

@media(max-width:800px){
 .top{height:58px!important}.screen{padding-top:16px!important}.adminwrap{padding-top:20px!important}
 .result-celebration{gap:13px!important;padding:15px 16px!important}.result-celebration .cup{font-size:32px!important}.result-celebration strong{font-size:19px!important}
 .result-hero{padding:17px!important}.result-index strong{font-size:52px!important}
 .principle{padding-left:50px!important;min-height:70px!important}
 .kpi{padding:12px!important}.kpi b{font-size:24px!important}
}
</style>`;
html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Institutional visual polish applied; logic untouched.');
