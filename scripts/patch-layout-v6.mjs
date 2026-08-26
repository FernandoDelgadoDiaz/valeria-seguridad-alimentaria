import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="layout-v6-5s">[\s\S]*?<\/style>/g,'');

const css=`<style id="layout-v6-5s">
/* V6.1 · Mobile vertical balance. Presentation only. */
@media(max-width:800px) and (min-height:700px){
  /* Intro de cada S: use the viewport instead of leaving a large empty area below. */
  .standard-screen{
    min-height:calc(100svh - 82px)!important;
    padding-top:14px!important;
    padding-bottom:18px!important;
    display:flex!important;
    align-items:flex-start!important;
  }
  .standard-card{
    min-height:min(600px,calc(100svh - 116px))!important;
    display:flex!important;
    flex-direction:column!important;
  }
  .standard-list{margin-top:4px!important}
  .standard-note{margin-top:auto!important}
  .standard-card .btn{margin-top:14px!important}

  /* Preguntas sólo texto: fill the usable viewport, not a fixed-height card. */
  .screen:has(.question-shell:not(:has(.visual-layout))){
    min-height:calc(100svh - 82px)!important;
    display:flex!important;
    flex-direction:column!important;
    padding-bottom:18px!important;
  }
  .screen:has(.question-shell:not(:has(.visual-layout))) .progress-wrap{
    flex:0 0 auto!important;
  }
  .question-shell:not(:has(.visual-layout)){
    flex:1 1 auto!important;
    min-height:0!important;
    display:flex!important;
    flex-direction:column!important;
  }
  .question-shell:not(:has(.visual-layout)) .question-head{
    flex:0 0 auto!important;
  }
  .question-shell:not(:has(.visual-layout)) .question-body{
    flex:1 1 auto!important;
    display:flex!important;
    flex-direction:column!important;
    padding-bottom:16px!important;
  }
  .question-shell:not(:has(.visual-layout)) .question-body h2{
    margin-bottom:10px!important;
  }
  .question-shell:not(:has(.visual-layout)) .options{
    flex:1 1 auto!important;
    display:grid!important;
    align-content:center!important;
    width:100%!important;
    margin:10px 0 14px!important;
  }
  .question-shell:not(:has(.visual-layout)) .nextbar{
    margin-top:auto!important;
    flex:0 0 auto!important;
  }
}

/* On shorter phones, keep natural content height to avoid forced scrolling. */
@media(max-width:800px) and (max-height:699px){
  .standard-card,.question-shell:not(:has(.visual-layout)){min-height:0!important}
  .question-shell:not(:has(.visual-layout)){flex:none!important}
}
</style>`;

html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Layout v6.1 applied: text-only questions now fill the usable mobile viewport.');
