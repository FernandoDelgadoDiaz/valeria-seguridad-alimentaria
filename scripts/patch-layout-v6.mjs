import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="layout-v6-5s">[\s\S]*?<\/style>/g,'');
html=html.replace(/<script id="layout-v6-runtime-5s">[\s\S]*?<\/script>/g,'');

const css=`<style id="layout-v6-5s">
/* V6.2 · Mobile vertical balance using explicit screen markers + dynamic viewport. */
@media(max-width:800px){
  /* Qué esperamos: compact content, intentionally centered in the usable viewport. */
  .standard-screen{
    min-height:calc(100dvh - 80px)!important;
    padding-top:14px!important;
    padding-bottom:max(18px,env(safe-area-inset-bottom))!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
  }
  .standard-card{
    min-height:0!important;
    height:auto!important;
    display:block!important;
  }
  .standard-list{margin-top:4px!important}
  .standard-note{margin-top:16px!important}
  .standard-card .btn{margin-top:14px!important}

  /* Preguntas sólo texto: progress + card fill the real mobile viewport. */
  .text-question-screen{
    min-height:calc(100dvh - 80px)!important;
    padding-top:12px!important;
    padding-bottom:max(16px,env(safe-area-inset-bottom))!important;
    display:grid!important;
    grid-template-rows:auto minmax(0,1fr)!important;
    gap:12px!important;
  }
  .text-question-screen>.progress-wrap{
    margin:0!important;
    align-self:start!important;
  }
  .text-question-screen .question-shell{
    margin-top:0!important;
    min-height:0!important;
    height:100%!important;
    display:flex!important;
    flex-direction:column!important;
  }
  .text-question-screen .question-head{
    flex:0 0 auto!important;
  }
  .text-question-screen .question-body{
    flex:1 1 auto!important;
    min-height:0!important;
    display:grid!important;
    grid-template-rows:auto minmax(0,1fr) auto!important;
    padding-bottom:16px!important;
  }
  .text-question-screen .question-body h2{
    margin:10px 0 12px!important;
  }
  .text-question-screen .options{
    width:100%!important;
    align-self:center!important;
    align-content:center!important;
    display:grid!important;
    gap:10px!important;
    margin:12px 0!important;
  }
  .text-question-screen .nextbar{
    align-self:end!important;
    margin-top:8px!important;
  }
}

@media(max-width:800px) and (max-height:699px){
  .standard-screen{min-height:0!important;display:block!important}
  .text-question-screen{min-height:0!important;display:block!important}
  .text-question-screen .question-shell{height:auto!important}
  .text-question-screen .question-body{display:block!important}
}
</style>`;

const runtime=`<script id="layout-v6-runtime-5s">
(()=>{
  const app=document.querySelector('#app');
  if(!app)return;
  const markScreens=()=>{
    app.querySelectorAll('.screen').forEach(screen=>{
      screen.classList.remove('text-question-screen');
      const shell=screen.querySelector('.question-shell');
      if(shell && !shell.querySelector('.visual-layout')) screen.classList.add('text-question-screen');
    });
  };
  new MutationObserver(markScreens).observe(app,{childList:true,subtree:true});
  markScreens();
})();
</script>`;

html=html.replace('</head>',css+'</head>').replace('</body>',runtime+'</body>');
fs.writeFileSync(file,html);
console.log('Layout v6.2 applied: explicit mobile text-question marker and dynamic viewport balance.');
