import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="visual-card-v2">[\s\S]*?<\/style>/g,'');
const css=`<style id="visual-card-v2">
@media(max-width:800px){
  .question-shell:has(.visual-layout){background:#fff!important;border:1px solid #dbe6f1!important;border-radius:24px!important;overflow:hidden!important;box-shadow:0 14px 34px rgba(20,54,91,.10)!important;padding:0!important}
  .question-shell:has(.visual-layout) .question-head{padding:18px 16px 14px!important;text-align:center!important;background:#fff!important}
  .question-shell:has(.visual-layout) .pill.visual{display:inline-flex!important;margin:0 auto!important;justify-content:center!important;text-align:center!important;max-width:100%!important}
  .question-shell:has(.visual-layout) .visual-layout{display:block!important;background:#fff!important}
  .question-shell:has(.visual-layout) .visual-photo{margin:0 14px!important;padding:0!important;background:#fff!important;border-radius:18px!important;overflow:hidden!important}
  .question-shell:has(.visual-layout) .visual-photo img{border-radius:18px!important;width:100%!important;display:block!important}
  .question-shell:has(.visual-layout) .visual-photo:before{left:14px!important;top:14px!important}
  .question-shell:has(.visual-layout) .visual-content{padding:24px 20px 28px!important;background:#fff!important}
}
</style>`;
html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Visual challenge card v2 applied');
