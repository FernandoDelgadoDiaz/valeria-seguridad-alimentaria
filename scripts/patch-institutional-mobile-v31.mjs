import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="institutional-mobile-v31-5s">[\s\S]*?<\/style>/g,'');
html=html.replace(/<script id="institutional-mobile-v31-runtime-5s">[\s\S]*?<\/script>/g,'');

const css=`<style id="institutional-mobile-v31-5s">
/* Institutional pass v3.3 — simplified mobile hero. Presentation only. */
@media(max-width:800px){
  /* Home: one clear message, one visual anchor, no explanatory mini-infographic. */
  .hero{min-height:228px!important;padding:20px 20px 58px!important}
  .hero-kicker{font-size:8.5px!important;letter-spacing:.18em!important}
  .hero h1{font-size:40px!important;line-height:.86!important;margin:7px 0 9px!important;max-width:56%!important}
  .hero-copy{font-size:12.8px!important;line-height:1.38!important;margin:0!important;max-width:61%!important}
  .institutional-wheel{right:18px!important;top:43px!important;width:92px!important;display:block!important}
  .wheel-disc{width:88px!important;height:88px!important;margin:0 auto!important;box-shadow:0 0 0 5px #ffffff1c!important}
  .wheel-disc:after{inset:26px!important;border-width:4px!important}
  .wheel-disc span{font-size:14px!important}
  .wheel-legend{display:none!important}
  .hero-meta{left:20px!important;right:20px!important;bottom:12px!important;gap:6px!important}
  .hero-chip{font-size:9.5px!important;padding:6px 7px!important;line-height:1.1!important}

  /* Test-mode banner must stay fully visible below the sticky corporate header. */
  .testbar.on{position:sticky!important;top:52px!important;z-index:19!important;padding:5px 10px!important;font-size:8.5px!important;line-height:1.15!important}

  /* Visual challenges: keep current image size so operational details remain readable. */
  .question-shell:has(.visual-layout) .visual-photo{max-height:min(34vh,280px)!important}
  .question-shell:has(.visual-layout) .visual-photo img{max-height:min(34vh,280px)!important}
  .question-shell:has(.visual-layout) .visual-content{padding-top:14px!important}

  /* Decision step: compact without sacrificing readability. */
  .question-shell:has(.visual-layout) .decision h2{font-size:19px!important;line-height:1.22!important;letter-spacing:-.01em!important;margin:8px 0 12px!important}
  .question-shell:has(.visual-layout) .decision{padding-top:13px!important}
  .question-shell:has(.visual-layout) .decision .options{gap:7px!important}
  .question-shell:has(.visual-layout) .decision .option{padding:10px 11px!important;min-height:50px!important}
  .question-shell:has(.visual-layout) .decision .option.selected{padding:9px 10px!important}
  .question-shell:has(.visual-layout) .decision .option span{font-size:15px!important;line-height:1.3!important}
}
</style>`;

html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Institutional mobile v3.3 applied; hero simplified; logic untouched.');
