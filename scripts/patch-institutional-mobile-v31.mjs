import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="institutional-mobile-v31-5s">[\s\S]*?<\/style>/g,'');

const css=`<style id="institutional-mobile-v31-5s">
/* Institutional pass v3.1 — mobile-only tuning. No logic or functionality changes. */
@media(max-width:800px){
  /* Home: preserve the approved composition, restore legibility and avoid collisions. */
  .hero{min-height:236px!important;padding:20px 20px 60px!important}
  .institutional-wheel{right:14px!important;top:28px!important;width:120px!important;gap:5px!important}
  .wheel-disc{width:72px!important;height:72px!important}
  .wheel-disc:after{inset:21px!important}
  .wheel-legend{width:120px!important;padding:5px 6px!important;gap:3px 5px!important}
  .wheel-legend b{font-size:7px!important;line-height:1.12!important;padding-left:8px!important}
  .wheel-legend b:before{width:4px!important;height:4px!important}
  .hero-meta{bottom:12px!important;gap:6px!important}
  .hero-chip{font-size:9.5px!important;padding:6px 7px!important;line-height:1.1!important}

  /* Visual challenges: complete image, but with a firm mobile height ceiling. */
  .question-shell:has(.visual-layout) .visual-photo{max-height:min(44vh,360px)!important}
  .question-shell:has(.visual-layout) .visual-photo img{max-height:min(44vh,360px)!important}
  .question-shell:has(.visual-layout) .visual-content{padding-top:15px!important}
}
</style>`;

html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Institutional mobile v3.1 applied; presentation-only tuning.');
