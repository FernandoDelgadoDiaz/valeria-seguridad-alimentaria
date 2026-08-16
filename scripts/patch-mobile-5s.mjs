import fs from 'node:fs';

const file = 'index.html';
let html = fs.readFileSync(file, 'utf8');

const css = `
<style id="mobile-hotfix-5s">
.mark{display:none!important}
.wordmark{gap:0!important}
.sprite-photo{width:100%;aspect-ratio:3/4;background-image:url('/assets/desafio5s/fotos-30.jpg?v=5');background-repeat:no-repeat;background-size:600% 500%;border-radius:16px;box-shadow:0 12px 32px rgba(14,52,89,.14);background-color:#fff}

@media (max-width:800px){
  .top{height:64px;padding:0 14px}
  .wordmark{gap:0!important}
  .wordmark b{font-size:13px;letter-spacing:.08em}
  .top-title{font-size:12px;letter-spacing:.05em}

  .home{display:block;background:#eef4fa}
  .hero{min-height:0!important;padding:28px 22px 74px!important;justify-content:flex-start!important}
  .hero:after{width:300px!important;height:105px!important;right:-120px!important;bottom:-52px!important}
  .tape.orange,.tape.green{display:none!important}
  .tape.pink{width:34px;height:8px;right:34px;bottom:28px}

  .hero-kicker{font-size:10px!important;line-height:1.35;letter-spacing:.18em!important;margin:0 0 10px!important;opacity:.95}
  .hero h1{font-size:54px!important;line-height:.86!important;margin:10px 0 14px!important;max-width:280px}
  .hero-copy{font-size:15px!important;line-height:1.42!important;margin:0 0 16px!important;max-width:330px}
  .hero-meta{gap:7px!important;display:grid!important;grid-template-columns:1fr 1fr}
  .hero-chip{padding:7px 9px!important;font-size:10px!important;line-height:1.2;text-align:center}
  .hero-chip:last-child{grid-column:1/-1;justify-self:start}

  .home-right{padding:0 12px 26px!important}
  .card{margin-top:-42px!important;padding:22px 20px!important;border-radius:20px!important}
  .card .eyebrow,.eyebrow{font-size:10px!important}
  .card h2{font-size:26px!important;margin:7px 0 16px!important}
  .field{margin:11px 0!important}
  .field input,.field select{padding:13px 12px!important}
  .btn{padding:14px!important}

  .visual-photo{min-height:0!important;padding:12px!important;align-items:flex-start!important}
  .visual-photo img{display:block!important;width:100%!important;height:auto!important;object-fit:contain!important;background:#fff!important;border-radius:16px!important}
  .sprite-photo{width:100%!important;aspect-ratio:3/4!important}
}
@media (max-width:390px){
  .hero{padding:24px 18px 68px!important}
  .hero h1{font-size:49px!important}
  .hero-copy{font-size:14px!important}
  .hero-meta{grid-template-columns:1fr 1fr}
  .card{padding:20px 17px!important}
}
</style>`;

if (!html.includes('id="mobile-hotfix-5s"')) {
  html = html.replace('</head>', `${css}\n</head>`);
} else {
  html = html.replace(/<style id="mobile-hotfix-5s">[\s\S]*?<\/style>/, css.trim());
}

// Safari-safe rendering: visual questions from V01..V30 are cropped directly
// from the JPG sprite, with no SVG/data-image indirection.
html = html.replace(
  'const img=imageMap[current.imagen_url]||current.imagen_url;app.innerHTML=',
  'const img=imageMap[current.imagen_url]||current.imagen_url;const vm=String(current.imagen_url||"").match(/v(\\d{2})\\.svg$/i);let visualMedia;if(vm){const n=Number(vm[1])-1,col=n%6,row=Math.floor(n/6);visualMedia=`<div class="sprite-photo" style="background-position:${col*20}% ${row*25}%"></div>`}else{visualMedia=`<img src="${img}" alt="Situación real de la sucursal" loading="eager">`};app.innerHTML='
);
html = html.replace(
  '<div class="visual-photo"><img src="${img}" alt="Situación real de la sucursal" loading="eager"></div>',
  '<div class="visual-photo">${visualMedia}</div>'
);

fs.writeFileSync(file, html);
console.log('5S mobile hotfix applied: direct JPG sprite rendering enabled.');
