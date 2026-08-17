import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const file = 'index.html';
const spriteHQ = 'assets/desafio5s/fotos-30-hq.jpg';
const spriteLegacy = 'assets/desafio5s/fotos-30.jpg';
const sprite = fs.existsSync(spriteHQ) ? spriteHQ : spriteLegacy;
const photosDir = 'assets/desafio5s/photos';

// Un único master fotográfico genera V01..V30. El master HQ tiene prioridad absoluta.
if (fs.existsSync(sprite)) {
  fs.mkdirSync(photosDir, { recursive: true });
  const source = () => sharp(sprite, { failOn: 'none', unlimited: true });
  const meta = await source().metadata();
  if (meta.width && meta.height) {
    const cols = 6, rows = 5;
    const cellW = Math.floor(meta.width / cols), cellH = Math.floor(meta.height / rows);
    if (cellW >= 100 && cellH >= 100) {
      for (let i = 0; i < 30; i++) {
        const col = i % cols, row = Math.floor(i / cols);
        const n = String(i + 1).padStart(2, '0');
        const target = path.join(photosDir, `v${n}.jpg`);
        try {
          await source().extract({ left: col * cellW, top: row * cellH, width: cellW, height: cellH })
            .jpeg({ quality: 91, chromaSubsampling: '4:4:4', mozjpeg: true }).toFile(target);
          const check = await sharp(target, { failOn: 'error' }).metadata();
          if (!check.width || !check.height) throw new Error('invalid generated image');
        } catch (e) {
          console.warn(`Visual v${n} skipped: ${e.message}`);
        }
      }
    }
  }
}

let html = fs.readFileSync(file, 'utf8');

html = html.replace(
  'Marcá todas las oportunidades de mejora que veas.',
  '¿Qué afirmaciones son correctas sobre esta situación?'
);
html = html.replace(
  'Puede haber más de una correcta. Mirá la escena completa antes de seguir.',
  'Marcá todas las afirmaciones correctas. Puede haber una o varias.'
);
html = html.replace('<span>QUÉ DETECTÁS</span>', '<span>OBSERVÁ</span>');
html = html.replace('<span>QUÉ HARÍAS</span>', '<span>DECIDÍ</span>');

const oldBreakdown = "${Object.entries(r.por_s||{}).map(([s,v])=>`<div><span>${s}</span><b>${v}/3</b></div>`).join('')}";
const newBreakdown = "${Object.entries(r.por_s||{}).map(([s,v])=>{const names={S1:'Clasificar',S2:'Ordenar',S3:'Limpiar',S4:'Estandarizar',S5:'Sostener'};const item=(v&&typeof v==='object')?v:{correctas:Number(v||0),total:0,porcentaje:0};const correctas=Number(item.correctas||0);const total=Number(item.total||0);const pct=Number(item.porcentaje||0);return `<div class=\"principle\"><div><span>${s} · ${names[s]||''}</span><small>${correctas} de ${total} correctas</small></div><b>${pct.toFixed(0)}%</b></div>`}).join('')}";
html = html.replace(oldBreakdown, newBreakdown);

const css = `
<style id="mobile-hotfix-5s">
.mark{display:none!important}.wordmark{gap:0!important}
.visual-photo{min-height:0!important;padding:12px!important;display:flex!important;align-items:flex-start!important;justify-content:center!important;background:#eef4fa!important}
.visual-photo img{display:block!important;width:100%!important;height:auto!important;max-height:76vh!important;object-fit:contain!important;background:#fff!important;border-radius:16px!important;box-shadow:0 12px 32px rgba(14,52,89,.14)!important}
.observe-title{font-size:20px!important;line-height:1.28!important;margin-bottom:7px!important}
.observe-help{font-size:14px!important;line-height:1.45!important;margin-bottom:16px!important}
.decision h2{font-size:24px!important;line-height:1.28!important;margin:12px 0 18px!important}
.breakdown{display:grid!important;grid-template-columns:repeat(5,minmax(0,1fr))!important;gap:10px!important;margin-top:16px!important}
.breakdown .principle{border:1px solid #d6e2ef;border-radius:14px;padding:14px 12px;background:#fbfdff;display:flex;align-items:center;justify-content:space-between;gap:10px;text-align:left!important}
.breakdown .principle span{display:block;font-size:11px;font-weight:900;color:#526a84;line-height:1.25}
.breakdown .principle small{display:block;font-size:10px;color:#657c95;margin-top:4px}
.breakdown .principle b{display:block;font-size:23px;color:#0755a5;white-space:nowrap}
@media (max-width:800px){
.top{height:64px;padding:0 14px}.wordmark b{font-size:13px}.top-title{font-size:12px}
.home{display:block;background:#eef4fa}.hero{min-height:0!important;padding:28px 22px 74px!important;justify-content:flex-start!important}
.hero h1{font-size:54px!important;line-height:.86!important;margin:10px 0 14px!important;max-width:280px}.hero-copy{font-size:15px!important;line-height:1.42!important;margin:0 0 16px!important;max-width:330px}
.hero-meta{gap:7px!important;display:grid!important;grid-template-columns:1fr 1fr}.hero-chip{padding:7px 9px!important;font-size:10px!important;text-align:center}.hero-chip:last-child{grid-column:1/-1;justify-self:start}
.home-right{padding:0 12px 26px!important}.card{margin-top:-42px!important;padding:22px 20px!important;border-radius:20px!important}
.visual-photo{padding:10px!important}.visual-photo img{width:100%!important;max-height:none!important}.visual-content{padding:24px 20px 28px!important}
.step{margin-bottom:10px!important}.step span{letter-spacing:.06em!important}.observe-title{font-size:20px!important}.observe-help{font-size:14px!important}.check{padding:14px 13px!important}.check span{font-size:15px!important;line-height:1.4!important}
.decision{margin-top:6px!important;padding-top:24px!important}.decision h2{font-size:22px!important}
.breakdown{grid-template-columns:1fr!important;gap:8px!important}.breakdown .principle{padding:13px 14px!important}.breakdown .principle span{font-size:12px!important}.breakdown .principle small{font-size:11px!important}.breakdown .principle b{font-size:22px!important}
}
</style>`;

if (!html.includes('id="mobile-hotfix-5s"')) html = html.replace('</head>', `${css}\n</head>`);
else html = html.replace(/<style id="mobile-hotfix-5s">[\s\S]*?<\/style>/, css.trim());

fs.writeFileSync(file, html);
console.log(`5S build OK using ${sprite}; visual instructions and dynamic 5S breakdown applied.`);
