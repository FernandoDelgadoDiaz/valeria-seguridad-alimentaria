import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const file = 'index.html';
const sprite = 'assets/desafio5s/fotos-30.jpg';
const photosDir = 'assets/desafio5s/photos';

if (!fs.existsSync(sprite)) throw new Error(`Missing visual master: ${sprite}`);
fs.mkdirSync(photosDir, { recursive: true });
const source = () => sharp(sprite, { failOn: 'none', unlimited: true });
const meta = await source().metadata();
if (!meta.width || !meta.height) throw new Error('Invalid visual master dimensions');

const cols = 6, rows = 5;
const cellW = Math.floor(meta.width / cols), cellH = Math.floor(meta.height / rows);
if (cellW < 100 || cellH < 100) throw new Error(`Unexpected visual master size: ${meta.width}x${meta.height}`);

for (let i = 1; i <= 30; i++) {
  const n = String(i).padStart(2, '0');
  const target = path.join(photosDir, `v${n}.jpg`);
  if (fs.existsSync(target)) fs.unlinkSync(target);
}
for (let i = 0; i < 30; i++) {
  const col = i % cols, row = Math.floor(i / cols);
  const n = String(i + 1).padStart(2, '0');
  const target = path.join(photosDir, `v${n}.jpg`);
  await source().extract({ left: col * cellW, top: row * cellH, width: cellW, height: cellH }).jpeg({ quality: 92, chromaSubsampling: '4:4:4' }).toFile(target);
  const check = await sharp(target, { failOn: 'error' }).metadata();
  if (!check.width || !check.height || check.width !== cellW || check.height !== cellH) throw new Error(`Generated visual v${n}.jpg failed validation`);
}
const generated = fs.readdirSync(photosDir).filter(n => /^v\d{2}\.jpg$/.test(n));
if (generated.length !== 30) throw new Error(`Visual bank incomplete: ${generated.length}/30`);

let html = fs.readFileSync(file, 'utf8');
html = html.replace('Marcá todas las oportunidades de mejora que veas.', '¿Qué afirmaciones son correctas sobre esta situación?');
html = html.replace('Puede haber más de una correcta. Mirá la escena completa antes de seguir.', 'Marcá todas las que correspondan. Puede haber una o varias respuestas correctas.');
html = html.replace('<span>QUÉ DETECTÁS</span>', '<span>OBSERVÁ LA IMAGEN</span>');
html = html.replace('<span>QUÉ HARÍAS</span>', '<span>DECIDÍ</span>');
html = html.replace("${Object.entries(r.por_s||{}).map(([s,v])=>`<div><span>${s}</span><b>${v}/3</b></div>`).join('')}", "${Object.entries(r.por_s||{}).map(([s,v])=>{const names={S1:'Clasificar',S2:'Ordenar',S3:'Limpiar',S4:'Estandarizar',S5:'Sostener'};const pct=Math.round(Number(v||0)/3*100);return `<div class=\"principle\"><span>${s} · ${names[s]||''}</span><b>${pct}%</b><small>${v}/3 correctas</small></div>`}).join('')}");

const css = `
<style id="mobile-hotfix-5s">
.mark{display:none!important}.wordmark{gap:0!important}
.visual-photo{min-height:0!important;padding:12px!important;display:flex!important;align-items:flex-start!important;justify-content:center!important;background:#eef4fa!important}
.visual-photo img{display:block!important;width:100%!important;height:auto!important;max-height:76vh!important;object-fit:contain!important;background:#fff!important;border-radius:16px!important;box-shadow:0 12px 32px rgba(14,52,89,.14)!important}
.breakdown{display:grid!important;grid-template-columns:repeat(5,minmax(0,1fr))!important;gap:10px!important}.breakdown .principle{border:1px solid #d6e2ef;border-radius:14px;padding:14px 10px;text-align:left!important;background:#fbfdff}.breakdown .principle span{display:block;font-size:11px;font-weight:900;color:#657c95;line-height:1.25}.breakdown .principle b{display:block;font-size:24px;color:#0755a5;margin-top:5px}.breakdown .principle small{display:block;font-size:10px;color:#657c95;margin-top:2px}
@media (max-width:800px){
.top{height:64px;padding:0 14px}.wordmark b{font-size:13px}.top-title{font-size:12px}.home{display:block;background:#eef4fa}.hero{min-height:0!important;padding:28px 22px 74px!important;justify-content:flex-start!important}.hero h1{font-size:54px!important;line-height:.86!important;margin:10px 0 14px!important;max-width:280px}.hero-copy{font-size:15px!important;line-height:1.42!important;margin:0 0 16px!important;max-width:330px}.hero-meta{gap:7px!important;display:grid!important;grid-template-columns:1fr 1fr}.hero-chip{padding:7px 9px!important;font-size:10px!important;text-align:center}.hero-chip:last-child{grid-column:1/-1;justify-self:start}.home-right{padding:0 12px 26px!important}.card{margin-top:-42px!important;padding:22px 20px!important;border-radius:20px!important}.visual-photo{padding:10px!important}.visual-photo img{width:100%!important;max-height:none!important}.visual-content{padding:24px 20px 28px!important}.observe-title{font-size:19px!important;line-height:1.28!important}.observe-help{font-size:14px!important;line-height:1.4!important}.check{padding:14px 13px!important}.check span{font-size:15px!important;line-height:1.4!important}.breakdown{grid-template-columns:1fr!important;gap:8px!important}.breakdown .principle{display:grid;grid-template-columns:1fr auto;align-items:center;padding:12px 14px}.breakdown .principle b{font-size:22px;margin:0}.breakdown .principle small{grid-column:1/-1}
}
</style>`;
if (!html.includes('id="mobile-hotfix-5s"')) html = html.replace('</head>', `${css}\n</head>`);
else html = html.replace(/<style id="mobile-hotfix-5s">[\s\S]*?<\/style>/, css.trim());
fs.writeFileSync(file, html);
console.log(`5S build OK: 30 images generated (${cellW}x${cellH}); prompts and result UI patched.`);
