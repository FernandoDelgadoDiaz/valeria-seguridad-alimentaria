import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const file = 'index.html';
const sprite = 'assets/desafio5s/fotos-30.jpg';
const photosDir = 'assets/desafio5s/photos';

// Definitive visual-bank build: turn the 6x5 master sheet into 30 real JPG files.
// The browser receives an ordinary <img> URL; no SVG, data URI or CSS crop is used.
if (!fs.existsSync(sprite)) {
  throw new Error(`Missing visual master: ${sprite}`);
}

fs.mkdirSync(photosDir, { recursive: true });
const meta = await sharp(sprite).metadata();
if (!meta.width || !meta.height) throw new Error('Invalid visual master dimensions');

const cols = 6;
const rows = 5;
const cellW = Math.floor(meta.width / cols);
const cellH = Math.floor(meta.height / rows);
if (cellW < 100 || cellH < 100) throw new Error(`Unexpected visual master size: ${meta.width}x${meta.height}`);

for (let i = 0; i < 30; i++) {
  const col = i % cols;
  const row = Math.floor(i / cols);
  const n = String(i + 1).padStart(2, '0');
  await sharp(sprite)
    .extract({ left: col * cellW, top: row * cellH, width: cellW, height: cellH })
    .jpeg({ quality: 88, mozjpeg: true })
    .toFile(path.join(photosDir, `v${n}.jpg`));
}

let html = fs.readFileSync(file, 'utf8');

const css = `
<style id="mobile-hotfix-5s">
.mark{display:none!important}
.wordmark{gap:0!important}
.visual-photo{min-height:0!important;padding:12px!important;display:flex!important;align-items:flex-start!important;justify-content:center!important;background:#eef4fa!important}
.visual-photo img{display:block!important;width:100%!important;height:auto!important;max-height:76vh!important;object-fit:contain!important;background:#fff!important;border-radius:16px!important;box-shadow:0 12px 32px rgba(14,52,89,.14)!important}

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
  .visual-photo{padding:10px!important}
  .visual-photo img{width:100%!important;max-height:none!important}
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

fs.writeFileSync(file, html);
console.log(`5S build OK: generated 30 standalone JPG files (${cellW}x${cellH}) and applied mobile UI fixes.`);
