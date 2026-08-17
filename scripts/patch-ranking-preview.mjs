import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');

html=html.replace(
  '<button id="photos" class="btn secondary">📷 GESTIONAR FOTOS</button>',
  '<button id="photos" class="btn secondary">📷 GESTIONAR FOTOS</button><button id="previewRank" class="btn secondary">👁 VISTA PREVIA DE RANKINGS</button>'
);
html=html.replace(
  "document.querySelector('#photos').onclick=adminPhotos;",
  "document.querySelector('#photos').onclick=adminPhotos;document.querySelector('#previewRank').onclick=previewRankings;"
);

const fn=`function previewRankings(){
 const sectorData=[
  ['Carnicería',92,7],['Lácteos',88,6],['Gerencia',86,3],['Panadería',82,8],['Administración',79,5],['Verdulería',76,6],['Salón',73,10],['Línea de Cajas',69,9],['Maestranza',64,3]
 ];
 const peopleData=[
  ['Lucía Fernández','Carnicería',98],['Martín Gómez','Lácteos',96],['Sofía Vargas','Panadería',94],['Nicolás Ruiz','Gerencia',92],['Carla Méndez','Administración',90],['Agustín Torres','Carnicería',88],['Valentina Soto','Verdulería',86],['Mateo Díaz','Salón',84],['Julieta Castro','Lácteos',82],['Tomás Romero','Línea de Cajas',80],['Camila López','Panadería',78],['Franco Silva','Salón',75],['Malena Ríos','Administración',72],['Bruno Pérez','Línea de Cajas',68],['Ana Morales','Maestranza',63]
 ];
 const tone=p=>p>=80?'green':p>=60?'yellow':'red';
 const medal=i=>i===0?'🥇':i===1?'🥈':i===2?'🥉':String(i+1);
 const sectorRows=sectorData.map((r,i)=>`<div class=\"preview-rank-row ${tone(r[1])}\"><div class=\"preview-pos\">${medal(i)}</div><div class=\"preview-main\"><strong>${esc(r[0])}</strong><span>${r[2]} participantes</span><i><em style=\"width:${r[1]}%\"></em></i></div><b>${r[1]}%</b></div>`).join('');
 const peopleRows=peopleData.map((r,i)=>`<div class=\"preview-rank-row people ${tone(r[2])}\"><div class=\"preview-pos\">${medal(i)}</div><div class=\"preview-main\"><strong>${esc(r[0])}</strong><span>${esc(r[1])}</span><i><em style=\"width:${r[2]}%\"></em></i></div><b>${r[2]}%</b></div>`).join('');
 app.innerHTML=`<section class=\"adminwrap preview-ranking-wrap\"><div class=\"admin-title\"><div><div class=\"eyebrow\">ADMINISTRADOR · VISTA PREVIA</div><h1>Rankings del Desafío 5S</h1><p class=\"small\">Datos ficticios para validar diseño. No modifican resultados ni estadísticas.</p></div><button id=\"menu\" class=\"btn secondary inline\">MENÚ</button></div><div class=\"preview-tabs\"><button class=\"active\" data-tab=\"sectors\">POR SECTORES</button><button data-tab=\"people\">POR PERSONAS</button></div><div id=\"previewSectors\" class=\"preview-panel\"><div class=\"preview-summary\"><div><span>SECTORES</span><b>9</b></div><div><span>PARTICIPANTES</span><b>57</b></div><div><span>PROMEDIO</span><b>79%</b></div></div><div class=\"preview-title-row\"><strong>Ranking global por sector</strong><span>Promedio del equipo</span></div>${sectorRows}</div><div id=\"previewPeople\" class=\"preview-panel hidden\"><div class=\"preview-summary\"><div><span>TOP</span><b>15</b></div><div><span>DOTACIÓN</span><b>57</b></div><div><span>MEJOR RESULTADO</span><b>98%</b></div></div><div class=\"preview-title-row\"><strong>Ranking individual</strong><span>Resultado personal</span></div>${peopleRows}<p class=\"preview-foot\">En producción puede mostrarse Top 15 y la posición propia de cada colaborador aunque quede fuera del Top.</p></div></section>`;
 document.querySelector('#menu').onclick=adminMenu;
 document.querySelectorAll('.preview-tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.preview-tabs button').forEach(x=>x.classList.toggle('active',x===b));document.querySelector('#previewSectors').classList.toggle('hidden',b.dataset.tab!=='sectors');document.querySelector('#previewPeople').classList.toggle('hidden',b.dataset.tab!=='people')});
}`;

html=html.replace('async function adminPhotos()',fn+'\nasync function adminPhotos()');

html=html.replace('</head>',`<style id="ranking-preview-admin">
.preview-ranking-wrap{max-width:820px!important}.preview-tabs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:20px 0 12px}.preview-tabs button{border:1px solid #cadceb;background:#fff;color:#0755a5;border-radius:12px;padding:13px;font-weight:900;cursor:pointer}.preview-tabs button.active{background:#0755a5;color:#fff;border-color:#0755a5}.preview-panel{background:#fff;border:1px solid #d7e3ef;border-radius:22px;padding:18px}.preview-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-bottom:18px}.preview-summary div{background:#eef5fb;border-radius:14px;padding:12px}.preview-summary span{display:block;color:#667d96;font-size:10px;font-weight:900}.preview-summary b{font-size:24px;color:#0755a5}.preview-title-row{display:flex;justify-content:space-between;align-items:end;gap:12px;margin:8px 2px 10px}.preview-title-row strong{font-size:18px}.preview-title-row span{font-size:11px;color:#667d96}.preview-rank-row{display:grid;grid-template-columns:46px 1fr 62px;gap:12px;align-items:center;border:1px solid #dde7f0;border-radius:15px;padding:12px;margin:8px 0}.preview-pos{font-size:21px;font-weight:900;text-align:center}.preview-main strong{display:block;font-size:15px}.preview-main span{display:block;font-size:11px;color:#667d96;margin:2px 0 8px}.preview-main i{display:block;height:6px;background:#edf2f6;border-radius:99px;overflow:hidden}.preview-main em{display:block;height:100%;border-radius:99px}.preview-rank-row>b{font-size:20px;text-align:right}.preview-rank-row.green{background:#f3fbf3;border-color:#b9dfbd}.preview-rank-row.green .preview-main em{background:#1b9c34}.preview-rank-row.green>b{color:#16892b}.preview-rank-row.yellow{background:#fffaf0;border-color:#ead28b}.preview-rank-row.yellow .preview-main em{background:#d79a00}.preview-rank-row.yellow>b{color:#9a6c00}.preview-rank-row.red{background:#fff3f2;border-color:#edb2ae}.preview-rank-row.red .preview-main em{background:#d93b35}.preview-rank-row.red>b{color:#b62f29}.preview-foot{font-size:11px;color:#667d96;text-align:center;margin:16px 4px 2px}@media(max-width:800px){.preview-ranking-wrap{width:calc(100% - 22px)!important;padding-top:20px!important}.preview-ranking-wrap .admin-title{align-items:flex-start}.preview-ranking-wrap .admin-title h1{font-size:28px;margin:6px 0}.preview-ranking-wrap .admin-title .inline{width:auto!important;min-width:0!important}.preview-panel{padding:12px}.preview-summary div{padding:10px 8px}.preview-summary b{font-size:20px}.preview-rank-row{grid-template-columns:38px 1fr 54px;padding:11px 9px;gap:8px}.preview-rank-row>b{font-size:18px}.preview-main strong{font-size:14px}}
</style></head>`);

fs.writeFileSync(file,html);
console.log('Admin ranking preview added');
