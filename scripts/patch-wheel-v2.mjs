import fs from 'node:fs';
const file='index.html';
let html=fs.readFileSync(file,'utf8');
const css=`<style id="wheel-v2-5s">
.wheel-legend{display:grid!important;grid-template-columns:1fr 1fr!important;gap:3px 7px!important;width:184px!important;padding:7px 8px!important;border:1px solid #ffffff2e!important;border-radius:10px!important;background:#003d7433!important;backdrop-filter:blur(2px)!important}
.wheel-legend b{position:relative!important;padding-left:9px!important;font-size:7.3px!important;line-height:1.15!important;color:#fff!important;font-weight:800!important;opacity:.98!important;text-align:left!important}
.wheel-legend b:before{content:'';position:absolute;left:0;top:.34em;width:5px;height:5px;border-radius:50%}.wheel-legend b:nth-child(1):before{background:#8f8f8f}.wheel-legend b:nth-child(2):before{background:#f7ca16}.wheel-legend b:nth-child(3):before{background:#3e86d6}.wheel-legend b:nth-child(4):before{background:#70bd44}.wheel-legend b:nth-child(5){grid-column:1/-1!important;text-align:center!important;padding-left:0!important}.wheel-legend b:nth-child(5):before{position:static!important;display:inline-block!important;margin-right:5px!important;background:#ff963b!important;vertical-align:middle!important}
@media(max-width:800px){.institutional-wheel{right:13px!important;top:38px!important;width:142px!important}.wheel-disc{width:98px!important;height:98px!important}.wheel-disc:after{inset:29px!important}.wheel-legend{width:142px!important;gap:3px 5px!important;padding:5px 6px!important;border-radius:8px!important}.wheel-legend b{font-size:6.2px!important;padding-left:8px!important}.wheel-legend b:before{width:4px!important;height:4px!important}}
</style>`;
html=html.replace('</head>',css+'</head>');
fs.writeFileSync(file,html);
console.log('Institutional wheel legend aligned and color-keyed.');
