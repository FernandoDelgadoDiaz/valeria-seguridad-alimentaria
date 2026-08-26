import fs from 'node:fs';

const file='index.html';
let html=fs.readFileSync(file,'utf8');
html=html.replace(/<style id="approved-ui-v7-5s">[\s\S]*?<\/style>/g,'');
html=html.replace(/<script id="approved-ui-v7-runtime-5s">[\s\S]*?<\/script>/g,'');

const css=`<style id="approved-ui-v7-5s">
/* V7.1 · Restore approved result/hangman presentation with semantic result icons. */

.result-celebration{
  display:flex!important;
  align-items:center!important;
  gap:18px!important;
  background:linear-gradient(135deg,#0755a5,#063f7f)!important;
  color:#fff!important;
  border-radius:22px!important;
  padding:22px 24px!important;
  margin-bottom:14px!important;
  box-shadow:0 14px 30px rgba(7,85,165,.18)!important;
}
.result-celebration .cup{
  display:grid!important;
  place-items:center!important;
  flex:0 0 54px!important;
  width:54px!important;
  height:54px!important;
  font-size:0!important;
  line-height:1!important;
}
.result-celebration .cup:before{display:block;line-height:1}
.result-celebration.tone-green .cup:before{content:'🏆';font-size:54px;filter:drop-shadow(0 5px 7px #0003)}
.result-celebration.tone-yellow .cup:before{content:'💡';font-size:48px;filter:drop-shadow(0 5px 7px #0002)}
.result-celebration.tone-red .cup:before{
  content:'!';
  width:42px;
  height:42px;
  border:3px solid #fff;
  border-radius:50%;
  display:grid;
  place-items:center;
  font-size:28px;
  font-weight:900;
  line-height:1;
}
.result-celebration strong{display:block!important;font-size:25px!important;line-height:1.05!important}
.result-celebration span{display:block!important;margin-top:6px!important;font-size:14px!important;opacity:.95!important}
.result-grid{gap:12px!important;margin-top:0!important}
.metric{border-radius:18px!important;padding:20px!important;position:relative!important;overflow:hidden!important}
.metric b{font-size:36px!important}
.metric:after{display:grid!important;position:absolute!important;right:15px!important;bottom:14px!important;width:38px!important;height:38px!important;border-radius:50%!important;place-items:center!important;color:#fff!important;font-weight:900!important;font-size:21px!important}
.metric:first-child:after{content:'✓'!important;background:#1b9c34!important}
.metric:nth-child(2):after{content:'★'!important;background:#f4bf00!important}
.result-hero{background:#fff!important;color:#10233f!important;border:1px solid #d7e3ef!important;border-radius:20px!important;padding:22px!important;margin-top:12px!important;box-shadow:none!important}
.result-hero:before{content:'RESULTADO FINAL'!important;display:block!important;color:#0755a5!important;font-size:12px!important;font-weight:900!important;letter-spacing:.08em!important;margin-bottom:10px!important}
.result-index small{display:none!important}.result-index strong{font-size:64px!important;line-height:.95!important}.result-top{align-items:center!important}
.result-hero .small,.result-hero p{color:#10233f!important}
.breakdown{grid-template-columns:1fr!important;gap:10px!important;margin-top:12px!important}
.principle{position:relative!important;border-radius:15px!important;padding:13px 16px 13px 58px!important;min-height:82px!important}
.principle:after{display:none!important}
.principle:before{position:absolute!important;left:15px!important;top:50%!important;transform:translateY(-50%)!important;width:auto!important;height:auto!important;background:transparent!important;border-radius:0!important;display:block!important;font-size:28px!important;color:inherit!important}
.principle:nth-child(1):before{content:'🏷️'!important}.principle:nth-child(2):before{content:'📍'!important}.principle:nth-child(3):before{content:'🧹'!important}.principle:nth-child(4):before{content:'📋'!important}.principle:nth-child(5):before{content:'🔄'!important}
.principle span{font-size:15px!important;text-transform:uppercase!important}.principle small{font-size:11px!important}.principle b{font-size:24px!important}
.principle.green{background:linear-gradient(90deg,#effaf0,#f8fff8)!important}.principle.yellow{background:linear-gradient(90deg,#fff8e4,#fffdf6)!important}.principle.red{background:linear-gradient(90deg,#fff0ef,#fff9f8)!important}
#bonus{background:linear-gradient(135deg,#0755a5,#064580)!important;border-radius:16px!important;padding:18px!important;font-size:16px!important;box-shadow:0 10px 20px rgba(7,85,165,.14)!important}

/* Hangman result: approved expressive outcome card. */
.outcome{max-width:720px!important}
.outcome.fail,.outcome.success{background:#fff!important;border-top:0!important}
.outcome-banner{border-radius:18px!important;padding:20px 18px!important;text-align:center!important;margin-top:12px!important}
.outcome.fail .outcome-banner{background:linear-gradient(180deg,#fff0ef,#fff8f7)!important;border:1px solid #efaaa5!important}
.outcome.success .outcome-banner{background:linear-gradient(180deg,#eaf9ed,#f8fff9)!important;border:1px solid #83ce8b!important}
.outcome-face{display:block!important;font-size:0!important;margin:0 0 6px!important}
.outcome.fail .outcome-face:before{content:'😢';font-size:72px;line-height:1;filter:drop-shadow(0 7px 12px #d93b3533)}
.outcome.success .outcome-face:before{content:'😄';font-size:72px;line-height:1;filter:drop-shadow(0 7px 12px #1b9c3433)}
.outcome.fail .eyebrow{color:#d93b35!important}.outcome.success .eyebrow{color:#1b9c34!important}
.outcome h1{font-size:25px!important}.outcome.fail h1 strong{font-size:38px!important}
.outcome-figure{width:260px!important;max-width:80%!important;margin:8px auto 12px!important}
.outcome-figure .hang-svg{height:240px!important;max-height:240px!important}
.outcome.fail .hang-svg{stroke:#d93b35!important}.outcome.success .hang-svg{stroke:#1b9c34!important}
.btn.danger{background:linear-gradient(135deg,#df312d,#c91e1e)!important;border-radius:12px!important;color:#fff!important}

@media(max-width:800px){
 .result-celebration{padding:18px!important}.result-celebration .cup{flex-basis:44px!important;width:44px!important;height:44px!important}.result-celebration.tone-green .cup:before{font-size:44px!important}.result-celebration.tone-yellow .cup:before{font-size:40px!important}.result-celebration.tone-red .cup:before{width:36px!important;height:36px!important;font-size:24px!important}
 .result-celebration strong{font-size:21px!important}
 .result-index strong{font-size:56px!important}.metric{padding:16px!important}.metric b{font-size:31px!important}.principle{padding-left:54px!important}
 .outcome-banner{padding:18px 14px!important}.outcome.fail .outcome-face:before,.outcome.success .outcome-face:before{font-size:60px!important}.outcome-figure{width:230px!important}.outcome-figure .hang-svg{height:210px!important;max-height:210px!important}
}
</style>`;

const runtime=`<script id="approved-ui-v7-runtime-5s">
(()=>{
 const app=document.querySelector('#app');if(!app)return;
 const restore=()=>{
   const hero=app.querySelector('.result-hero');
   const banner=app.querySelector('.result-celebration');
   if(hero&&banner){
     const status=(hero.querySelector('.result-status')?.textContent||'').toUpperCase();
     banner.classList.remove('tone-green','tone-yellow','tone-red');
     if(status.includes('AFIANZADO')) banner.classList.add('tone-green');
     else if(status.includes('REFUERZO')) banner.classList.add('tone-yellow');
     else banner.classList.add('tone-red');
   }

   const out=app.querySelector('.outcome');
   if(out&&!out.dataset.approvedV7){
     out.dataset.approvedV7='1';
     if(!out.querySelector(':scope > .outcome-banner')){
       const wrap=document.createElement('div');wrap.className='outcome-banner';
       while(out.firstChild)wrap.appendChild(out.firstChild);
       out.appendChild(wrap);
     }
   }
 };
 new MutationObserver(restore).observe(app,{childList:true,subtree:true});
 restore();
})();
</script>`;

html=html.replace('</head>',css+'</head>').replace('</body>',runtime+'</body>');
fs.writeFileSync(file,html);
console.log('Approved UI v7.1 applied: semantic result icon + approved hangman outcome; logic unchanged.');
