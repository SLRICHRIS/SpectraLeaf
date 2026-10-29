'use strict';
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const STORE = 'spectraleaf.session.v1';
const MAX_CAPTURES = 20;
let captures = [], draft = null, stream = null, band = 'white', toastTimer, cameraRequest = 0;
const escapeHTML = (v) => String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uid = () => typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `capture-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,12)}`;
function toast(message) { const el=$('#toast'); el.textContent=message; el.hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true,6500); }
function validDataURL(value) { return typeof value==='string' && value.length<2500000 && /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/.test(value); }
function cleanMeta(r) {
 if (!r || !validDataURL(r.image)) throw new Error('A capture has an invalid or oversized image.');
 if (typeof r.specimen!=='string' || !r.specimen.trim()) throw new Error('Each capture needs a specimen ID.');
 const wave = r.wavelength_nm===null ? null : Number(r.wavelength_nm);
 if (wave!==null && (!Number.isInteger(wave)||wave<350||wave>1100)) throw new Error('Wavelength must be an integer from 350 to 1100 nm.');
 return {id:uid(),specimen:r.specimen.trim().slice(0,60),crop:String(r.crop||'Other').slice(0,60),condition:String(r.condition||'unlabelled').slice(0,80),wavelength_nm:wave,notes:String(r.notes||'').slice(0,500),created_at:!isNaN(Date.parse(r.created_at))?new Date(r.created_at).toISOString():new Date().toISOString(),image:r.image,source:String(r.source||'import').slice(0,30)};
}
try { const saved=JSON.parse(localStorage.getItem(STORE)||'[]'); if(Array.isArray(saved)) captures=saved.slice(0,MAX_CAPTURES).map(r=>({...cleanMeta(r),width:Number(r.width)||0,height:Number(r.height)||0,metrics:r.metrics||{r:0,g:0,b:0,brightness:0,clipped_pct:0}})); }
catch { captures=[]; toast('The saved session could not be loaded. Import an exported session to recover it.'); }
function persist(next) { try {localStorage.setItem(STORE,JSON.stringify(next)); captures=next; render(); return true;} catch {toast('Browser storage is full or unavailable. Export your session, remove older captures, then try again.');return false;} }
function renderStage() {
 const live=Boolean(stream);
 $('#video').hidden=!live;$('#live-badge').hidden=!live;$('#snapshot').hidden=!live;
 $('#empty-state').hidden=live||Boolean(draft);$('#preview').hidden=live||!draft;$('#image-overlay').hidden=live||!draft;
 $('#captured-frame').hidden=!live||!draft;
 $('#source-status').textContent=live?'Camera live':draft?'Image ready':'Awaiting image';
 $('#camera-toggle').innerHTML=`<svg><use href="#camera"/></svg>${live?'Stop live camera':'Start live camera'}`;
 $('#save-capture').disabled=!draft;document.dispatchEvent(new CustomEvent('spectraleaf-frame')); 
 if(draft){$('#preview').src=draft.image;$('#captured-thumb').src=draft.image;$('#image-dimensions').textContent=`${draft.width} × ${draft.height} px`;$('#captured-info').textContent=`${draft.width} × ${draft.height} px · ready to save`;}
}
function stopCamera() {cameraRequest++;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;$('#video').srcObject=null;renderStage();}
async function refreshCameras(){
 if(!navigator.mediaDevices?.enumerateDevices)return;
 try{const cameras=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');const active=stream?.getVideoTracks()[0]?.getSettings?.().deviceId;$('#camera-select').innerHTML='<option value="">Default webcam</option>'+cameras.map((d,i)=>`<option value="${escapeHTML(d.deviceId)}">${escapeHTML(d.label||`Camera ${i+1}`)}</option>`).join('');if(cameras.some(d=>d.deviceId===active))$('#camera-select').value=active;$('#camera-select').disabled=!cameras.length;}catch{/* Default camera remains available even when enumeration is restricted. */}
}
async function cameraToggle(){
 if(stream){stopCamera();return;}
 if(!navigator.mediaDevices?.getUserMedia){toast('Webcam access is unavailable here. Open the site in a browser over HTTPS, or upload an image.');return;}
 const button=$('#camera-toggle'),request=++cameraRequest;button.disabled=true;button.textContent='Opening camera…';
 const deviceId=$('#camera-select').value;
 try {
  const opened=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:960},...(deviceId?{deviceId:{exact:deviceId}}:{})},audio:false});
  if(request!==cameraRequest||document.hidden||!['','#capture','#optics','#evaluation'].includes(location.hash)){opened.getTracks().forEach(t=>t.stop());return;}
  stream=opened;$('#video').srcObject=opened;await $('#video').play();
  if(request!==cameraRequest)return;
  renderStage();await refreshCameras();
  opened.getVideoTracks().forEach(t=>t.addEventListener('ended',()=>{if(stream===opened){stopCamera();toast('The camera disconnected. Reconnect it and start the live view again.');}}));
 }catch(e){if(request===cameraRequest){stopCamera();toast(e.name==='NotAllowedError'?'Camera permission was denied. Allow camera access in your browser or upload an image.':e.name==='NotFoundError'?'No webcam was found. Connect a camera, or upload a plant image.':'Could not open the camera. Check it is connected and not in use, or upload an image.');}}
 finally{button.disabled=false;if(!stream)renderStage();}
}
function measure(canvas) {
 const ctx=canvas.getContext('2d',{willReadFrequently:true});const {data}=ctx.getImageData(0,0,canvas.width,canvas.height);let r=0,g=0,b=0,clipped=0;const n=data.length/4;
 for(let i=0;i<data.length;i+=4){r+=data[i];g+=data[i+1];b+=data[i+2];if(data[i]>=250||data[i+1]>=250||data[i+2]>=250)clipped++;}
 return {r:Math.round(r/n*10)/10,g:Math.round(g/n*10)/10,b:Math.round(b/n*10)/10,brightness:Math.round((.2126*r+.7152*g+.0722*b)/n*10)/10,clipped_pct:Math.round(clipped/n*1000)/10};
}
function makeDraft(source,sourceType){
 const w=source.videoWidth||source.naturalWidth,h=source.videoHeight||source.naturalHeight;if(!w||!h)throw new Error('The image is not ready. Try again.');
 const scale=Math.min(1,960/Math.max(w,h));const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(w*scale));canvas.height=Math.max(1,Math.round(h*scale));canvas.getContext('2d').drawImage(source,0,0,canvas.width,canvas.height);
 draft={image:canvas.toDataURL('image/jpeg',.86),width:canvas.width,height:canvas.height,metrics:measure(canvas),source:sourceType};
 if(sourceType!=='webcam')stopCamera();renderStage();
}
async function upload(file){
 if(!file)return;if(!['image/jpeg','image/png','image/webp'].includes(file.type)){toast('Choose a JPG, PNG or WebP image.');return;}
 if(file.size>12*1024*1024){toast('This image is larger than 12 MB. Resize it and try again.');return;}
 const url=URL.createObjectURL(file), img=new Image();
 try{await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error('This image could not be decoded.'));img.src=url;});makeDraft(img,'upload');}
 catch(e){toast(e.message);}finally{URL.revokeObjectURL(url);$('#file-input').value='';}
}
function wavelength(){if(band==='white')return null;const value=Number(band==='custom'?$('#custom-nm').value:band);if(!Number.isInteger(value)||value<350||value>1100)throw new Error('Enter a centre wavelength from 350 to 1100 nm.');return value;}
function label(r){return r.wavelength_nm===null?'White light':`${r.wavelength_nm} nm`;}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportJSON(){if(!captures.length){toast('Add a capture before exporting.');return;}download(new Blob([JSON.stringify({format:'spectraleaf.session.v1',exported_at:new Date().toISOString(),measurement_note:'Whole-frame uncalibrated camera RGB values, not reflectance.',captures},null,2)],{type:'application/json'}),`spectraleaf-session-${new Date().toISOString().slice(0,10)}.json`);toast('Session exported with its images and metadata.');}
function thumb(r,large=false){const src=validDataURL(r.image)?r.image:'';return `<article class="capture-thumb"><img src="${src}" alt="${escapeHTML(r.specimen)} under ${escapeHTML(label(r))}"><div class="thumb-body"><b>${escapeHTML(r.specimen)} <span class="pill">${escapeHTML(label(r))}</span></b><div class="thumb-meta">${escapeHTML(r.crop)} · ${escapeHTML(r.condition)}${large?` · ${r.width} × ${r.height} px`:''}</div>${large?`<p class="field-note">${escapeHTML(r.notes||'No observation notes.')}</p><p class="field-note">Brightness: ${Number(r.metrics.brightness).toFixed(1)}/255 · Pixels with ≥1 channel near saturation: ${Number(r.metrics.clipped_pct).toFixed(1)}%</p>`:''}<div class="thumb-actions"><a href="${src}" download="${escapeHTML(r.specimen.replace(/[^a-zA-Z0-9_-]/g,'_'))}-${r.wavelength_nm||'white'}.jpg">Download image</a><button data-remove="${r.id}" aria-label="Remove ${escapeHTML(r.specimen)} capture"><svg><use href="#trash"/></svg></button></div></div></article>`;}
function render(){
 $('#capture-count').textContent=captures.length;
 $('#capture-gallery').innerHTML=captures.length?captures.map(r=>thumb(r)).join(''):'<div class="empty-gallery"><svg><use href="#grid"/></svg>Your saved captures will appear here.</div>';
 const selected=$('#compare-specimen').value;const specimens=[...new Set(captures.map(r=>r.specimen))];$('#compare-specimen').innerHTML=specimens.length?specimens.map(s=>`<option>${escapeHTML(s)}</option>`).join(''):'<option value="">No specimens yet</option>';
 if(specimens.includes(selected))$('#compare-specimen').value=selected;renderComparison();
}
function renderComparison(){
 const selected=$('#compare-specimen').value,items=captures.filter(r=>r.specimen===selected);
 $('#comparison-gallery').innerHTML=items.length?items.map(r=>thumb(r,true)).join(''):'<div class="empty-gallery">Add images in Capture studio to start comparing.</div>';
 const colors={r:'#b98271',g:'#7c9b66',b:'#819cb8'};
 $('#channel-chart').innerHTML=items.length?'<div class="chart-legend">'+Object.entries(colors).map(([k,c])=>`<span><i style="background:${c}"></i>${{r:'Red',g:'Green',b:'Blue'}[k]} channel</span>`).join('')+'</div>'+items.map(r=>`<div class="channel-row"><div class="channel-label">${label(r)}<small>${new Date(r.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</small></div><div class="channel-bars">${['r','g','b'].map(k=>{const v=Number(r.metrics[k]);return `<div class="channel-bar" style="background:${colors[k]};width:${Math.max(3,Math.min(100,v/255*100))}%" title="${k.toUpperCase()}: ${v}">${v.toFixed(1)}</div>`;}).join('')}</div></div>`).join(''):'<p class="field-note">No measurements yet.</p>';
}
function route(){let name=location.hash.slice(1)||'capture';if(!['capture','compare','datasets','guide','optics','evaluation'].includes(name))name='capture';$$('.page').forEach(p=>p.hidden=p.id!==name);$$('[data-route]').forEach(a=>a.classList.toggle('active',a.dataset.route===name));$('#breadcrumb').textContent=$(`[data-route="${name}"]`).textContent.trim();if(!['capture','optics','evaluation'].includes(name))stopCamera();document.dispatchEvent(new CustomEvent('spectraleaf-route'));window.scrollTo(0,0);}
$('#camera-toggle').addEventListener('click',cameraToggle);
$('#camera-select').addEventListener('change',()=>{if(stream){stopCamera();cameraToggle();}});
$('#snapshot').addEventListener('click',()=>{try{makeDraft($('#video'),'webcam');}catch(e){toast(e.message);}});
$('#upload-button').onclick=()=>$('#file-input').click();$('#file-input').onchange=e=>upload(e.target.files[0]);
$('#drop-zone').addEventListener('click',()=>{if(!stream&&!draft)$('#file-input').click();});
$('#drop-zone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file-input').click();}});
$('#drop-zone').addEventListener('dragover',e=>{e.preventDefault();$('#drop-zone').classList.add('drag-over');});$('#drop-zone').addEventListener('dragleave',()=>$('#drop-zone').classList.remove('drag-over'));$('#drop-zone').addEventListener('drop',e=>{e.preventDefault();$('#drop-zone').classList.remove('drag-over');upload(e.dataTransfer.files[0]);});
$$('[data-band]').forEach(button=>button.addEventListener('click',()=>{band=button.dataset.band;$$('[data-band]').forEach(b=>b.classList.toggle('selected',b===button));$('#custom-wrap').hidden=band!=='custom';}));
$('#capture-form').addEventListener('submit',e=>{e.preventDefault();if(!draft){toast('Take or upload a photograph first.');return;}if(captures.length>=MAX_CAPTURES){toast('This session holds up to 20 captures. Export it, then clear it to start a new session.');return;}try{const specimen=$('#specimen').value.trim();if(!specimen)throw new Error('Enter a specimen ID.');const record={...draft,id:uid(),specimen,crop:$('#crop').value,condition:$('#condition').value,wavelength_nm:wavelength(),notes:$('#notes').value.trim(),created_at:new Date().toISOString()};if(persist([...captures,record])){draft=null;$('#preview').removeAttribute('src');$('#captured-thumb').removeAttribute('src');renderStage();toast(`${specimen} · ${label(record)} added. Change your physical light, then take the next image.`);}}catch(error){toast(error.message);}});
document.addEventListener('click',e=>{const button=e.target.closest('[data-remove]');if(button){const record=captures.find(r=>r.id===button.dataset.remove);if(record&&confirm(`Remove ${record.specimen} · ${label(record)} from this session? Export first if you need a backup.`)){persist(captures.filter(r=>r.id!==record.id));toast('Capture removed.');}}});
$$('.export-json').forEach(b=>b.onclick=exportJSON);
$('#export-csv').onclick=()=>{if(!captures.length){toast('Add a capture before exporting.');return;}const cols=['id','specimen','crop','condition','wavelength_nm','created_at','source','width','height','notes','mean_r','mean_g','mean_b','brightness','clipped_pct'];const quote=v=>'"'+String(v??'').replace(/"/g,'""').replace(/^[=+@\-]/,"'$&")+'"';const rows=captures.map(r=>({...r,mean_r:r.metrics.r,mean_g:r.metrics.g,mean_b:r.metrics.b,...r.metrics}));download(new Blob(['\ufeff'+[cols.join(','),...rows.map(r=>cols.map(c=>quote(r[c])).join(','))].join('\r\n')],{type:'text/csv;charset=utf-8'}),'spectraleaf-manifest.csv');};
$('#import-button').onclick=()=>$('#import-input').click();
$('#import-input').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{if(f.size>35*1024*1024)throw new Error('Session file must be under 35 MB.');const data=JSON.parse(await f.text());if(data.format!=='spectraleaf.session.v1'||!Array.isArray(data.captures))throw new Error('Choose a SpectraLeaf session JSON export.');if(data.captures.length+captures.length>MAX_CAPTURES)throw new Error('Import would exceed 20 captures. Export and clear the current session first.');const imported=[];for(const raw of data.captures){const r=cleanMeta(raw),img=new Image();await new Promise((resolve,reject)=>{img.onload=resolve;img.onerror=()=>reject(new Error('An imported image could not be decoded.'));img.src=r.image;});const s=Math.min(1,960/Math.max(img.naturalWidth,img.naturalHeight));const c=document.createElement('canvas');c.width=Math.max(1,Math.round(img.naturalWidth*s));c.height=Math.max(1,Math.round(img.naturalHeight*s));c.getContext('2d').drawImage(img,0,0,c.width,c.height);imported.push({...r,image:c.toDataURL('image/jpeg',.86),width:c.width,height:c.height,metrics:measure(c)});}if(persist([...captures,...imported]))toast(`${imported.length} captures imported.`);}catch(err){toast(err.message||'Could not import this file.');}finally{e.target.value='';}};
$('#clear-session').onclick=()=>{if(!captures.length){toast('Your session is already empty.');return;}if(confirm('Clear all saved captures from this browser? Export a session backup first.')){if(persist([]))toast('Session cleared.');}};
$('#compare-specimen').onchange=renderComparison;window.addEventListener('hashchange',route);window.addEventListener('beforeunload',()=>{if(stream)stream.getTracks().forEach(t=>t.stop());});document.addEventListener('visibilitychange',()=>{if(document.hidden&&stream){stopCamera();$('#source-status').textContent=draft?'Image ready':'Camera stopped';}});render();route();
