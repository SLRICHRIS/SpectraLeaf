'use strict';
// Browser-only inference. Captures never leave the browser in this interface.
let modelConfig=null, modelReport=null, modelSession=null, loadingSession=null;
let requiresExperimentalConsent=false;
let analysing=false, liveAnalysis=false, analysisTimer=null, analysisGeneration=0, observedImage=null;
const prettyClass={healthy:'Healthy',early_blight:'Early blight',late_blight:'Late blight'};
const percent=value=>`${(100*value).toFixed(1)}%`;
const modelMessage=message=>$('#inference-help').textContent=message;
function supportsInput(){return modelConfig&&$('#crop').value===modelConfig.crop&&band==='white'&&$('#confirm-tomato').checked&&(!requiresExperimentalConsent||$('#allow-experimental').checked);}
function updateAnalysisButtons(){
 const supported=supportsInput();
 $('#analyse-image').disabled=!supported||!draft||analysing;
 $('#analyse-live').disabled=!supported||!stream||analysing&&!liveAnalysis;
 if(liveAnalysis&&(!supported||!stream))stopLiveAnalysis();
 if(observedImage!==draft?.image){observedImage=draft?.image;analysisGeneration++;if(!liveAnalysis)$('#prediction-result').hidden=true;}
 if(!supported&&!liveAnalysis)modelMessage('Select Tomato and White illumination, then confirm that the image shows a tomato leaf under white light.');
}
function stopLiveAnalysis(){liveAnalysis=false;clearTimeout(analysisTimer);analysisGeneration++;$('#analyse-live').textContent='Start live analysis';}
function probabilities(logits){const maximum=Math.max(...logits),exp=Array.from(logits,v=>Math.exp(v-maximum)),sum=exp.reduce((a,b)=>a+b,0);return exp.map(v=>v/sum);}
async function loadModel(){
 if(modelSession)return modelSession;
 if(loadingSession)return loadingSession;
 loadingSession=(async()=>{
  modelMessage('Loading the trained model and checking its file checksum…');
  const response=await fetch('model/tomato-rgb.onnx');if(!response.ok)throw new Error('The model file could not be downloaded.');
  const buffer=await response.arrayBuffer();
  const checksum=globalThis.crypto?.subtle?Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join(''):sha256(buffer);
  if(checksum!==modelConfig.model_sha256)throw new Error('Model checksum mismatch. Refresh the site before trying again.');
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('vendor/',location.href).href;
  const session=await ort.InferenceSession.create(buffer,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  const fixtureResponse=await fetch('model/parity_fixture.json');if(!fixtureResponse.ok)throw new Error('Model parity fixture is missing.');
  const fixture=await fixtureResponse.json(),[height,width]=modelConfig.input_size;
  const tensor=new Float32Array(3*height*width);
  for(let ch=0;ch<3;ch++)tensor.fill((fixture.pixel_value*modelConfig.scale-modelConfig.mean[ch])/modelConfig.std[ch],ch*height*width,(ch+1)*height*width);
  const output=await session.run({image:new ort.Tensor('float32',tensor,[1,3,height,width])});
  const error=Math.max(...fixture.logits.map((v,i)=>Math.abs(v-output.logits.data[i])));
  if(!Number.isFinite(error)||error>fixture.tolerance){await session.release();throw new Error('Browser model parity check failed. Predictions are unavailable in this browser.');}
  modelSession=session;$('#model-status').textContent=requiresExperimentalConsent?'Experimental · runtime verified':'Weights & runtime verified';
  return session;
 })().catch(error=>{loadingSession=null;throw error;});
 return loadingSession;
}
function imageTensor(source){
 const [height,width]=modelConfig.input_size,c=document.createElement('canvas');c.width=width;c.height=height;
 const context=c.getContext('2d',{willReadFrequently:true});context.imageSmoothingEnabled=true;context.imageSmoothingQuality='low';context.drawImage(source,0,0,width,height);
 const pixels=context.getImageData(0,0,width,height).data,tensor=new Float32Array(3*height*width);
 for(let i=0;i<height*width;i++)for(let ch=0;ch<3;ch++)tensor[ch*height*width+i]=(pixels[i*4+ch]*modelConfig.scale-modelConfig.mean[ch])/modelConfig.std[ch];
 return new ort.Tensor('float32',tensor,[1,3,height,width]);
}
function showPrediction(values,source){
 const top=values.indexOf(Math.max(...values)),confident=values[top]>=modelConfig.abstention_threshold;
 const result=$('#prediction-result');result.hidden=false;
 result.innerHTML=`<div class="prediction-box ${confident?'':'uncertain'}"><div class="prediction-title"><b>${confident?`Model suggests: ${prettyClass[modelConfig.class_names[top]]}`:'Uncertain — no class accepted'}</b><small>${escapeHTML(source)}</small></div>${modelConfig.class_names.map((name,i)=>`<div class="prediction-row"><span>${prettyClass[name]}</span><div class="prob-track"><div class="prob-fill" style="width:${values[i]*100}%"></div></div><span>${percent(values[i])}</span></div>`).join('')}<p class="field-note">Research model scores, not a confirmed diagnosis. Other tomato diseases and unrelated images can receive confident incorrect predictions.</p></div>`;
}
async function analyse(source,description,generation){
 const session=await loadModel();if(generation!==analysisGeneration||!supportsInput())return;
 const input=imageTensor(source),output=await session.run({image:input});
 if(generation!==analysisGeneration||!supportsInput())return;
 const values=probabilities(output.logits.data);showPrediction(values,description);
 modelMessage('White-light RGB baseline. Check the separate-source evaluation below before interpreting a prediction.');
}
$('#analyse-image').addEventListener('click',async()=>{
 if(!supportsInput()||!draft||analysing)return;stopLiveAnalysis();analysing=true;updateAnalysisButtons();const generation=analysisGeneration;
 try{const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Could not read the captured image.'));image.src=draft.image;});await analyse(image,'Captured image',generation);}
 catch(error){modelMessage(error.message);$('#model-status').textContent='Inference unavailable';}
 finally{analysing=false;updateAnalysisButtons();}
});
$('#analyse-live').addEventListener('click',async()=>{
 if(liveAnalysis){stopLiveAnalysis();updateAnalysisButtons();return;}
 if(!supportsInput()||!stream)return;liveAnalysis=true;$('#analyse-live').textContent='Stop live analysis';
 async function tick(){
  if(!liveAnalysis||!stream||!supportsInput()){stopLiveAnalysis();return;}
  analysing=true;updateAnalysisButtons();const generation=analysisGeneration;
  try{if($('#video').videoWidth)await analyse($('#video'),'Live camera · latest analysed frame',generation);}
  catch(error){modelMessage(error.message);stopLiveAnalysis();}
  finally{analysing=false;updateAnalysisButtons();if(liveAnalysis)analysisTimer=setTimeout(tick,1200);}
 }
 tick();
});
document.addEventListener('spectraleaf-frame',updateAnalysisButtons);
$('#allow-experimental').addEventListener('change',()=>{analysisGeneration++;$('#prediction-result').hidden=true;updateAnalysisButtons();});
$('#confirm-tomato').addEventListener('change',()=>{analysisGeneration++;$('#prediction-result').hidden=true;updateAnalysisButtons();});
$('#crop').addEventListener('change',()=>{analysisGeneration++;$('#prediction-result').hidden=true;updateAnalysisButtons();});
$$('[data-band]').forEach(button=>button.addEventListener('click',()=>{analysisGeneration++;$('#prediction-result').hidden=true;updateAnalysisButtons();}));
window.addEventListener('hashchange',()=>{if(location.hash!=='#capture'&&location.hash!=='')stopLiveAnalysis();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopLiveAnalysis();});
$('#verify-model').addEventListener('click',async()=>{const button=$('#verify-model');button.disabled=true;button.textContent='Checking model…';try{if(!modelConfig)throw new Error('Model configuration is not loaded.');await loadModel();button.textContent='Checksum & runtime parity passed';toast('Model file checksum and browser inference parity passed. Field validation is a separate requirement.');}catch(error){button.textContent='Retry browser verification';toast(error.message);}finally{button.disabled=false;}});
async function initEvidence(){
 try{
  const responses=await Promise.all([fetch('model/model_config.json'),fetch('model/training_report.json')]);
  if(responses.some(r=>!r.ok))throw new Error('The versioned model files are unavailable.');
  [modelConfig,modelReport]=await Promise.all(responses.map(r=>r.json()));
  if(modelReport.model_sha256!==modelConfig.model_sha256)throw new Error('Model and evaluation versions do not match.');
  const test=modelReport.results.test,external=modelReport.results.external_test,q=modelReport.data_quality;
  $('#model-status').textContent='Research baseline · report loaded';
  $('#model-metrics').innerHTML=`<div class="model-metric"><b>${percent(test.accuracy)}</b><span>Held-out leaf accuracy · ${test.n} images</span></div><div class="model-metric"><b>${percent(external.accuracy)}</b><span>Separate-source accuracy · ${external.n} images</span></div><div class="model-metric"><b>${percent(external.macro_f1)}</b><span>Separate-source macro F1</span></div>`;
  const externalLimited=external.accuracy<.85||external.macro_f1<.8;requiresExperimentalConsent=externalLimited;if(externalLimited){$('#domain-warning').hidden=false;$('#domain-warning').textContent=`Separate-source accuracy is ${percent(external.accuracy)}. This model is not ready for reliable field diagnosis. Predictions are disabled until you opt into experimental exploration.`;$('#research-opt-in').hidden=false;$('#model-status').textContent='Separate-source validation below target';}
  $('#model-evidence').innerHTML=`<p><b>Model:</b> MobileNetV3-Small ImageNet encoder with a fitted multinomial logistic head. The encoder is frozen; four regularisation candidates were compared using validation data only. Temperature and the score threshold were selected on validation data, before test evaluation.</p><table class="evidence-table"><thead><tr><th>Partition</th><th>Images</th><th>Accuracy</th><th>Macro F1</th></tr></thead><tbody>${['train','validation','test','external_test'].map(s=>`<tr><td>${{train:'Training (original views)',validation:'Validation',test:'Held-out leaves',external_test:'Separate source · PlantDoc'}[s]}</td><td>${modelReport.results[s].n}</td><td>${percent(modelReport.results[s].accuracy)}</td><td>${percent(modelReport.results[s].macro_f1)}</td></tr>`).join('')}</tbody></table><p class="model-warning">${externalLimited?'Separate-source results are below the release target (85% accuracy and 80% macro F1). This is an experimental baseline; it is not validated for dependable field use.':'Separate-source evaluation is available, but your webcam and lighting setup still need independent validation.'} No wavelength-based disease model has been trained.</p><ul><li>${q.ingested} source images verified against pinned repository checksums; ${q.accepted} retained after curation.</li><li>${q.quarantined} quarantined; ${q.exact_duplicates_removed} exact duplicate removed. Unresolved leaf identities and conflicting labels are recorded.</li><li>60/20/20 group-stratified internal split. All views of a known leaf and detected duplicate group stay in one partition. Group, byte-hash, decoded-pixel-hash and known-leaf overlap checks passed.</li><li>PlantDoc was held out entirely from fitting, model selection and calibration. Its overlapping or conflicting duplicate images were excluded.</li><li>Plant IDs are unavailable; leaf-level grouping does not prove plant-level independence. Near-duplicate detection is conservative.</li><li>ONNX weights have a SHA-256 identity, and a numeric browser parity check runs before the first prediction. No images are sent to an inference server by this interface.</li></ul><p>Run <code>${escapeHTML(modelReport.run_id)}</code> · Training manifest <code>${escapeHTML(modelReport.manifest_sha256.slice(0,16))}…</code>. See the JSON report for per-class recall, confusion matrices, group-bootstrap confidence intervals and exact versions.</p>`;
  updateAnalysisButtons();
 }catch(error){$('#model-status').textContent='Model unavailable';modelMessage(error.message);$('#model-evidence').textContent=error.message;}
}
initEvidence();
