(function(root){
'use strict';
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
function rng(seed){let s=seed>>>0;return ()=>{s=(1664525*s+1013904223)>>>0;return s/4294967296;};}
function plan({specimens,conditions,repeats=1,seed=1}){
 if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw new Error('Seed must be an integer from 0 to 4294967295.');
 if(!Number.isInteger(repeats)||repeats<1||repeats>3)throw new Error('Choose one to three technical rounds.');
 if(!specimens.length||specimens.length>60||specimens.some(s=>!s.plant||!s.specimen||s.plant.length>60||s.specimen.length>60)||new Set(specimens.map(s=>JSON.stringify([s.plant,s.specimen]))).size!==specimens.length)throw new Error('Enter 1–60 unique plant/specimen pairs, each ID at most 60 characters.');
 if(conditions.length<2||conditions.length>6||new Set(conditions.map(c=>c.nm)).size!==conditions.length||conditions.some(c=>!Number.isFinite(c.nm)||c.nm<350||c.nm>1100||![0,1,2].includes(c.channel)||!Number.isFinite(c.fwhm)||c.fwhm<=0||c.fwhm>300))throw new Error('Enter 2–6 distinct bands as wavelength:channel:FWHM, e.g. 450:B:20.');
 const rand=rng(seed),order=conditions.map((_,i)=>i);for(let i=order.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
 const slots=[];let block=0;
 for(let round=0;round<repeats;round++)for(const specimen of specimens){const cycle=Math.floor(block/order.length),shift=block%order.length;let row=Array.from({length:order.length},(_,i)=>order[(i+shift)%order.length]);if(cycle%2)row.reverse();row.forEach((condition,position)=>slots.push({id:`slot-${slots.length+1}`,...specimen,round:round+1,position:position+1,condition}));block++;}
 return {version:'spectraleaf.physical-plan.v1',specimens,conditions,repeats,seed,slots,endpoint:'Paired difference of dark/white ROI-mean camera-response ratios; specimen means then equal-weight plant means.',order_method:'Seeded condition permutation with cyclic positions; alternate complete cycles reversed. Partial cycles need not be balanced; no carryover balance claim.',qc_policy:'First acquisition per slot passing all frozen QC rules. Rejected attempts retained; no outcome-based replacement.',qc:{minimum_frames:8,maximum_clipped_fraction:.01,maximum_white_cv:.1,maximum_absolute_drift:.02,minimum_denominator_dn:10,minimum_valid_pixel_fraction:.95,ratio_min:0,ratio_max:1.2,reference_max_age_minutes:20,require_lock:true}};
}
function next(study,samples){return study.plan.slots.find(slot=>!study.skips?.some(s=>s.slot===slot.id)&&!samples.some(r=>r.study?.id===study.id&&r.study.slot===slot.id&&r.study.usable))||null;}
function assess(study,slot,row,dark,white,samples){
 const q=study.plan.qc,c=study.plan.conditions[slot.condition],p=row.acquisition.profile,reasons=[];
 if(!dark||!white)return {id:study.id,slot:slot.id,condition:slot.condition,round:slot.round,usable:false,reasons:['Matching dark and white references are missing or stale']};
 if(row.plant!==slot.plant||row.specimen!==slot.specimen||row.wavelength_nm!==c.nm||row.channel!==c.channel||row.fwhm_nm!==c.fwhm)reasons.push('Profile does not match the planned plant/specimen/band/channel/FWHM');
 if(!p.operator_locked)reasons.push('Operator declaration of exposure / gain / WB lock is missing');
 if(!row.response.valid||!Number.isFinite(row.response.global_ratio)||row.response.global_ratio<q.ratio_min||row.response.global_ratio>q.ratio_max)reasons.push('Invalid response ratio or flat-field reference');
 for(const [name,b] of [['sample',row.acquisition],['dark',dark],['white',white]]){
  if(b.n<q.minimum_frames||b.source!=='distinct live frames')reasons.push(`${name}: distinct live-frame minimum not met`);
  const v=b.channels[c.channel];if(v.clipped_fraction>q.maximum_clipped_fraction)reasons.push(`${name}: clipping exceeds limit`);
  if(Math.abs(v.drift_fraction)>q.maximum_absolute_drift)reasons.push(`${name}: drift exceeds limit`);
 }
 if(white.channels[c.channel].spatial_cv>q.maximum_white_cv)reasons.push('White-field CV exceeds limit');
 if(white.channels[c.channel].mean-dark.channels[c.channel].mean<q.minimum_denominator_dn||row.response.invalid_pixel_fraction>1-q.minimum_valid_pixel_fraction)reasons.push('Reference signal or valid-pixel minimum not met');
 if([dark,white].some(r=>{const age=Date.parse(row.acquisition.timestamp)-Date.parse(r.timestamp);return age<0||age>q.reference_max_age_minutes*60000;}))reasons.push('Reference age exceeds frozen limit');
 const anchor=samples.find(r=>r.study?.id===study.id&&r.study.usable&&r.plant===row.plant&&r.specimen===row.specimen);
 if(anchor&&(anchor.geometry!==row.geometry||JSON.stringify(anchor.acquisition.profile.settings)!==JSON.stringify(p.settings)))reasons.push('Geometry / camera settings differ from this specimen’s first accepted slot');
 return {id:study.id,slot:slot.id,condition:slot.condition,round:slot.round,usable:!reasons.length,reasons};
}
function summary(study,samples,baseline=0){
 const accepted=new Map();for(const row of samples)if(row.study?.id===study.id&&row.study.usable&&!accepted.has(row.study.slot))accepted.set(row.study.slot,row);
 return study.plan.conditions.map((condition,index)=>{
  const byPlant=new Map();let specimens=0,pairs=0;
  for(const s of study.plan.specimens){const values=[];for(let round=1;round<=study.plan.repeats;round++){
   const find=c=>study.plan.slots.find(slot=>slot.plant===s.plant&&slot.specimen===s.specimen&&slot.round===round&&slot.condition===c);
   const a=accepted.get(find(index)?.id),b=accepted.get(find(baseline)?.id);if(a&&b)values.push({a:a.response.global_ratio,b:b.response.global_ratio});
  }
  if(values.length!==study.plan.repeats)continue;specimens++;pairs+=values.length;
  if(!byPlant.has(s.plant))byPlant.set(s.plant,[]);byPlant.get(s.plant).push({a:mean(values.map(v=>v.a)),delta:mean(values.map(v=>v.a-v.b))});
  }
  const plantValues=[...byPlant.values()].map(v=>({a:mean(v.map(x=>x.a)),delta:mean(v.map(x=>x.delta))})),deltas=plantValues.map(v=>v.delta);let ci=null;
  if(deltas.length>=2){const rand=rng(study.plan.seed),boot=[];for(let i=0;i<1000;i++)boot.push(mean(deltas.map(()=>deltas[Math.floor(rand()*deltas.length)])));boot.sort((a,b)=>a-b);ci=[boot[24],boot[974]];}
  return {condition,index,plants:deltas.length,specimens,technical_pairs:pairs,mean:mean(plantValues.map(v=>v.a)),delta:mean(deltas),ci,exploratory:deltas.length<5};
 });
}
root.SLStudy={plan,next,assess,summary};if(typeof module!=='undefined'&&module.exports)module.exports=root.SLStudy;
})(typeof window!=='undefined'?window:globalThis);
