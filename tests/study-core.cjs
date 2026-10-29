const assert=require('node:assert/strict'),S=require('../web/study-core.js');
const conditions=[{nm:450,channel:2,fwhm:20},{nm:530,channel:1,fwhm:20},{nm:660,channel:0,fwhm:20}],specimens=Array.from({length:6},(_,i)=>({plant:'p'+i,specimen:'l1'}));
const config={conditions,specimens,repeats:2,seed:42},p=S.plan(config),study={id:'study',plan:p};
assert.deepEqual(p,S.plan(config));assert.ok([1,9,991,123456789].some(seed=>JSON.stringify(p.slots)!==JSON.stringify(S.plan({...config,seed}).slots))); 
for(const c of [0,1,2])for(const position of [1,2,3])assert.equal(p.slots.filter(s=>s.condition===c&&s.position===position).length,4);
assert.throws(()=>S.plan({...config,conditions:[conditions[0],conditions[0]]}));assert.throws(()=>S.plan({...config,specimens:[specimens[0],specimens[0]]}));assert.throws(()=>S.plan({...config,repeats:0}));
function measurement(slot,value){const c=p.conditions[slot.condition];return {plant:slot.plant,specimen:slot.specimen,wavelength_nm:c.nm,channel:c.channel,fwhm_nm:c.fwhm,geometry:'g',response:{valid:true,global_ratio:value,invalid_pixel_fraction:0},acquisition:{n:12,source:'distinct live frames',timestamp:'2026-10-30T12:01:00Z',profile:{operator_locked:true,settings:{exposureTime:100}},channels:Array.from({length:3},()=>({clipped_fraction:0,drift_fraction:0}))}};}
const ref=(mean)=>({n:12,source:'distinct live frames',timestamp:'2026-10-30T12:00:00Z',channels:Array.from({length:3},()=>({mean,clipped_fraction:0,drift_fraction:0,spatial_cv:.01}))}),dark=ref(10),white=ref(150),rows=[];
const first=p.slots[0],bad=measurement(first,.5);bad.acquisition.n=1;bad.study=S.assess(study,first,bad,dark,white,rows);assert.equal(bad.study.usable,false);rows.push(bad);assert.equal(S.next(study,rows).id,first.id);
for(const slot of p.slots){const row=measurement(slot,.5+.1*slot.condition);row.study=S.assess(study,slot,row,dark,white,rows);assert.equal(row.study.usable,true);rows.push(row);}
assert.equal(S.next(study,rows),null);
const result=S.summary(study,rows);assert.equal(result[2].plants,6);assert.equal(result[2].technical_pairs,12);assert.ok(Math.abs(result[2].delta-.2)<1e-10);assert.ok(result[2].ci.every(x=>Math.abs(x-.2)<1e-10));
assert.deepEqual(S.summary(study,[...rows,rows.at(-1)]),result);assert.equal(S.summary({...study,id:'other'},rows)[0].plants,0);
const incomplete=rows.filter(r=>!(r.plant==='p0'&&r.study.condition===2&&r.study.round===2));assert.equal(S.summary(study,incomplete)[2].plants,5);
const changed=measurement(first,.5);changed.geometry='changed';assert.equal(S.assess(study,first,changed,dark,white,rows).usable,false);
const stale={...white,timestamp:'2026-10-30T11:00:00Z'};assert.equal(S.assess(study,first,measurement(first,.5),dark,stale,[]).usable,false);
assert.equal(S.assess(study,first,measurement(first,.5),null,white,[]).usable,false);
const unlocked=measurement(first,.5);unlocked.acquisition.profile.operator_locked=false;assert.equal(S.assess(study,first,unlocked,dark,white,[]).usable,false);
// Multiple leaves must not give their plant extra weight.
const unequal=S.plan({conditions:conditions.slice(0,2),specimens:[{plant:'A',specimen:'1'},{plant:'A',specimen:'2'},{plant:'B',specimen:'1'}],seed:1}),u={id:'u',plan:unequal};
const ur=unequal.slots.map(slot=>({plant:slot.plant,specimen:slot.specimen,study:{id:'u',slot:slot.id,usable:true},response:{global_ratio:slot.condition?(slot.plant==='A'?.8:.2):.2}}));assert.ok(Math.abs(S.summary(u,ur)[1].delta-.3)<1e-10);
console.log('PASS: deterministic cyclic order, frozen-plan validation, QC exclusions, matched repeats, plan isolation and equal-weight plant summaries.');
