(function(root){
'use strict';
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
const sd=a=>a.length>1?Math.sqrt(a.reduce((s,v)=>s+(v-mean(a))**2,0)/(a.length-1)):null;
const finite=a=>a.filter(Number.isFinite);
function outputSummary(rgba){const n=rgba.length/4,sums=[0,0,0],clips=[0,0,0];for(let i=0;i<rgba.length;i+=4)for(let c=0;c<3;c++){sums[c]+=rgba[i+c];if(rgba[i+c]>=250)clips[c]++;}return {means:sums.map(v=>v/n),clipped:clips.map(v=>v/n),pixel_count:n};}
function frame(rgba){
 const n=rgba.length/4,values=[[],[],[]],clipped=[0,0,0];
 for(let i=0;i<n;i++)for(let c=0;c<3;c++){const v=rgba[i*4+c];values[c].push(v);if(v>=250)clipped[c]++;}
 return {pixels:values,means:values.map(mean),clipped:clipped.map(v=>v/n)};
}
function burst(frames){
 if(!frames.length)throw new Error('No frames were acquired.');
 const n=frames.length,p=frames[0].pixels[0].length;
 const map=[0,1,2].map(c=>Array.from({length:p},(_,i)=>mean(frames.map(f=>f.pixels[c][i]))));
 const channels=[0,1,2].map(c=>{
  const series=frames.map(f=>f.means[c]),blocks=[];
  for(let i=0;i+1<n;i+=2)blocks.push((series[i]+series[i+1])/2);
  const blockSD=sd(blocks),u=blockSD===null?null:blockSD/Math.sqrt(blocks.length),m=mean(series);
  const portion=Math.max(1,Math.floor(n/3));
  return {mean:m,temporal_sd:sd(series),standard_uncertainty:u,blocks:blocks.length,
   spatial_cv:m>0?sd(map[c])/m:null,clipped_fraction:mean(frames.map(f=>f.clipped[c])),
   drift_fraction:(mean(series.slice(-portion))-mean(series.slice(0,portion)))/Math.max(Math.abs(m),1),series};
 });
 return {n,channels,map};
}
function t95(df){const table=[null,12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228];return df<1?null:df<=10?table[df]:df<=20?2.086:df<=30?2.042:1.96;}
function normalise(sample,dark,white,channel=0,options={}){
 if(!sample||!dark||!white)return {valid:false,reason:'Matching sample, dark and white acquisitions are required.'};
 const s=sample.channels[channel],d=dark.channels[channel],w=white.channels[channel],den=w.mean-d.mean;
 if(den<10)return {valid:false,reason:'White-minus-dark response is below the project minimum of 10 DN.'};
 const q=(s.mean-d.mean)/den,whiteOnly=w.mean? s.mean/w.mean:null;
 const ratios=sample.map[channel].map((v,i)=>{const divisor=white.map[channel][i]-dark.map[channel][i];return divisor>=10?(v-dark.map[channel][i])/divisor:NaN;});
 const usable=finite(ratios),flat=mean(usable),invalid=1-usable.length/ratios.length;
 const knownRho=Number.isFinite(options.rho)&&options.rho>0&&options.rho<=1;
 const rho=knownRho?options.rho:1;
 let standard=null,expanded=null,reflectanceStandard=null,reflectanceExpanded=null,df=null;
 if([s,d,w].every(v=>Number.isFinite(v.standard_uncertainty))){
  const a=s.mean-d.mean,b=den;
  const variance=(s.standard_uncertainty/b)**2+(-a/b**2*w.standard_uncertainty)**2+((s.mean-w.mean)/b**2*d.standard_uncertainty)**2;
  standard=Math.sqrt(variance);df=Math.min(s.blocks,d.blocks,w.blocks)-1;expanded=standard*t95(df);
  if(knownRho){reflectanceStandard=Math.sqrt(rho*rho*variance+(Number.isFinite(options.uRho)?(q*options.uRho)**2:0));reflectanceExpanded=reflectanceStandard*t95(df);}
 }
 const flags=[];
 if(s.clipped_fraction>.01||w.clipped_fraction>.01)flags.push('clipping >1%');
 if(invalid>.05)flags.push('more than 5% invalid reference pixels');
 if(w.spatial_cv>.1)flags.push('white-field CV >10%');
 if([s,d,w].some(v=>Math.abs(v.drift_fraction)>.02))flags.push('burst drift >2%');
 if(q<0||q>1.2)flags.push('response ratio outside 0–1.2; inspect geometry/references');
 if(sample.n<8||dark.n<8||white.n<8)flags.push('fewer than eight frames in at least one acquisition');
 const valid=!flags.some(f=>f.startsWith('clipping')||f.startsWith('more than'))&&Number.isFinite(flat);
 return {valid,reason:valid?'QC passed for response-ratio calculation.':flags.join('; '),flags,
  raw_dn:s.mean,dark_corrected_dn:s.mean-d.mean,white_only_ratio:whiteOnly,
  global_ratio:q,flatfield_ratio:flat,invalid_pixel_fraction:invalid,
  estimate:flat===null?null:flat*rho,global_estimate:q*rho,standard_uncertainty:standard,expanded_uncertainty:expanded,reflectance_standard_uncertainty:reflectanceStandard,reflectance_expanded_uncertainty:reflectanceExpanded,df,
  uncertainty_scope:'Shown ratio uncertainty covers temporal blocks only. Reflectance uncertainty additionally includes entered reference uncertainty when available; other systematic errors remain unestablished.',
  quantity:knownRho&&options.linearEvidence&&options.spectralEvidence&&valid&&q*rho>=0&&q*rho<=1?'operator-qualified band reflectance estimate':'relative camera-response ratio'};
}
function linearity(points){
 if(points.length<5)return {valid:false,reason:'At least five matched dark/illuminated exposure points are required.'};
 const x=points.map(p=>p.exposure),y=points.map(p=>p.light-p.dark),mx=mean(x),my=mean(y),ss=x.reduce((s,v)=>s+(v-mx)**2,0);
 if(ss===0||new Set(x).size<5)return {valid:false,reason:'At least five distinct exposure values are required.'};
 const slope=x.reduce((s,v,i)=>s+(v-mx)*(y[i]-my),0)/ss,intercept=my-slope*mx;
 const residual=y.map((v,i)=>v-(intercept+slope*x[i])),sse=residual.reduce((s,v)=>s+v*v,0),sst=y.reduce((s,v)=>s+(v-my)**2,0),span=Math.max(...y)-Math.min(...y);
 const error=span>0?Math.max(...residual.map(Math.abs))/span:Infinity;
 return {valid:slope>0&&span>=30&&error<=.05,points:points.length,slope,intercept,r2:sst?1-sse/sst:null,
  maximum_deviation_fraction_of_span:error,span_dn:span,residuals:residual,
  scope:'Output-response check over the entered range, not EMVA 1288 certification or a raw-sensor characterisation.'};
}
function indices(rows){
 const eligible=rows.filter(r=>r.qualified&&r.nirVerified&&r.fwhm_nm>0&&r.fwhm_nm<=40&&Number.isFinite(r.value)&&r.value>=0&&r.value<=1);
 function nearest(min,max,target){return eligible.filter(r=>r.wavelength_nm>=min&&r.wavelength_nm<=max).sort((a,b)=>Math.abs(a.wavelength_nm-target)-Math.abs(b.wavelength_nm-target))[0];}
 const nir=nearest(760,900,850),red=nearest(640,680,660),edge=nearest(705,740,720),result={};
 for(const [key,other] of [['NDVI',red],['NDRE',edge]]){
  if(!nir||!other||nir.specimen!==other.specimen||nir.plant!==other.plant||nir.geometry!==other.geometry||nir.value+other.value<=0){result[key]={available:false,reason:'Matched qualified physical bands, same plant/specimen/geometry, and verified NIR sensitivity are required.'};continue;}
  const n=nir.value,r=other.value,value=(n-r)/(n+r);
  const u=Number.isFinite(nir.u)&&Number.isFinite(other.u)?Math.hypot(2*r/(n+r)**2*nir.u,-2*n/(n+r)**2*other.u):null;
  result[key]={available:true,value,standard_uncertainty:u,bands:[other.wavelength_nm,nir.wavelength_nm],scope:'Vegetation index, not a disease label; cross-band systematic covariance is omitted.'};
 }
 return result;
}
function photon(nm,irradiance=null){const joule=6.62607015e-34*299792458/(nm*1e-9);return {energy_joule:joule,energy_ev:joule/1.602176634e-19,photon_flux_per_m2_s:Number.isFinite(irradiance)&&irradiance>=0?irradiance/joule:null};}
root.SLOptics={mean,sd,outputSummary,frame,burst,normalise,linearity,indices,photon};
if(typeof module!=='undefined'&&module.exports)module.exports=root.SLOptics;
})(typeof window!=='undefined'?window:globalThis);
