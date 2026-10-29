(function(root){
'use strict';
const classes=['healthy','early_blight','late_blight'];
function accepted(trials,model){const seen=new Set();return trials.filter(t=>{const key=[t.plant,t.specimen,t.model_sha256].join('|');if(t.demo||!t.scored||!t.evidence||!classes.includes(t.reference)||t.model_sha256!==model||seen.has(key))return false;seen.add(key);return true;});}
function accuracy(trials,variant){return trials.length?trials.filter(t=>t.results.find(r=>r.variant===variant)?.prediction===t.reference).length/trials.length:null;}
function macroF1(trials,variant){if(!trials.length)return null;const m=Array.from({length:3},()=>[0,0,0]);for(const t of trials){const prediction=t.results.find(r=>r.variant===variant)?.prediction;if(classes.includes(prediction))m[classes.indexOf(t.reference)][classes.indexOf(prediction)]++;}return m.reduce((s,row,i)=>{const tp=row[i],fp=m.reduce((a,r,j)=>a+(j===i?0:r[i]),0),fn=row.reduce((a,v,j)=>a+(j===i?0:v),0);return s+(2*tp+fp+fn?2*tp/(2*tp+fp+fn):0);},0)/3;}
function deltaCI(trials,variant){const plants=[...new Set(trials.map(t=>t.plant))];if(plants.length<2)return null;let seed=20261029;const rand=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296;},values=[];for(let i=0;i<400;i++){const sample=Array.from({length:plants.length},()=>plants[Math.floor(rand()*plants.length)]).flatMap(p=>trials.filter(t=>t.plant===p));values.push(accuracy(sample,variant)-accuracy(sample,'rgb'));}values.sort((a,b)=>a-b);return [values[10],values[389]];}
root.SLEvaluation={accepted,accuracy,macroF1,deltaCI};if(typeof module!=='undefined'&&module.exports)module.exports=root.SLEvaluation;
})(typeof window!=='undefined'?window:globalThis);
