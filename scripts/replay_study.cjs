const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'..'),web=fs.existsSync(path.join(root,'dist/study-core.js'))?'dist':'web',S=require(path.join(root,web,'study-core.js')),O=require(path.join(root,web,'optics-core.js'));
if(!process.argv[2])throw new Error('Usage: node scripts/replay_study.cjs spectraleaf-physical-study.json');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));assert.equal(data.format,'spectraleaf.physical-study.v1');const {study,samples,reference_history:refs}=data;
assert.equal(crypto.createHash('sha256').update(study.plan_json).digest('hex'),study.plan_sha256,'Frozen plan checksum differs');assert.deepEqual(JSON.parse(study.plan_json),study.plan,'Frozen plan content differs');
const rebuilt=S.plan(study.plan);rebuilt.evidence_id=study.plan.evidence_id;assert.deepEqual(rebuilt,study.plan,'Unsupported or changed schedule / QC protocol');
const prior=[];for(const row of samples){const slot=S.next(study,prior);assert.ok(slot,'Acquisition exists after schedule completion');assert.equal(slot.id,row.study.slot);const dark=refs[row.references.dark],white=refs[row.references.white];
 if(dark&&white){assert.equal(row.acquisition.profile.key,dark.profile.key);assert.equal(dark.profile.key,white.profile.key);const result=O.normalise(row.acquisition,dark,white,row.channel,row.calibration_options||{});assert.deepEqual(result,row.response,'Replayed optical response differs');}
 else assert.equal(row.study.usable,false,'Accepted acquisition missing references');
 assert.deepEqual(S.assess(study,slot,row,dark,white,prior),row.study,'Replayed study QC differs');prior.push(row);
}assert.deepEqual(S.summary(study,samples),data.summary,'Replayed plant-level table differs');
console.log(`PASS: frozen plan checksum, ${samples.length} retained attempts, references, QC, matched pairs and plant-bootstrap table replayed.`);
