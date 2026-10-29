"""Check the actual six-condition artifact and its prediction lineage."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--public',default='dist/model');parser.add_argument('--artifacts',default='ml/artifacts');parser.add_argument('--runtime',action='store_true');args=parser.parse_args()
public,artifacts=Path(args.public),Path(args.artifacts)
report=json.loads((public/'ablation_report.json').read_text())
assert report['variant_order']==['rgb','noaug','gray','red','green','blue']
assert hashlib.sha256((public/'ablation-heads.onnx').read_bytes()).hexdigest()==report['model_sha256']
assert hashlib.sha256((public/'tomato-rgb.onnx').read_bytes()).hexdigest()==report['baseline_model_sha256']
assert hashlib.sha256((artifacts/'ablation_code_at_run.py').read_bytes()).hexdigest()==report['ablation_script_sha256']
manifest=[json.loads(x) for x in (artifacts/'manifest.jsonl').read_text().splitlines()]
expected={(r['split'],r['sha256']):r for r in manifest if r['split'] in ['test','external_test']}
predictions=[json.loads(x) for x in (artifacts/'ablation_predictions.jsonl').read_text().splitlines()]
for variant in report['variant_order']:
    actual=[p for p in predictions if p['variant']==variant]
    assert {(p['split'],p['sha256']) for p in actual}==set(expected)
    assert len(actual)==len(expected)
    for p in actual:
        row=expected[p['split'],p['sha256']]
        assert row['group_id']==p['group_id'] and row['label_id']==p['label']
    for split in ['test','external_test']:
        selected=[p for p in actual if p['split']==split]
        matrix=[[0]*3 for _ in range(3)]
        for p in selected:matrix[p['label']][p['prediction']]+=1
        r=report['variants'][variant]['results'][split]
        assert matrix==r['confusion_matrix'] and len(selected)==r['n']
        assert abs(sum(matrix[i][i] for i in range(3))/len(selected)-r['accuracy'])<1e-12
if args.runtime:
    import numpy as np
    import onnxruntime as ort
    options=ort.SessionOptions();options.log_severity_level=3
    session=ort.InferenceSession(str(public/'ablation-heads.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    fixture=report['parity_fixture'];size=report['input_size'][0]
    tensor=((np.full((size,size,3),fixture['pixel_value'],dtype=np.float32)*report['scale']-np.array(report['mean'],dtype=np.float32))/np.array(report['std'],dtype=np.float32)).transpose(2,0,1)[None].astype(np.float32)
    output=session.run(None,{'image':tensor})[0]
    assert output.shape==(1,6,3)
    assert np.max(np.abs(output[0]-np.array(fixture['logits'])))<fixture['tolerance']
print('PASS: six-condition artifact identity, prediction coverage, reference lineage, actual counts and numeric fixture.')
