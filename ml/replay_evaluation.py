"""Replay a browser evaluation ledger from its retained lossless input frames."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('ledger');p.add_argument('--public',default='dist/model');a=p.parse_args()
public=Path(a.public);report=json.loads((public/'ablation_report.json').read_text());data=json.loads(Path(a.ledger).read_text())
assert data['format']=='spectraleaf.evaluation.v1'
assert hashlib.sha256((public/'ablation-heads.onnx').read_bytes()).hexdigest()==report['model_sha256']
options=ort.SessionOptions();options.log_severity_level=3
runtime=ort.InferenceSession(str(public/'ablation-heads.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
for trial in data['trials']:
    assert not trial.get('demo') and trial['model_sha256']==report['model_sha256']
    image=Image.open(io.BytesIO(base64.b64decode(trial['frame_png'].split(',',1)[1]))).convert('RGBA')
    rgba=np.asarray(image,dtype=np.uint8)
    assert hashlib.sha256(rgba.tobytes()).hexdigest()==trial['frame_pixel_sha256']
    original=rgba[:,:,:3].astype(np.float32)/255
    for i,key in enumerate(report['variant_order']):
        pixels=original.copy();transform=report['variants'][key]['transform']
        if transform=='gray':pixels=np.repeat((pixels*np.array([.2126,.7152,.0722],dtype=np.float32)).sum(2)[:,:,None],3,2)
        elif transform in ['red','green','blue']:
            c=['red','green','blue'].index(transform);pixels[:,:,[j for j in range(3) if j!=c]]=0
        tensor=((pixels-np.array(report['mean'],dtype=np.float32))/np.array(report['std'],dtype=np.float32)).transpose(2,0,1)[None].astype(np.float32)
        logits=runtime.run(None,{'image':tensor})[0][0,i];scores=np.exp(logits-logits.max());scores/=scores.sum()
        stored=next(r for r in trial['results'] if r['variant']==key)
        assert np.max(np.abs(scores-np.array(stored['scores'])))<.002
print(f"PASS: replayed all six condition outputs for {len(data['trials'])} frozen specimen trials. Timing is hardware-specific and not replay-asserted.")
