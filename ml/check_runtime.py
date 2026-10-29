"""Implementation smoke test, not model-quality validation."""
import json
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--public", default="dist/model")
root = Path(parser.parse_args().public)
config = json.loads((root / "model_config.json").read_text())
fixture = json.loads((root / "parity_fixture.json").read_text())
height, width = config["input_size"]
rgb = np.full((height, width, 3), fixture["pixel_value"], dtype=np.float32)
tensor = ((rgb * config["scale"] - np.asarray(config["mean"], dtype=np.float32))/np.asarray(config["std"], dtype=np.float32)).transpose(2,0,1)[None].astype(np.float32)
session = ort.InferenceSession(str(root / "tomato-rgb.onnx"), providers=["CPUExecutionProvider"])
actual = session.run(None, {"image": tensor})[0][0]
assert np.max(np.abs(actual-np.asarray(fixture["logits"]))) < fixture["tolerance"]
print("PASS: trained ONNX artifact executes and matches the deterministic numeric fixture.")
