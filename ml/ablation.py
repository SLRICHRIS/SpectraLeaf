"""Predeclared, matched-split ablations of camera channels and training augmentation."""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

from PIL import Image, ImageOps
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
SEED=20261029
SIZE=160
CLASS_NAMES=["healthy","early_blight","late_blight"]
MEAN=np.array([.485,.456,.406],dtype=np.float32)
STD=np.array([.229,.224,.225],dtype=np.float32)

VARIANTS = [
    {"id": "rgb", "name": "RGB · full baseline", "transform": "rgb", "augment": True},
    {"id": "noaug", "name": "RGB · no training augmentation", "transform": "rgb", "augment": False},
    {"id": "gray", "name": "Grayscale · retrained head", "transform": "gray", "augment": True},
    {"id": "red", "name": "Red camera channel only", "transform": "red", "augment": True},
    {"id": "green", "name": "Green camera channel only", "transform": "green", "augment": True},
    {"id": "blue", "name": "Blue camera channel only", "transform": "blue", "augment": True},
]


def extract_variant(runtime, rows, root, transform, augment=False):
    features, targets, batch = [], [], []
    entries=[(row,view) for row in rows for view in (range(3) if augment else [0])]
    for i,(row,view) in enumerate(entries):
        with Image.open(root / row["raw_path"]) as image:
            im=ImageOps.exif_transpose(image).convert("RGB").resize((SIZE,SIZE),Image.Resampling.BILINEAR)
            if view==1:im=ImageOps.mirror(im)
            elif view==2:im=im.rotate(10,resample=Image.Resampling.BILINEAR,fillcolor=(125,125,125))
            pixels=np.asarray(im,dtype=np.float32)/255
        if transform=="gray":pixels=np.repeat((pixels*np.array([.2126,.7152,.0722],dtype=np.float32)).sum(2)[:,:,None],3,2)
        elif transform in ("red","green","blue"):
            channel=["red","green","blue"].index(transform)
            pixels[:,:,[c for c in range(3) if c!=channel]]=0
        batch.append(((pixels-MEAN)/STD).transpose(2,0,1))
        targets.append(row["label_id"])
        if len(batch)==64 or i==len(entries)-1:
            features.append(runtime.run(["embedding"],{"image":np.stack(batch).astype(np.float32)})[0])
            batch=[]
            if (i+1)%1280==0:print(f"  extracted {i+1}/{len(entries)}",flush=True)
    return np.concatenate(features),np.asarray(targets)


def paired_stats(y, baseline, candidate, groups):
    correct_b, correct_c = baseline == y, candidate == y
    b_only, c_only = int((correct_b & ~correct_c).sum()), int((~correct_b & correct_c).sum())
    groups = np.asarray(groups)
    unique = sorted(set(groups))
    lookup = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(500):
        indices = np.concatenate([lookup[g] for g in rng.choice(unique, len(unique), replace=True)])
        differences.append(float(correct_c[indices].mean() - correct_b[indices].mean()))
    # McNemar's exact binomial test assumes independent discordant observations;
    # report a leaf/group-majority version rather than counting multiple views.
    b_groups = np.array([float(correct_b[lookup[g]].mean()) >= .5 for g in unique])
    c_groups = np.array([float(correct_c[lookup[g]].mean()) >= .5 for g in unique])
    gb, gc = int((b_groups & ~c_groups).sum()), int((~b_groups & c_groups).sum())
    p = float(binomtest(min(gb, gc), gb + gc, .5).pvalue) if gb + gc else 1.
    return {"accuracy_delta": float(correct_c.mean() - correct_b.mean()),
        "paired_group_bootstrap_delta_95ci": np.quantile(differences, [.025, .975]).tolist(),
        "baseline_correct_candidate_wrong_images": b_only, "candidate_correct_baseline_wrong_images": c_only,
        "group_majority_discordant_baseline_only": gb, "group_majority_discordant_candidate_only": gc,
        "group_majority_mcnemar_exact_p": p}



def run(root, artifacts, public):
    root, artifacts, public = Path(root), Path(artifacts), Path(public)
    np.random.seed(SEED); random.seed(SEED)
    manifest = artifacts / "manifest.jsonl"
    original = json.loads((artifacts / "training_report.json").read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == original["manifest_sha256"]
    protocol = {"defined_at": datetime.now(timezone.utc).isoformat(), "variants": VARIANTS, "seed": SEED,
        "manifest_sha256": original["manifest_sha256"], "selection": "validation macro F1 only; C in [0.01,0.1,1,10]",
        "backbone": "same frozen ImageNet MobileNetV3-Small encoder for all conditions",
        "interpretation": "Camera RGB channel ablations are not physical wavelength ablations.",
        "test_reuse": "Secondary comparative analysis on the previously audited fixed sets; not a fresh external validation.",
        "promotion": "No variant is promoted based on test/external results. Original full-RGB model remains the reference."}
    (artifacts / "ablation_protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    partitions = {s: [r for r in rows if r["split"] == s] for s in ["train", "validation", "test", "external_test"]}
    for row in rows:
        assert hashlib.sha256((root / row["raw_path"]).read_bytes()).hexdigest() == row["sha256"]
    graph=onnx.load(str(public/"tomato-rgb.onnx"))
    gemm=next(node for node in reversed(graph.graph.node) if node.op_type=="Gemm")
    feature_name=gemm.input[0]
    encoder_graph=onnx.ModelProto();encoder_graph.CopyFrom(graph)
    encoder_graph.graph.node.append(helper.make_node("Identity",[feature_name],["embedding"]))
    encoder_graph.graph.output.append(helper.make_tensor_value_info("embedding",TensorProto.FLOAT,["batch",576]))
    options=ort.SessionOptions();options.intra_op_num_threads=6
    encoder=ort.InferenceSession(encoder_graph.SerializeToString(),sess_options=options,providers=["CPUExecutionProvider"])
    base_key = hashlib.sha256((original["manifest_sha256"] + original["backbone_weights_sha256"] + original["training_script_sha256"] + f"stretch-bilinear-{SIZE}-views-3-v1").encode()).hexdigest()
    base_cache = root / "features" / (base_key + ".npz")
    base = dict(np.load(base_cache))
    base_head = dict(np.load(artifacts / "head_parameters.npz"))
    all_weights, all_biases, summary, stored_predictions = [], [], {}, []
    base_predictions = {s: [json.loads(line) for line in (artifacts / "predictions.jsonl").read_text().splitlines() if json.loads(line)["split"] == s] for s in partitions}
    baseline_by_split = {s: np.array([CLASS_NAMES.index(p["prediction"]) for p in base_predictions[s]]) for s in partitions}
    for variant in VARIANTS:
        print(f"Ablation: {variant['id']}", flush=True)
        if variant["transform"] == "rgb":
            arrays = {k: v.copy() for k, v in base.items()}
            if not variant["augment"]:
                arrays["train_x"], arrays["train_y"] = arrays["train_x"][::3], arrays["train_y"][::3]
        else:
            key = hashlib.sha256((original["manifest_sha256"] + original["backbone_weights_sha256"] + variant["id"] + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()
            cache = root / "features" / ("ablation-" + key + ".npz")
            if cache.exists(): arrays = dict(np.load(cache))
            else:
                arrays = {}
                for split, subset in partitions.items():
                    print(f"  {split}", flush=True)
                    arrays[split+"_x"], arrays[split+"_y"] = extract_variant(encoder, subset, root, variant["transform"], augment=split == "train")
                np.savez_compressed(cache, **arrays)
        if variant["id"] == "rgb":
            coefficients, intercept, mean, scale = (base_head[k] for k in ["coef", "intercept", "scaler_mean", "scaler_scale"])
            temperature = original["selection"]["temperature"]
            trials, chosen = original["selection"]["candidates"], original["selection"]["selected_C"]
        else:
            scaler = StandardScaler().fit(arrays["train_x"])
            tx, vx = scaler.transform(arrays["train_x"]), scaler.transform(arrays["validation_x"])
            trials, best = [], None
            for c in [.01, .1, 1., 10.]:
                clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED).fit(tx, arrays["train_y"])
                score = f1_score(arrays["validation_y"], clf.predict(vx), average="macro")
                trials.append({"C": c, "validation_macro_f1": float(score)})
                if best is None or score > best[0]: best = (score, clf, c)
            _, clf, chosen = best
            val_logits = clf.decision_function(vx)
            temperature = float(minimize_scalar(lambda t: log_loss(arrays["validation_y"], softmax(val_logits/t, axis=1), labels=[0,1,2]), bounds=(.25,8), method="bounded").x)
            coefficients, intercept, mean, scale = clf.coef_, clf.intercept_, scaler.mean_, scaler.scale_
        weights = coefficients / scale[None] / temperature
        bias = (intercept - np.sum(coefficients * mean[None] / scale[None], axis=1))/temperature
        all_weights.append(weights); all_biases.append(bias)
        results = {}
        for split, subset in partitions.items():
            features, target = arrays[split+"_x"], arrays[split+"_y"]
            if split == "train" and variant["augment"]: features, target = features[::3], target[::3]
            scores = softmax(features @ weights.T + bias, axis=1); prediction = scores.argmax(1)
            results[split] = {"n": len(target), "accuracy": float(accuracy_score(target, prediction)),
                "macro_f1": float(f1_score(target, prediction, average="macro")), "confusion_matrix": confusion_matrix(target, prediction, labels=[0,1,2]).tolist()}
            if split in ["test", "external_test"]:
                results[split]["paired_against_rgb"] = paired_stats(target, baseline_by_split[split], prediction, [r["group_id"] for r in subset])
                stored_predictions.extend({"variant": variant["id"], "split": split, "sha256": row["sha256"], "group_id": row["group_id"], "label": int(y), "prediction": int(p), "scores": score.tolist()} for row,y,p,score in zip(subset,target,prediction,scores))
        summary[variant["id"]] = dict(variant, selection={"C": chosen, "temperature": temperature, "trials": trials}, results=results)
        print(json.dumps({"variant": variant["id"], "test": results["test"]["accuracy"], "external": results["external_test"]["accuracy"]}), flush=True)
    # Holm correction within each dataset's five non-baseline comparisons.
    for split in ["test", "external_test"]:
        ranked = sorted([(v["results"][split]["paired_against_rgb"]["group_majority_mcnemar_exact_p"], k) for k,v in summary.items() if k != "rgb"])
        previous = 0
        for i,(p,k) in enumerate(ranked):
            adjusted = max(previous, min(1., (len(ranked)-i)*p)); previous = adjusted
            summary[k]["results"][split]["paired_against_rgb"]["holm_adjusted_p"] = adjusted
    weight_matrix=np.stack(all_weights).astype(np.float32).reshape(18,576).T.copy()
    biases=np.stack(all_biases).astype(np.float32)
    graph.graph.node.remove(gemm)
    graph.graph.node.extend([
        helper.make_node("MatMul",[feature_name,"ablation_weights"],["all_head_logits"]),
        helper.make_node("Reshape",["all_head_logits","ablation_shape"],["reshaped_logits"]),
        helper.make_node("Add",["reshaped_logits","ablation_biases"],["logits"])
    ])
    graph.graph.initializer.extend([numpy_helper.from_array(weight_matrix,"ablation_weights"),numpy_helper.from_array(biases,"ablation_biases"),numpy_helper.from_array(np.array([-1,6,3],dtype=np.int64),"ablation_shape")])
    del graph.graph.output[:]
    graph.graph.output.append(helper.make_tensor_value_info("logits",TensorProto.FLOAT,["batch",6,3]))
    onnx.checker.check_model(graph)
    output=public/"ablation-heads.onnx";onnx.save(graph,str(output))
    runtime=ort.InferenceSession(str(output),sess_options=options,providers=["CPUExecutionProvider"])
    tensor=((np.full((SIZE,SIZE,3),120,dtype=np.float32)/255-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    embedding=encoder.run(["embedding"],{"image":tensor})[0]
    expected=(embedding@weight_matrix).reshape(-1,6,3)+biases
    actual=runtime.run(None,{"image":tensor})[0]
    assert np.max(np.abs(expected-actual))<.001
    result = {"protocol": protocol, "variants": summary, "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "baseline_model_sha256": original["model_sha256"], "class_names": CLASS_NAMES, "variant_order": [v["id"] for v in VARIANTS],
        "input_size": [SIZE,SIZE], "mean": MEAN.tolist(), "std": STD.tolist(), "scale": 1/255,
        "parity_fixture": {"pixel_value":120,"logits":actual[0].tolist(),"tolerance":.002},
        "interpretation": "Matched secondary ablation comparison; not physical wavelength selection or prospective validation."}
    (public/"ablation_report.json").write_text(json.dumps(result,indent=2))
    (artifacts/"ablation_report.json").write_text(json.dumps(result,indent=2))
    (artifacts/"ablation_predictions.jsonl").write_text(''.join(json.dumps(x)+'\n' for x in stored_predictions))
    print("Ablation bundle exported and verified.",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--artifacts',default='ml/artifacts');p.add_argument('--public',default='dist/model');a=p.parse_args()
    run(a.data_root,a.artifacts,a.public)
