"""Train a deep transfer-learning baseline; evaluate frozen test sets once after selection."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

SEED = 20261029
SIZE = 160
CLASS_NAMES = ["healthy", "early_blight", "late_blight"]
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


class Images(Dataset):
    def __init__(self, rows, root, augment=False):
        self.root = root
        self.entries = [(r, view) for r in rows for view in (range(3) if augment else [0])]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row, view = self.entries[index]
        path = self.root / row["raw_path"]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            if view == 1:
                im = ImageOps.mirror(im)
            elif view == 2:
                im = im.rotate(10, resample=Image.Resampling.BILINEAR, fillcolor=(125, 125, 125))
            pixels = np.asarray(im, dtype=np.float32) / 255
        tensor = torch.from_numpy(((pixels - MEAN) / STD).transpose(2, 0, 1).copy())
        return tensor, row["label_id"]


def extract(encoder, dataset):
    features, targets = [], []
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for index, (x, y) in enumerate(loader):
            features.append(encoder(x).numpy())
            targets.append(y.numpy())
            if (index + 1) % 20 == 0:
                print(f"  extracted {min((index+1)*64,len(dataset))}/{len(dataset)}", flush=True)
    return np.concatenate(features), np.concatenate(targets)


def group_intervals(y, pred, groups):
    rng = np.random.default_rng(SEED)
    unique = sorted(set(groups))
    locations = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    accuracy, macro = [], []
    for _ in range(400):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([locations[g] for g in sampled])
        accuracy.append(accuracy_score(y[indices], pred[indices]))
        macro.append(f1_score(y[indices], pred[indices], labels=[0, 1, 2], average="macro", zero_division=0))
    return {"method": "400 leaf/duplicate-group bootstrap resamples; percentile intervals",
            "accuracy_95ci": np.quantile(accuracy, [.025, .975]).tolist(),
            "macro_f1_95ci": np.quantile(macro, [.025, .975]).tolist()}


def evaluation(y, probabilities, rows, threshold):
    pred = probabilities.argmax(1)
    accepted = probabilities.max(1) >= threshold
    return {"n": len(y), "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "balanced_accuracy": float(np.mean([np.mean(pred[y == i] == i) for i in range(3)])),
        "log_loss": log_loss(y, probabilities, labels=[0, 1, 2]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(y, pred, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "selection_threshold": threshold, "accepted_coverage": float(accepted.mean()),
        "accepted_accuracy": float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else None,
        "intervals": group_intervals(y, pred, [r["group_id"] for r in rows])}


def train(root, artifacts, public):
    started = time.monotonic()
    root, artifacts, public = Path(root), Path(artifacts), Path(public)
    public.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    manifest = artifacts / "manifest.jsonl"
    quality = json.loads((artifacts / "data_quality.json").read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != quality["manifest_sha256"]:
        raise RuntimeError("Split manifest hash changed; refusing training.")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    # Recheck immutable bytes before generating any features.
    for row in rows:
        if hashlib.sha256((root / row["raw_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Raw data changed after curation.")
    partitions = {s: [r for r in rows if r["split"] == s] for s in ["train", "validation", "test", "external_test"]}
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    encoder = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten()).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    weights_file = Path(torch.hub.get_dir()) / "checkpoints/mobilenet_v3_small-047dcff4.pth"
    backbone_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()
    cache_key = hashlib.sha256((quality["manifest_sha256"] + backbone_sha + hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + f"stretch-bilinear-{SIZE}-views-3-v1").encode()).hexdigest()
    cache = root / "features" / (cache_key + ".npz")
    cache.parent.mkdir(exist_ok=True)
    arrays = {}
    if cache.exists():
        arrays = dict(np.load(cache))
        print("Reusing content-keyed feature cache.", flush=True)
    else:
        for split, subset in partitions.items():
            print(f"Extracting pretrained deep features: {split} ({len(subset)} original images)", flush=True)
            x, y = extract(encoder, Images(subset, root, augment=split == "train"))
            arrays[split + "_x"], arrays[split + "_y"] = x, y
        np.savez_compressed(cache, **arrays)
    scaler = StandardScaler().fit(arrays["train_x"])
    tx, vx = scaler.transform(arrays["train_x"]), scaler.transform(arrays["validation_x"])
    trials, best = [], None
    for c in [.01, .1, 1., 10.]:
        start = time.monotonic()
        classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED, solver="lbfgs")
        classifier.fit(tx, arrays["train_y"])
        prediction = classifier.predict(vx)
        score = f1_score(arrays["validation_y"], prediction, average="macro")
        trial = {"C": c, "validation_macro_f1": score, "iterations": classifier.n_iter_.tolist(), "fit_seconds": time.monotonic()-start}
        trials.append(trial); print(json.dumps(trial), flush=True)
        if best is None or score > best[0]:
            best = (score, classifier, c)
    _, classifier, selected_c = best
    val_logits = classifier.decision_function(vx)
    vy = arrays["validation_y"]
    temperature = float(minimize_scalar(lambda t: log_loss(vy, softmax(val_logits/t, axis=1), labels=[0, 1, 2]),
        bounds=(.25, 8), method="bounded").x)
    val_probabilities = softmax(val_logits/temperature, axis=1)
    threshold = .9
    for candidate in np.arange(.5, .951, .01):
        accepted = val_probabilities.max(1) >= candidate
        if accepted.sum() >= 50 and accepted.mean() >= .5 and accuracy_score(vy[accepted], val_probabilities.argmax(1)[accepted]) >= .95:
            threshold = round(float(candidate), 2); break
    # Freeze every selection before inspecting either held-out test result.
    selection = {"selected_C": selected_c, "temperature": temperature, "abstention_threshold": threshold,
        "selection_data": "validation only", "candidates": trials}
    (artifacts / "selection.json").write_text(json.dumps(selection, indent=2))
    results = {}
    predictions = []
    for split, subset in partitions.items():
        x, y = arrays[split + "_x"], arrays[split + "_y"]
        # Training extraction has 3 views per original; report the original view only.
        if split == "train":
            x, y = x[::3], y[::3]
        probabilities = softmax(classifier.decision_function(scaler.transform(x))/temperature, axis=1)
        results[split] = evaluation(y, probabilities, subset, threshold)
        predictions.extend(dict(source_sha256=r["sha256"], dataset=r["dataset"], split=split, group_id=r["group_id"],
            reference=r["label"], prediction=CLASS_NAMES[int(p.argmax())], probabilities=p.tolist()) for r, p in zip(subset, probabilities))
    weights = classifier.coef_ / scaler.scale_[None, :] / temperature
    bias = (classifier.intercept_ - np.sum(classifier.coef_ * scaler.mean_[None, :] / scaler.scale_[None, :], axis=1)) / temperature
    head = nn.Linear(576, 3)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weights.astype(np.float32)))
        head.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    model = nn.Sequential(encoder, head).eval()
    example = torch.zeros(1, 3, SIZE, SIZE)
    onnx_path = public / "tomato-rgb.onnx"
    torch.onnx.export(model, example, str(onnx_path), input_names=["image"], output_names=["logits"], opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    runtime = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity_errors = []
    for subset in partitions.values():
        image, _ = Images(subset[:2], root)[0]
        inp = image.numpy()[None]
        with torch.inference_mode():
            expected = model(image[None]).numpy()
        actual = runtime.run(None, {"image": inp})[0]
        parity_errors.append(float(np.max(np.abs(expected-actual))))
    if max(parity_errors) > .001:
        raise RuntimeError("ONNX export failed numeric parity check.")
    # Deterministic, non-biological parity fixture: no source photographs redistributed.
    fixture_pixels = np.full((SIZE, SIZE, 3), 120, dtype=np.uint8)
    fixture = ((fixture_pixels.astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)[None]
    fixture_logits = runtime.run(None, {"image": fixture})[0][0].tolist()
    model_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    config = {"input_type": "rgb_image", "illumination": "white", "input_size": [SIZE, SIZE], "layout": "NCHW",
        "resize": "stretch", "scale": 1/255, "mean": MEAN.tolist(), "std": STD.tolist(), "class_names": CLASS_NAMES,
        "output_type": "logits", "crop": "Tomato", "abstention_threshold": threshold, "model_sha256": model_sha,
        "model_name": "MobileNetV3-Small frozen ImageNet encoder + fitted multinomial logistic head",
        "temperature_included_in_model": True}
    (public / "model_config.json").write_text(json.dumps(config, indent=2))
    (public / "parity_fixture.json").write_text(json.dumps({"pixel_value": 120, "logits": fixture_logits, "tolerance": .002}, indent=2))
    report = {"run_id": model_sha[:16], "model_sha256": model_sha, "manifest_sha256": quality["manifest_sha256"],
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "class_order": CLASS_NAMES, "method": config["model_name"], "seed": SEED, "backbone_weights_sha256": backbone_sha,
        "training_views_per_image": 3, "training_views": ["original", "horizontal mirror", "10 degree rotation"],
        "augmentation_applied_to": "train only", "preprocessing": config, "selection": selection, "results": results,
        "onnx_parity_max_absolute_error": max(parity_errors), "elapsed_seconds": time.monotonic()-started,
        "data_quality": quality, "versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "scikit-learn", "numpy", "Pillow", "onnx", "onnxruntime"]},
        "deployment_status": "research_baseline_not_validated_for_user_webcam_or_multispectral_inputs",
        "limitations": quality["limitations"] + ["Closed-set classification among three tomato conditions; unrelated images and unrepresented diseases may receive confident incorrect predictions.",
            "Validation data was used for hyperparameter selection, temperature and threshold; reported test and external results were not used for selection.",
            "The model is trained for white-light RGB images only. No disease inference is permitted for wavelength-labelled captures."]}
    (artifacts / "training_report.json").write_text(json.dumps(report, indent=2))
    (artifacts / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
    (public / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({s: {k: v[k] for k in ["n", "accuracy", "macro_f1", "accepted_coverage", "accepted_accuracy"]} for s, v in results.items()}, indent=2), flush=True)
    print(f"Model export verified: {model_sha}; run completed in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--artifacts", default="ml/artifacts")
    p.add_argument("--public", default="dist/model")
    a = p.parse_args()
    train(a.data_root, a.artifacts, a.public)
