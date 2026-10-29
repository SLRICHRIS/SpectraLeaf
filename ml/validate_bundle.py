"""Fail on data/model identity errors; report a separate field-readiness gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate(artifacts, public):
    artifacts, public = Path(artifacts), Path(public)
    report = json.loads((public / "training_report.json").read_text())
    config = json.loads((public / "model_config.json").read_text())
    quality = json.loads((artifacts / "data_quality.json").read_text())
    manifest = artifacts / "manifest.jsonl"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report["manifest_sha256"] == quality["manifest_sha256"], "Manifest identity mismatch"
    assert hashlib.sha256((public / "tomato-rgb.onnx").read_bytes()).hexdigest() == config["model_sha256"] == report["model_sha256"], "Model identity mismatch"
    assert hashlib.sha256(Path(__file__).with_name("train.py").read_bytes()).hexdigest() == report["training_script_sha256"], "Training code changed after the recorded run"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    splits = ["train", "validation", "test", "external_test"]
    for field in ["group_id", "sha256", "pixel_sha256", "leaf_id"]:
        sets = [{r[field] for r in rows if r["split"] == split and r.get(field)} for split in splits]
        assert all(not sets[i] & sets[j] for i in range(4) for j in range(i)), f"Split overlap: {field}"
    assert config["class_names"] == report["class_order"] == quality["class_order"]
    assert report["selection"]["selection_data"] == "validation only"
    assert report["onnx_parity_max_absolute_error"] <= .001
    predictions = [json.loads(line) for line in (artifacts / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == len(rows), "Missing or excess prediction rows"
    row_lookup = {(r["split"], r["sha256"]): r for r in rows}
    assert len(row_lookup) == len(rows), "Unexpected duplicate identity in accepted manifest"
    prediction_keys = {(p["split"], p["source_sha256"]) for p in predictions}
    assert prediction_keys == set(row_lookup), "Predictions do not cover exactly the accepted manifest"
    for prediction in predictions:
        row = row_lookup[prediction["split"], prediction["source_sha256"]]
        assert row["label"] == prediction["reference"] and row["group_id"] == prediction["group_id"], "Prediction lineage mismatch"
        scores = prediction["probabilities"]
        assert len(scores) == 3 and all(math.isfinite(v) and 0 <= v <= 1 for v in scores) and abs(sum(scores)-1) < 1e-6
        assert report["class_order"][max(range(3), key=lambda i: scores[i])] == prediction["prediction"]
    for split in splits:
        selected = [r for r in rows if r["split"] == split]
        assert report["results"][split]["n"] == len(selected), f"Evaluation count mismatch: {split}"
        actual = [p for p in predictions if p["split"] == split]
        matrix = [[0]*3 for _ in range(3)]
        for p in actual:
            matrix[report["class_order"].index(p["reference"])][report["class_order"].index(p["prediction"])] += 1
        assert matrix == report["results"][split]["confusion_matrix"], "Reported confusion matrix differs from prediction records"
        accuracy = sum(matrix[i][i] for i in range(3))/len(actual)
        assert abs(accuracy-report["results"][split]["accuracy"]) < 1e-12, "Reported accuracy differs from predictions"
    # Project-specific release targets, not an industry certification or guarantee.
    external = report["results"]["external_test"]
    ready = external["accuracy"] >= .85 and external["macro_f1"] >= .8
    policy = {"artifact_integrity": "passed", "field_readiness": "not_established" if ready else "failed_external_target",
        "project_external_targets": {"accuracy": .85, "macro_f1": .8},
        "actual_external_accuracy": external["accuracy"], "actual_external_macro_f1": external["macro_f1"],
        "reliable_diagnosis_enabled": False, "experimental_predictions_require_opt_in": True,
        "webcam_and_physical_illumination_validation": "not_performed", "multispectral_model": "not_trained"}
    (public / "release_gate.json").write_text(json.dumps(policy, indent=2))
    (artifacts / "release_gate.json").write_text(json.dumps(policy, indent=2))
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="ml/artifacts")
    parser.add_argument("--public", default="dist/model")
    args = parser.parse_args()
    validate(args.artifacts, args.public)
