# Reproducible RGB training pipeline

This is a three-class **tomato, white-light RGB** research baseline. The model uses an ImageNet-pretrained MobileNetV3-Small encoder, frozen during training, followed by a trained multinomial logistic classifier. It is not a multispectral classifier.

The first recorded run achieved strong internal results and weak separate-source results. That external failure remains visible. Do not describe the model as ready for reliable field or webcam diagnosis.

## Reproduce

Use Python 3.11 or 3.12 and a CPU environment. From the repository root:

```powershell
python -m pip install -r ml/requirements.txt
python ml/ingest.py --data-root data --output ml/artifacts
python ml/curate.py --data-root data --output ml/artifacts
python ml/train.py --data-root data --artifacts ml/artifacts --public dist/model
python ml/validate_bundle.py --artifacts ml/artifacts --public dist/model
```

The first command for ingestion makes network requests to the public GitHub repositories, pinned to the commits in `ingest.py`. The original images remain under `data/raw`; never edit them. Partial downloads use temporary files and become raw files only after checksum verification. Successful files are reused on retry. Any failed expected download prevents training.

Data is not bundled in the project archive. `ml/artifacts/source_inventory.json` and `manifest.jsonl` contain immutable source URLs, Git blob identities and SHA-256 checksums. Dataset licences and attribution are retained in the archive. The training pipeline downloads ImageNet backbone weights from the official PyTorch model host and records their SHA-256 identity.

## Data engineering controls

1. **Raw ingestion:** immutable repository revisions and exact Git blob checksum validation. Record SHA-256, dataset, original label, source URL and size. No silently skipped download failures.
2. **Quality:** decode every image, enforce dimension bounds and validate raw checksums again. Unresolved or ambiguous PlantVillage leaf identities are quarantined rather than given invented independent IDs.
3. **Deduplication:** group exact file hashes, decoded-pixel hashes, known leaf IDs, and conservative rotation/mirror-canonical perceptual hashes. Quarantine groups with conflicting labels; exclude external duplicates of internal data. This does not prove every near duplicate has been found.
4. **Frozen splits:** deterministic, stratified group split at approximately 60/20/20. Known leaf and duplicate groups are atomic. Check zero cross-split overlap in group IDs, file hashes, decoded-pixel hashes and known leaf IDs.
5. **External audit:** the selected PlantDoc classes, including its original train and test folders, are external-only. They are never used to fit the scaler or classifier, select regularisation, calibrate temperature or choose the score threshold. Labels are inherited and not independently adjudicated.
6. **Training isolation:** fit StandardScaler and classifier on training features only. Apply augmentation only to the training partition. Compare four predefined regularisation candidates on validation macro F1. Freeze selection before evaluating internal and external tests.
7. **Lineage:** hash the manifest, backbone weights, training script and ONNX artifact. Save package versions, feature-cache identity, model selection trials, all evaluation predictions and quality/quarantine reports.
8. **Evaluation:** report accuracy, macro F1, per-class precision/recall, confusion matrices, log loss, abstention coverage and group-bootstrap intervals. Threshold-based abstention is not an out-of-distribution detector; the observed external failure demonstrates this limitation.
9. **Deployment:** verify exported numeric outputs and model integrity separately from model quality. A passing checksum or runtime test does not make a model accurate. Keep experimental predictions behind opt-in when external project targets fail.

## Limits and next dataset requirement

Plant IDs are unavailable, so leaf-level grouping cannot prove plant-level independence. The separate-source audit is useful but has inherited labels and scene differences. No data from the user's webcam, physical lights, or actual wavelength experiment has been collected or labelled for training. Browser and PIL image-resize implementations may differ slightly despite matching RGB ordering, size, scaling and normalisation.

The next training version needs independently confirmed images from multiple plants under the actual camera and lighting, with plant IDs, device IDs, acquisition times and illumination metadata. Freeze a new prospective plant-level test set before fitting or selecting that version. Preserve this run and its failed external results for comparison. Do not tune against this external audit until it looks good and then call it untouched testing.

The pipeline uses single-process data loading for compatibility with restricted execution environments; model operations still use CPU threads. Feature caches are keyed by the manifest, backbone, preprocessing and training-code identity. No GPU is required for this baseline. Repeated runs on different hardware/library versions may differ slightly; record a new run when inputs change.

## Matched ablation study

`python ml/ablation.py --data-root data --public dist/model` reproduces the six declared conditions using the original trained ONNX graph's frozen encoder and separately fitted heads. It resolves the original content-addressed feature cache, reuses the frozen split manifest, and selects regularisation/temperature on validation data only. `ml/artifacts/ablation_code_at_run.py` preserves the executed source; the replay script has a subsequent exact-cache-resolution repair documented in the report.

`python ml/check_ablation.py --runtime` validates all six artifact hashes, prediction identities, confusion-matrix counts and the numerical runtime fixture. In the downloadable Streamlit archive, use `--public model`. Tests and paired intervals are secondary analyses of the already-audited sets. Channel removal is not a physical wavelength experiment, and no variant is promoted on the basis of test results.
