# SpectraLeaf tomato RGB model card

**Status: trained research baseline; external validation failed the project release target. Not ready for reliable field or webcam disease diagnosis.**

Run: `87f8f0663749a117`. Date: 29 October 2026.

## Supported task

Classify a close image of a tomato leaf under white light into **healthy**, **early blight**, or **late blight**. The class order is `healthy, early_blight, late_blight`. Other diseases, other species, full-scene photographs and coloured illumination are outside the model's established scope. No multispectral or early-detection claim is supported.

Architecture: MobileNetV3-Small frozen ImageNet encoder + fitted multinomial logistic head. Only the multinomial logistic head is fitted; the deep ImageNet encoder remains frozen. This is transfer learning, not an end-to-end fine-tuned CNN. Input: RGB, 160×160 pixels, NCHW float32, `(pixel/255 - ImageNet mean)/ImageNet std`. The recorded temperature is already incorporated into the exported logits.

## Measured results

| Partition | Images | Accuracy | Macro F1 | Accuracy 95% group-bootstrap interval |
| --- | ---: | ---: | ---: | --- |
| Training | 1750 | 100.00% | 1.0000 | 100.00%–100.00% |
| Validation | 584 | 95.55% | 0.9548 | 93.15%–97.60% |
| Held-out leaves | 584 | 95.21% | 0.9513 | 92.81%–97.09% |
| Separate source (PlantDoc) | 259 | 48.26% | 0.3659 | 41.79%–54.24% |

The gap between held-out PlantVillage results and separate-source results is a material generalisation failure. In the separate-source evaluation, **healthy recall was 0.0%**, early-blight recall was 93.1%, and late-blight recall was 40.4%. A high class score did not reliably reject these errors: external accepted accuracy remained 48.3%. Temperature calibration on controlled images did not establish calibration on the new source.

The browser and Streamlit interfaces therefore require explicit opt-in to experimental predictions. Passing model checksum or runtime parity checks establishes artifact identity and implementation consistency, not diagnostic accuracy. No webcam image from the user has been used for training or prospective validation.

## Data lineage and engineering

- 4762 expected source images downloaded and verified against pinned Git blob checksums; raw bytes remain immutable.
- 3177 retained after curation; 1584 quarantined; 1 exact decoded-image duplicate removed.
- 1582 had unresolved leaf identity. Two images belonged to a detected duplicate group with conflicting labels.
- Approximately 60/20/20 group-stratified internal split. Known leaf IDs and detected duplicate groups do not cross splits. Group, byte-hash, decoded-pixel-hash and known-leaf overlap checks found no overlap.
- The external PlantDoc images never entered fitting, hyperparameter selection, temperature fitting or threshold selection. Original source labels were inherited without independent pathology confirmation.
- Three training views per image: original, mirrored and 10-degree rotation. Evaluation images were not augmented.
- Four predefined L2 regularisation candidates were selected on validation macro F1. StandardScaler was fitted on training features only.
- Plant identifiers are unavailable; leaf-level separation does not prove plant-level independence. Conservative perceptual hashes cannot guarantee detection of all near duplicates.

Manifest SHA-256: `2ce13eb0130b7ccd3969317e4e7f07a27c3bc640980655eb529ac25a46b6b8db`.

Model SHA-256: `87f8f0663749a117002d1266320b3b76bbc2f9c7cda5ef48cd354cc645adb8f8`.

Backbone SHA-256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

Training script SHA-256: `cb0b8d51f34cbc949d736468160af56fa39a97b41247a838a2915fc62d945192`.

ONNX export parity maximum absolute logit error: `6.1988831e-06`. The browser additionally checks the model SHA-256 and a deterministic synthetic-input fixture before predictions. The fixture is an implementation test, not disease-validation evidence.

## Reproduction and next validation

See `ml/README.md`, the immutable source inventory, accepted/quarantine manifests, selection trials, per-image predictions, pinned requirements and environment record in the source download. Raw datasets are fetched from the recorded source URLs and are not redistributed in the archive. Results can vary slightly across hardware or dependency versions; record a new run when inputs change.

Collect independently confirmed images of multiple plants using the actual webcam and physical lighting. Record plant ID, specimen ID, device, illumination, time and exposure conditions. Reserve a new prospective plant-level test set before fitting or selecting the next model. The failed external results in this run must remain visible. Do not tune against them repeatedly and then describe them as untouched test evidence.

## Sources and licences

PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset, revision `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`. Dataset card declares CC BY-SA 3.0. Cite Mohanty, Hughes and Salathé (2016), DOI 10.3389/fpls.2016.01419.

PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset, revision `5467f6012d78d1c446145d5f582da6096f852ae8`. CC BY 4.0. Cite Singh et al. (2020), DOI 10.1145/3371158.3371196.

Encoder: torchvision MobileNetV3-Small IMAGENET1K_V1 weights, https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth. Browser runtime: ONNX Runtime Web 1.20.1. Attribution and copied licence text accompany the project download.
