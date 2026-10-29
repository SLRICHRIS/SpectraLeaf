# Camera-channel and training augmentation ablation report

These are actual retrained-head comparisons on the same frozen leaf-group splits. They are not physical LED wavelength ablations, and the existing external audit is reused for this secondary analysis.

| Condition | Held-out accuracy | External accuracy | External macro F1 | External accuracy delta vs RGB | Paired group-bootstrap 95% delta interval |
| --- | ---: | ---: | ---: | ---: | --- |
| RGB · full baseline | 95.2% | 48.3% | 0.366 | 0.0 pp | 0.0 to 0.0 pp |
| RGB · no training augmentation | 94.3% | 49.8% | 0.379 | 1.5 pp | -1.2 to 4.4 pp |
| Grayscale · retrained head | 89.9% | 38.6% | 0.296 | -9.7 pp | -15.2 to -3.6 pp |
| Red camera channel only | 91.3% | 38.6% | 0.347 | -9.7 pp | -17.4 to -1.9 pp |
| Green camera channel only | 89.4% | 39.4% | 0.312 | -8.9 pp | -15.3 to -2.5 pp |
| Blue camera channel only | 89.2% | 37.8% | 0.288 | -10.4 pp | -16.2 to -3.9 pp |

The no-augmentation variant shows a small external difference with an interval crossing zero. No variant resolves the external failure. No winning model is promoted from these test scores.

All conditions use the same frozen MobileNetV3-Small encoder. New heads are fitted and their regularisation/temperature are selected using validation data only. The full RGB head remains the original reference. R/G/B ablations preserve one broadband camera channel and set the other image channels to zero before normalisation; grayscale repeats the luminance-weighted gray input across all three channels. Each input condition is matched to its separately trained head.

Live inference compares the same frozen frame, rotates condition order, records preprocessing and inference times separately, and excludes model loading and numeric warm-up. Without an independently recorded reference label, accuracy and F1 remain unavailable. One plant/specimen trial is recorded once; biological uncertainty uses plant clusters, not repeated frames.

The JSON report includes per-condition validation trials, confusion matrices, paired group-bootstrap delta intervals, group-majority exact McNemar tests and Holm-adjusted p-values within each fixed dataset. The tests use leaf/duplicate groups, not unavailable plant IDs.

The single attributed reference demonstration image already belongs to the fixed held-out PlantVillage test set. It is excluded from prospective study records and cannot establish new performance.
