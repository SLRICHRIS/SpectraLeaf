# SpectraLeaf optical measurement protocol

This workbench supports a controlled experiment. It does not turn a standard RGB webcam into a calibrated spectrometer. The existing disease models are white-light RGB baselines with weak external validation; the optical measurements are a separate evidence stream.

## Signal path and acquisition architecture

1. **Illumination:** use a stable, current-regulated LED source or a characterised broadband source with filters. Record peak wavelength, measured FWHM, source/driver identity, warm-up condition, temperature and irradiance when a radiometer is available. Equal LED drive current does not imply equal optical power across colours. Nominal colour is not a measured spectrum.
2. **Specimen:** identify plant and leaf independently, fix the leaf plane, record orientation and illumination angle, and avoid motion, shadows and surface glare. A diffuse enclosure can improve repeatability. If cross-polarisers or filters are used, fix their orientation and include them in every reference acquisition.
3. **Collection optics:** record working distance, field of view, aperture/focus state, optical filters and any IR-cut element. Reference targets occupy the specimen plane. A flat white reference cannot fully correct leaf BRDF, curvature, angle, or subsurface scattering.
4. **Detector and analogue chain:** spectral responsivity, quantum efficiency, read noise, dark current, analogue gain and ADC range belong to the actual camera/AFE. Ordinary browser frames are already demosaiced and processed. Do not infer electron counts, quantum efficiency, NEP, raw-sensor read noise or EMVA compliance from these pixels.
5. **Digital pipeline:** exposure, gain, white balance, denoising, gamma, tone mapping and compression can change the relationship between photon exposure and digital values. The browser requests supported manual controls and checks the reported settings. Unsupported settings are shown as unknown; an attempted constraint is not a verified lock.

Before nonlinear ISP processing, a simplified channel model is

`DN_c ≈ D_c + g t ∫ E(λ) ρ(λ) T(λ) Q_c(λ) dλ + noise`.

The source spectrum, optical transmission and detector response weight the measured band. The R, G and B outputs are broad overlapping response channels. A single channel does not equal a specific LED wavelength. Photon energy at a selected centre is `hc/λ`; this describes a nominal photon, not the full LED spectrum or its delivered power.

## Referencing and quality checks

Keep source, exposure, gain, processing, ROI, channel and geometry fixed. Acquire a covered-lens dark reference, a diffuse white target at the leaf plane, and the specimen. Each profile has an exact key containing recorded acquisition settings and geometry. Changed settings, a changed ROI or a different source invalidate matching references. References also expire after the operator-selected maximum age.

The full browser bench captures 8, 12 or 20 **distinct video frames**. ROI mean signal, clipping and the frame-to-frame mean series are computed before downsampling, using original cropped camera-output pixels. A 64×64 map is retained for the spatial flat-field comparison; its spatial CV is a downsampled-field metric. Neither output is a raw-sensor noise characterisation. A still upload contains one observation, so it cannot provide temporal repeatability or a temporal standard error. The native Streamlit optical page explicitly supports single-image referencing; the browser bench provides the more complete burst workflow.

The current project QC defaults are white-minus-dark ≥10 DN, selected-channel clipping ≤1%, and ≥95% usable flat-field pixels. White spatial CV >10%, temporal drift >2%, small frame counts and out-of-range ratios are warnings. These thresholds are project defaults, not limits prescribed by EMVA or a disease-detection standard.

For each channel and matched ROI:

`r_global = (mean(S) − mean(D))/(mean(W) − mean(D))`

`r_flat = mean_valid_pixels((S_i − D_i)/(W_i − D_i))`.

The global and pixel-flat-field ratios answer different spatial-weighting questions. Compare raw DN, dark-only DN, white-only ratio, fully corrected ROI ratio, pixel flat-field ratio and whole-frame ratio as optical-processing ablations. Do not compare their numeric magnitudes as though all had the same units, and do not call them disease accuracies.

By default, the displayed quantity is a **relative camera-response ratio**. A known white reflectance, certificate/evidence ID, documented linear-response evidence, spectral-response evidence, stable settings, adequate bursts and passing QC are required for the app's operator-qualified band reflectance estimate. The evidence is operator-declared; the app does not independently certify the equipment or uploaded report. The estimate still has unmeasured systematic uncertainty.

## Uncertainty

For `ρ̂ = ρ_w (S−D)/(W−D)`, the sensitivity coefficients are:

| Input | Derivative |
| --- | --- |
| S | `ρ_w/(W−D)` |
| W | `−ρ_w(S−D)/(W−D)²` |
| D | `ρ_w(S−W)/(W−D)²` |
| ρ_w | `(S−D)/(W−D)` |

The D derivative handles the same dark acquisition appearing in numerator and denominator. The implementation combines temporal standard uncertainties through these derivatives and adds entered white-target uncertainty for a reflectance estimate. Ratio uncertainty and reflectance uncertainty are kept separate.

Two-frame non-overlapping block means provide a descriptive temporal uncertainty estimate. The displayed approximate 95% expansion uses a conservative minimum block degrees of freedom. Blocks can remain correlated, and the table does not provide a complete metrological uncertainty budget. Unmeasured spectral mismatch, geometry/angle, drift, source instability, target uncertainty and nonlinear processing cannot be assumed zero. A one-frame acquisition reports temporal uncertainty as unavailable.

## Exposure-response experiment

At constant irradiance, gain, geometry and white balance, collect dark/illuminated pairs at at least five distinct exposure coordinates. Browser-reported exposureTime uses 100 µs units according to W3C Image Capture. If only an operator-supplied coordinate is available, identify it as such. The app checks agreement with a reported exposure when available and keeps series with different observed settings separate.

The fit reports slope, intercept, R², dynamic span and maximum deviation from the fitted response. The project requires a positive slope, at least 30 DN of span and maximum deviation ≤5% of the span. Passing is an output-linearity check over that tested range; it is **not** EMVA 1288 certification, proof of raw-camera linearity, or verification of all hidden ISP operations.

## Physical bands and indices

Collect and reference each physical LED/filter band separately. Record the detector channel used at that band. For NIR measurements, first establish that the sensor and optics actually pass/respond to that band; many camera configurations include IR rejection. An RGB file cannot recover an unmeasured NIR band.

NDVI and NDRE remain unavailable unless the ledger contains qualified narrow-band estimates with the same plant, specimen and geometry plus operator-verified NIR sensitivity. The implementation uses appropriately ranged red/red-edge and NIR bands and reports their actual centre labels. A maximum 40 nm FWHM is a project eligibility setting, not a universal instrument specification. These indices describe vegetation response; they do not identify a disease. Cross-band systematic covariance is not established by the displayed uncertainty.

## Disease evaluation and replication

Obtain independent reference labels before model prediction. Record the label source/evidence ID. Live model scores are not accuracy. Unlabelled previews have no reference match, accuracy or F1. Record one paired observation per plant/specimen/model version; repeated frames do not create independent leaves. Several leaves from a plant remain a cluster. Current browser intervals bootstrap plants, and intervals from fewer than five plants are explicitly exploratory.

For the actual wavelength study, predefine physical-band conditions, counterbalance illumination order, acquire dark/white references per band, and collect labelled samples from multiple plants and relevant non-disease stresses. Train matched multiband/leave-one-band-out models and reserve a prospective plant-level test set before selection. The current RGB heads do not implement that experiment. The six trained camera-channel/augmentation ablations are reported separately on reused fixed datasets.

## Sources

- PlantCV raw hyperspectral calibration: https://docs.plantcv.org/en/stable/calibrate/
- EMVA 1288 specifications and distinction between linear/general models: https://www.emva.org/standards-technology/emva-1288/emva-standard-1288-downloads-2/
- W3C MediaStream Image Capture controls and exposure units: https://www.w3.org/TR/image-capture/
- Tomato early/late blight hyperspectral study: https://www.nature.com/articles/srep16564
- Leaf geometry and spectral calibration discussion: https://www.mdpi.com/1424-8220/23/1/44

## Replayable records

Every new specimen evaluation record stores its lossless 160×160 frozen input, a SHA-256 of the decoded RGBA pixels, model identity, locked reference, each condition's scores and timing. The browser's reference example cannot enter prospective trial counts. Run `python ml/replay_evaluation.py exported-ledger.json --public model` from the Streamlit archive to verify saved scores against the retained model.

Optical exports retain historical dark/white acquisitions by immutable acquisition ID, even after a profile is re-referenced. Each specimen record identifies its original reference IDs and calibration options. `node scripts/replay_optical.cjs exported-optical-session.json` recomputes the recorded ratios. Uncertainty replay does not establish omitted systematics or validate the physical apparatus.

## Prospective physical-band plan and paired endpoint

Before acquiring study samples, enter real plant/specimen pairs and documented illumination conditions (`centre_nm:detector_channel:FWHM_nm`). The first declared condition is the fixed baseline. The seed generates a reproducible condition permutation; cyclic rows rotate positions and alternate complete cycles reverse direction. Complete cycles balance positions, incomplete cycles may not. This is not a fully carryover-balanced design. Use separate plants as biological units; rounds are technical repetitions. When channels differ, the comparison is between illumination/detector systems, not a wavelength-only intervention.

The frozen JSON records the schedule, source evidence ID and project QC policy: at least 8 distinct live frames each for sample, dark and white; operator-declared exposure/gain/WB lock; selected-channel clipping ≤1%; absolute burst drift ≤2%; white spatial CV ≤10%; usable reference pixels ≥95%; white-minus-dark signal ≥10 DN; global response ratio within 0–1.2; references no older than 20 minutes. These thresholds are project rules, not certification limits. Camera-output checks cannot prove raw detector linearity, a physical control lock or complete calibration. Unobserved settings remain an operator declaration. Geometry and observed settings must match the specimen’s first accepted slot across bands and rounds.

Every acquired study attempt is retained, including missing-reference and QC failures. Only the first passing acquisition fills a slot; failed slots can be reacquired under the same frozen rules. Close a plan to end incomplete collection, preserving all pending slots and failures. A new plan creates a separate analysis population. The app does not mix historical or unplanned measurements into the plan.

For each comparison, only specimens with all declared rounds available for both the condition and baseline are included. Compute a paired difference per round; average within specimen, then within plant, then equally across plants. A percentile bootstrap resamples plant means 1,000 times using the recorded seed. Fewer than two plants has no interval; fewer than five remains explicitly exploratory. Comparison subsets may differ and counts are shown. Intervals describe variation across the observed plants, not LED spectral uncertainty, missingness bias, calibration systematics or a disease classifier. The output always uses the ROI-mean dark/white response ratio to avoid mixing quantities. Missing/failed slots and the number of retries remain visible. No p-values or optimal-wavelength claims are generated.

Export the plan and results with historical dark/white maps, burst statistics and timestamps. `node scripts/replay_study.cjs spectraleaf-physical-study.json` verifies the frozen-plan checksum, rebuilds the schedule/QC policy, recomputes the optical responses and acceptance decisions in acquisition order, then reproduces the plant-level summary. This is a reproducible numerical record rather than a signed or tamper-proof laboratory record. No physical apparatus has been validated by the software test suite.
