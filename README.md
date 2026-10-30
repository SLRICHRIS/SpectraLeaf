# SpectraLeaf — Streamlit leaf imaging lab

**Live app:** [Open SpectraLeaf on Streamlit](https://spectraleaf-nwsenequdnxogsxtzebdrg.streamlit.app/)

Show a plant through a continuous live webcam feed, capture individual frames without stopping it, or upload an image. Capture the same leaf under physical illumination bands that you supply. Record specimen IDs, crop, reference labels, wavelengths and notes; compare images and export a portable session. A trained three-class tomato RGB research model is bundled; raw datasets are not bundled.

**Measured limitation:** held-out PlantVillage accuracy was 95.2% on 584 images, but separate-source PlantDoc accuracy was 48.3% on 259 images, with 0% healthy-leaf recall on that source. This model is not ready for reliable field or webcam diagnosis. Experimental inference requires opt-in. It is trained for white-light RGB, not your physical wavelength experiment.

## Run on Windows

Use Python 3.11 or 3.12. Open a terminal inside this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` followed by `.venv\Scripts\python.exe -m streamlit run app.py` without activating.

## Deploy with GitHub

1. Create a GitHub repository, for example `spectraleaf-lab`.
2. Upload this folder's contents so that `app.py` and `requirements.txt` are at the repository root. Include `.streamlit/config.toml` for the theme.
3. In Streamlit Community Cloud, create an app from that repository and select `app.py`. Use Python 3.11 or 3.12.
4. Open the deployed HTTPS URL and grant webcam permission, or upload a photo.

The browser-hosted SpectraLeaf site is a separate JavaScript acquisition interface. Its session JSON files are interoperable with this Streamlit app. Streamlit is not running inside that site.

## Live webcam

Choose **Live webcam**, press **START** in the video component, and allow camera access. Select the required camera in the component's device settings. Use **Capture frame from live camera**, then label and add the frame to the session. The live video continues until **STOP** is pressed or you leave the capture page.

The Streamlit version uses `streamlit-webrtc`; live frames are sent to the Streamlit server and returned for display, with audio disabled. The browser-hosted interface displays its feed locally and offers a camera-source selector after the first camera permission.

Remote deployment needs HTTPS and a working WebRTC network path. A public STUN server is configured; restricted networks may require your own TURN service. Supply TURN URLs and credentials through Streamlit secrets and extend `rtc_configuration` rather than committing credentials. The **Take photograph** and **Upload image** modes remain available when live WebRTC cannot connect.

## Physical experiment

- The application labels illumination; it does not control LEDs or change the webcam's wavelength response.
- Keep leaf, camera distance, angle and illumination geometry fixed. Lock exposure and white balance where possible.
- Example wavelengths in the UI are starting labels, not experimentally established optimal disease bands. Enter the wavelength of the actual source used.
- RGB means include the whole frame and background. They are uncalibrated camera values, not reflectance, absorption, a measured spectrum or a diagnostic score.
- Any near-infrared experiment requires verified camera sensitivity; many webcams include IR-blocking filters.
- Give each leaf a unique specimen ID; also record plant IDs in notes or extend the manifest before a multi-plant study. Split training and evaluation by plant, never by image of the same plant.
- Disease labels require independent confirmation. RGB public datasets do not replace paired measurements captured with your LEDs.

## Bundled model and optional custom ONNX classifier

The Model setup page defaults to the bundled trained tomato baseline. The downloadable project includes `model/tomato-rgb.onnx`, its matching configuration, evaluation report and model card. In the Sites source checkout, the app can also locate these under `dist/model`. Select **Upload model** to supply a different model and configuration. The adapter supports one three-channel RGB image, one float32 4D input and one output class vector. It does **not** implement multi-band stacking, hyperspectral input, object detection, segmentation, binary single-logit output or quantised inputs.

Copy `model_config.example.json` and replace **every setting** with the preprocessing and class order used to train your model. The example is not a disease model or a guarantee of model compatibility.

- `input_size`: `[height, width]`.
- `layout`: `NCHW` or `NHWC`.
- `resize`: `stretch`, with bilinear interpolation; change code if training used a different crop/resize pipeline.
- `scale`, `mean`, `std`: applied as `(pixel * scale - mean) / std` in RGB order.
- `class_names`: exact output order.
- `output_type`: `logits` (softmax applied) or `probabilities` (validated).
- `illumination`: `"white"` or an integer wavelength that matches the capture. Train and validate for that exact input condition.

The model card reports the actual measured results and failed external release target. Evaluate any new version on independent plants from the actual camera and lighting setup. A high model score is not calibrated diagnostic confidence.

## Reproduce training and audit data lineage

`ml/` contains the actual ingestion, curation, training and validation code plus versioned artifact records. Install `ml/requirements.txt` separately from the smaller app runtime dependencies. See `ml/README.md` for the protocol.

For this downloadable Streamlit archive, use `--public model` when running `ml/train.py`, `ml/validate_bundle.py` or `ml/check_runtime.py`; the Sites source checkout uses `dist/model`. Raw downloads belong in a local `data/` directory and should not be committed. Preserve the split manifest and original evaluation records when collecting data for a new version.

## Data and persistence

Streamlit sessions are connection-scoped; export JSON with images before leaving. Up to 20 captures are supported per session. JSON can be imported in either interface. CSV is metadata only, without images. The hosted browser interface stores captures locally in the current browser; it has no central dataset storage. Streamlit processes uploaded images on its server in session memory.

## Dataset references

- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset
- Tomato Disease HSI: https://zenodo.org/records/15800414
- Oilseed rape multispectral: https://data.mendeley.com/datasets/ydmtggnzbw/1

Descriptions checked 29 October 2026; raw files not audited. Review the original data licences and cite the creators before use.

## Live evaluation, ablations and optical measurements

Choose **Live evaluation** for the six trained camera-channel/augmentation conditions, either on saved session captures or a fresh live webcam frame. Lock an independently obtained reference before prediction. The current-frame table refreshes on live input, but recorded specimen trials are counted once; model scores do not become accuracy when labels are missing. These are secondary comparisons on the existing fixed datasets, not fresh external confirmation or physical wavelength ablation.

Choose **Optical bench** for native single-image dark/white referencing and processing comparisons. This native page does not manufacture temporal uncertainty from one photograph. The full browser bench under `web/` supports distinct-frame bursts, partial temporal uncertainty, camera control verification, exposure-response checks, evidence-qualified physical-band indices and plant-bootstrap evaluation intervals.

To run the complete browser interface locally on Windows, open a terminal inside the extracted project and run `python -m http.server 8000 --directory web`, then open `http://localhost:8000`. Localhost is required for local webcam permissions; remote use needs HTTPS. The server is a local preview and does not train a model or control external LEDs.

See `docs/OPTICAL_PROTOCOL.md`, `model/ABLATION_REPORT.md` and the source protocol/prediction records. The single attributed held-out PlantVillage demonstration image is already in the reported benchmark and is excluded from prospective user-study counts. No real physical-band experiment or new webcam study has been performed in the shipped empty ledgers.

The included browser app now also supports a frozen prospective physical-band plan, seeded acquisition positions, retained QC failures and a live paired response table with plant-bootstrap intervals. Launch `web/` for that workflow; the native Streamlit optical page remains single-image. Replay a browser study export with `node scripts/replay_study.cjs spectraleaf-physical-study.json`.

© 2026 S L Richris. All rights reserved. Third-party materials retain their respective licences and required attributions.
