"""SpectraLeaf: physical-illumination leaf acquisition and optional RGB ONNX inference."""
import base64
import csv
import hashlib
import io
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="SpectraLeaf | Leaf Imaging Lab", page_icon="🌿", layout="wide")
st.markdown("""<style>.stApp{background:#f6f7f2}h1,h2,h3{color:#204d3b}
div[data-testid='stMetric']{background:white;border:1px solid #e0e6d8;padding:16px;border-radius:12px}
.stCaption{color:#7b8971}</style>""", unsafe_allow_html=True)
if "captures" not in st.session_state:
    st.session_state.captures = []


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((960, 960))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=86)
    return image, "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def metrics(image):
    pixels = np.asarray(image, dtype=np.float64)
    means = pixels.mean(axis=(0, 1))
    return dict(r=round(float(means[0]), 1), g=round(float(means[1]), 1),
                b=round(float(means[2]), 1), brightness=round(float(means @ [.2126, .7152, .0722]), 1),
                clipped_pct=round(float((pixels >= 250).any(axis=2).mean() * 100), 1))


def decode_image(data):
    if not isinstance(data, str) or len(data) > 2_500_000:
        raise ValueError("Invalid or oversized capture image.")
    prefix, encoded = data.split(",", 1)
    if prefix not in {"data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"}:
        raise ValueError("Unsupported capture image format.")
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        return im.convert("RGB")


def band_label(record):
    return "White light" if record["wavelength_nm"] is None else f'{record["wavelength_nm"]} nm'


def export_session():
    return json.dumps({"format": "spectraleaf.session.v1", "exported_at": datetime.now(timezone.utc).isoformat(),
                       "measurement_note": "Whole-frame uncalibrated camera RGB values, not reflectance.",
                       "captures": st.session_state.captures}, indent=2)


with st.sidebar:
    st.title("🌿 SpectraLeaf")
    st.caption("LEAF IMAGING LAB")
    page = st.radio("Workspace", ["Capture studio", "Compare captures", "Dataset library", "Model setup", "Session tools", "Optical bench", "Live evaluation"])
    st.divider()
    st.metric("Session captures", len(st.session_state.captures))
    st.caption("Session state lasts for this Streamlit connection. Export JSON to keep your images and metadata.")
    st.caption("Your LEDs are controlled separately. Wavelength labels record the physical illumination you used.")

if page == "Capture studio":
    st.caption("CONTROLLED LIGHT. MEANINGFUL DATA.")
    st.title("A new perspective on plant health.")
    st.write("Photograph the same leaf under each selected light. Keep its specimen ID unchanged across bands.")
    left, right = st.columns([1.3, 1])
    with left:
        source = st.radio("Image source", ["Live webcam", "Take photograph", "Upload image"], horizontal=True)
        current = None
        image_file = None
        if source == "Live webcam":
            try:
                from streamlit_webrtc import WebRtcMode, webrtc_streamer
                if "live_frame_state" not in st.session_state:
                    st.session_state.live_frame_state = {"lock": threading.Lock(), "frame": None, "timestamp": 0.0}
                frame_state = st.session_state.live_frame_state

                def receive_video(frame):
                    # This callback runs off the Streamlit thread; do not call st.* here.
                    with frame_state["lock"]:
                        frame_state["frame"] = frame
                        frame_state["timestamp"] = time.monotonic()
                    return frame

                ctx = webrtc_streamer(
                    key="spectraleaf-live", mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": {"width": {"ideal": 960}, "height": {"ideal": 720}}, "audio": False},
                    video_frame_callback=receive_video, async_processing=True,
                )
                st.caption("Start the camera above and allow access. The plant stays live while you capture frames. Use STOP to end the feed.")
                if st.button("Capture frame from live camera", disabled=not ctx.state.playing):
                    with frame_state["lock"]:
                        frame, timestamp = frame_state["frame"], frame_state["timestamp"]
                    if frame is None or time.monotonic() - timestamp > 3:
                        st.warning("Waiting for a fresh camera frame. Try Capture frame again once the live image is visible.")
                    else:
                        _, saved = prepare_image(frame.to_image())
                        st.session_state.live_draft = saved
                if st.session_state.get("live_draft"):
                    current, current_data = prepare_image(decode_image(st.session_state.live_draft))
                    st.image(current, caption="Captured frame ready to save · live feed continues above", width="stretch")
                st.caption("Streamlit live video is processed on the app server. If the connection fails on a restricted network, use Take photograph or Upload image.")
            except ImportError:
                st.error("Live webcam needs streamlit-webrtc. Install the dependencies in requirements.txt, then restart the app.")
        else:
            image_file = st.camera_input("Photograph the leaf") if source == "Take photograph" else st.file_uploader(
                "Upload a leaf photograph", type=["jpg", "jpeg", "png", "webp"])
        if image_file:
            try:
                if image_file.size > 12 * 1024 * 1024:
                    raise ValueError("Image must be under 12 MB.")
                with Image.open(image_file) as uploaded:
                    current, current_data = prepare_image(uploaded)
                st.image(current, caption=f"Saved resolution: {current.width} × {current.height} px", width="stretch")
            except Exception as error:
                st.error(f"Could not read this image: {error}")
        st.caption("Fix camera position and leaf orientation. Lock exposure and white balance where supported.")
    with right:
        st.subheader("Label this capture")
        specimen = st.text_input("Specimen ID", "LEAF-001", max_chars=60)
        crop = st.selectbox("Plant species", ["Tomato", "Potato", "Mango", "Chilli", "Grape", "Other"])
        condition = st.selectbox("Reference label", ["unlabelled", "healthy", "diseased", "other-stress"])
        light = st.selectbox("Physical illumination", ["White light", "450 nm", "530 nm", "660 nm", "Custom"])
        wavelength = None if light == "White light" else int(light.split()[0]) if light != "Custom" else int(
            st.number_input("Centre wavelength (nm)", 350, 1100, 720, 1))
        st.caption("This records your light source; it does not switch LEDs or change camera sensitivity.")
        notes = st.text_area("Observation notes", max_chars=500, placeholder="Disease name, light source, exposure, symptoms…")
        if st.button("Add to session", type="primary", disabled=current is None):
            if not specimen.strip():
                st.error("Enter a specimen ID.")
            elif len(st.session_state.captures) >= 20:
                st.error("Export and clear this session before adding more than 20 captures.")
            else:
                st.session_state.captures.append(dict(id=str(uuid.uuid4()), specimen=specimen.strip(), crop=crop,
                    condition=condition, wavelength_nm=wavelength, notes=notes,
                    created_at=datetime.now(timezone.utc).isoformat(), image=current_data,
                    width=current.width, height=current.height, metrics=metrics(current), source=source.lower()))
                if source == "Live webcam":
                    st.session_state.pop("live_draft", None)
                st.success("Capture saved. Change your physical light and take the next photograph.")
    st.info("A trained white-light tomato RGB baseline is available in Model setup. Separate-source accuracy is limited; experimental predictions require opt-in.")

elif page == "Compare captures":
    st.title("Compare your observations.")
    records = st.session_state.captures
    if not records:
        st.info("Add images in Capture studio first.")
    else:
        chosen = st.selectbox("Specimen", list(dict.fromkeys(r["specimen"] for r in records)))
        subset = [r for r in records if r["specimen"] == chosen]
        columns = st.columns(min(3, len(subset)))
        for index, record in enumerate(subset):
            with columns[index % len(columns)]:
                st.image(decode_image(record["image"]), caption=band_label(record), width="stretch")
                st.caption(f'{record["crop"]} · {record["condition"]} · {record["notes"]}')
                st.download_button("Download image", base64.b64decode(record["image"].split(",")[1]),
                    file_name=f'leaf-{index + 1}.jpg', mime="image/jpeg", key="dl-"+record["id"])
        st.subheader("Mean camera channel values")
        st.caption("Whole-frame RGB means (0–255), including background. These are not calibrated reflectance or a measured spectrum.")
        st.dataframe([{"illumination": band_label(r), "mean_R": r["metrics"]["r"], "mean_G": r["metrics"]["g"],
            "mean_B": r["metrics"]["b"], "brightness": r["metrics"]["brightness"],
            "near_saturation_%": r["metrics"]["clipped_pct"]} for r in subset], width="stretch")

elif page in ["Optical bench", "Live evaluation"]:
    from lab_tools import render_lab
    lab_model_dir = Path(__file__).resolve().parent / "model"
    if not lab_model_dir.exists():
        lab_model_dir = Path(__file__).resolve().parent.parent / "dist/model"
    render_lab(page, lab_model_dir)

elif page == "Dataset library":
    st.title("Choose the right evidence.")
    st.markdown("""
    - [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset): RGB baseline; 54,306 images.
    - [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset): varied RGB photographs; authors report 2,598 data points.
    - [Tomato Disease HSI](https://zenodo.org/records/15800414): reported 400–950 nm measurements, about 6 GB in MATLAB files.
    - [Oilseed rape multispectral data](https://data.mendeley.com/datasets/ydmtggnzbw/1): light leaf spot experiments; ENVI-compatible RAW files.
    """)
    st.warning("A normal webcam cannot supply hyperspectral inputs. Use spectral data to explore bands, then validate on data from your own camera and lights.")
    st.caption("Dataset descriptions checked 29 October 2026. Raw files have not been audited.")

elif page == "Model setup":
    st.title("Evaluate a trained RGB classifier.")
    st.write("Use the bundled tomato RGB research baseline or supply a compatible ONNX classifier and its configuration.")
    st.warning("This adapter accepts three-channel RGB images only. It does not support hyperspectral cubes or multiple-illumination stacks. A model trained for white light should only receive matching white-light images.")
    choice = st.radio("Model source", ["Bundled tomato baseline", "Upload model"], horizontal=True)
    model_bytes = config_bytes = None
    consent = True
    if choice == "Bundled tomato baseline":
        model_dir = Path(__file__).resolve().parent / "model"
        if not model_dir.exists():
            model_dir = Path(__file__).resolve().parent.parent / "dist/model"
        if (model_dir / "tomato-rgb.onnx").exists():
            model_bytes = (model_dir / "tomato-rgb.onnx").read_bytes()
            config_bytes = (model_dir / "model_config.json").read_bytes()
            report = json.loads((model_dir / "training_report.json").read_text())
            st.dataframe([{"evaluation": key, "images": report["results"][key]["n"],
                "accuracy": report["results"][key]["accuracy"], "macro_f1": report["results"][key]["macro_f1"]}
                for key in ["validation", "test", "external_test"]], width="stretch")
            st.warning(f"Separate-source accuracy is {report['results']['external_test']['accuracy']:.1%}; healthy-leaf recall on that source is {report['results']['external_test']['per_class']['healthy']['recall']:.1%}. This model is not ready for reliable field diagnosis. No wavelength-based model is included.")
            consent = st.checkbox("Enable experimental predictions despite the limited separate-source performance.")
        else:
            st.error("Bundled model files are missing. Download the complete project archive or upload a compatible model.")
    else:
        model_file = st.file_uploader("Trained ONNX model", type=["onnx"])
        config_file = st.file_uploader("Model configuration JSON", type=["json"])
        if model_file and config_file:
            model_bytes, config_bytes = model_file.getvalue(), config_file.getvalue()
    st.caption("Use model_config.example.json as a template. The input size, layout, normalisation, class order and output type must match training exactly.")
    if st.session_state.captures:
        record = st.selectbox("Capture to classify", st.session_state.captures,
            format_func=lambda r: f'{r["specimen"]} · {band_label(r)} · {r["created_at"]}')
        if st.button("Run selected model", disabled=not(model_bytes and config_bytes and consent), type="primary"):
            try:
                import onnxruntime as ort
                config = json.loads(config_bytes)
                if config.get("model_sha256") and hashlib.sha256(model_bytes).hexdigest() != config["model_sha256"]:
                    raise ValueError("Model file checksum does not match the configuration.")
                if config.get("crop") and record["crop"] != config["crop"]:
                    raise ValueError("This capture's plant species is outside the selected model scope.")
                if config.get("input_type") != "rgb_image":
                    raise ValueError("Configuration must explicitly declare input_type: rgb_image.")
                expected_light = config.get("illumination", "white")
                actual_light = "white" if record["wavelength_nm"] is None else record["wavelength_nm"]
                if expected_light != actual_light:
                    raise ValueError("This capture illumination does not match the model configuration.")
                classes = config["class_names"]
                if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(c, str) for c in classes):
                    raise ValueError("class_names must contain at least two class labels.")
                height, width = map(int, config["input_size"])
                if not (16 <= height <= 2048 and 16 <= width <= 2048):
                    raise ValueError("Input dimensions must be 16–2048 pixels.")
                if config["layout"] not in ("NCHW", "NHWC"):
                    raise ValueError("layout must be NCHW or NHWC.")
                if config["output_type"] not in ("logits", "probabilities"):
                    raise ValueError("output_type must be logits or probabilities.")
                if config.get("resize") != "stretch":
                    raise ValueError("This adapter supports resize: stretch only. Match your training preprocessing.")
                mean, std = np.asarray(config["mean"], dtype=np.float32), np.asarray(config["std"], dtype=np.float32)
                scale = float(config["scale"])
                if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any() or not np.isfinite(scale):
                    raise ValueError("Provide finite three-channel mean/std and a finite scale; std must be positive.")
                im = decode_image(record["image"]).resize((width, height), Image.Resampling.BILINEAR)
                tensor = (np.asarray(im, dtype=np.float32) * scale - mean) / std
                if config["layout"] == "NCHW":
                    tensor = tensor.transpose(2, 0, 1)
                tensor = tensor[None].astype(np.float32)
                session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
                if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
                    raise ValueError("Model must have exactly one input and one output.")
                inp = session.get_inputs()[0]
                if inp.type != "tensor(float)" or len(inp.shape) != 4:
                    raise ValueError("Model input must be a four-dimensional float32 tensor.")
                if any(isinstance(dim, int) and dim != size for dim, size in zip(inp.shape, tensor.shape)):
                    raise ValueError("Model input shape does not match the configuration.")
                raw_output = np.asarray(session.run(None, {inp.name: tensor})[0])
                if raw_output.shape not in ((len(classes),), (1, len(classes))):
                    raise ValueError("Output must be one class vector for one image.")
                scores = raw_output.reshape(-1).astype(float)
                if not np.isfinite(scores).all():
                    raise ValueError("Model returned non-finite scores.")
                if config["output_type"] == "logits":
                    scores = np.exp(scores - scores.max()); scores /= scores.sum()
                elif (scores < 0).any() or (scores > 1).any() or not np.isclose(scores.sum(), 1, atol=.01):
                    raise ValueError("Declared probabilities must be in [0,1] and sum to one.")
                best = int(scores.argmax())
                st.success(f"Selected model prediction: {classes[best]}")
                st.dataframe([{"class": name, "model_score": float(score)} for name, score in zip(classes, scores)])
                st.caption("Model scores are not calibrated diagnostic confidence. Validate this model on independent plants from your actual setup.")
            except Exception as error:
                st.error(f"Could not run the selected model: {error}")
    else:
        st.info("Add a capture before running inference.")

elif page == "Session tools":
    st.title("Keep your experiment portable.")
    st.download_button("Export session JSON with images", export_session(), "spectraleaf-session.json", "application/json",
        disabled=not st.session_state.captures)
    if st.session_state.captures:
        output = io.StringIO()
        fields = ["id", "specimen", "crop", "condition", "wavelength_nm", "created_at", "notes", "width", "height"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for r in st.session_state.captures:
            writer.writerow({k: ("'"+str(v) if str(v).startswith(("=", "+", "-", "@")) else v) for k, v in r.items()})
        st.download_button("Export CSV manifest", output.getvalue(), "spectraleaf-manifest.csv", "text/csv")
    file = st.file_uploader("Import a SpectraLeaf session", type=["json"])
    if st.button("Import captures", disabled=file is None):
        try:
            if file.size > 35 * 1024 * 1024:
                raise ValueError("Session file must be under 35 MB.")
            data = json.loads(file.getvalue())
            if data.get("format") != "spectraleaf.session.v1" or not isinstance(data.get("captures"), list):
                raise ValueError("Choose a SpectraLeaf session JSON.")
            if len(st.session_state.captures) + len(data["captures"]) > 20:
                raise ValueError("Import would exceed 20 captures. Export and clear the current session first.")
            imported = []
            for r in data["captures"]:
                if not isinstance(r.get("specimen"), str) or not r["specimen"].strip():
                    raise ValueError("Each capture needs a specimen ID.")
                wave = r.get("wavelength_nm")
                if wave is not None and (not isinstance(wave, int) or isinstance(wave, bool) or not 350 <= wave <= 1100):
                    raise ValueError("Invalid wavelength in imported capture.")
                im, image_data = prepare_image(decode_image(r["image"]))
                imported.append(dict(id=str(uuid.uuid4()), specimen=r["specimen"].strip()[:60], crop=str(r.get("crop", "Other"))[:60],
                    condition=str(r.get("condition", "unlabelled"))[:80], wavelength_nm=wave, notes=str(r.get("notes", ""))[:500],
                    created_at=str(r.get("created_at", datetime.now(timezone.utc).isoformat()))[:50], image=image_data,
                    width=im.width, height=im.height, metrics=metrics(im), source="import"))
            st.session_state.captures.extend(imported)
            st.success(f"Imported {len(imported)} captures.")
        except Exception as error:
            st.error(f"Could not import session: {error}")
    st.divider()
    confirm_clear = st.checkbox("I have exported anything I need to keep.")
    if st.button("Clear session", disabled=not confirm_clear):
        st.session_state.captures = []
        st.rerun()

st.divider()
st.caption("© 2026 S L Richris. All rights reserved.")
st.caption("SpectraLeaf · Research prototype. Physical illumination is supplied by your experimental setup. Bundled RGB model: three tomato classes; separate-source performance is limited.")
