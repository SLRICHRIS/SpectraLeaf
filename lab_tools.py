"""Native Streamlit reference measurements and live matched ablation tables."""
import base64
import hashlib
import io
import json
import threading
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


def roi_pixels(image, roi):
    im = ImageOps.exif_transpose(image).convert("RGB")
    x, y, w, h = roi
    box = (round(x*im.width), round(y*im.height), round((x+w)*im.width), round((y+h)*im.height))
    return np.asarray(im.crop(box).resize((64,64), Image.Resampling.BILINEAR), dtype=float)


def ratios(sample, dark, white, channel):
    s,d,w = [a[:,:,channel] for a in (sample,dark,white)]
    denominator = float(w.mean()-d.mean())
    if denominator < 10:
        raise ValueError("White-minus-dark response is below 10 DN.")
    valid = (w-d)>=10
    if valid.mean()<.95:
        raise ValueError("More than 5% of reference pixels have an insufficient denominator.")
    return {"raw_DN":float(s.mean()),"dark_only_DN":float(s.mean()-d.mean()),
            "white_only_ratio":float(s.mean()/w.mean()) if w.mean() else None,
            "global_ratio":float((s.mean()-d.mean())/denominator),
            "flat_field_ratio":float(((s-d)[valid]/(w-d)[valid]).mean()),
            "selected_channel_clipping":float((s>=250).mean())}


@st.cache_resource
def ablation_runtime(path, expected_sha):
    import onnxruntime as ort
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha:
        raise ValueError("Ablation model integrity check failed.")
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(data,sess_options=options,providers=["CPUExecutionProvider"])
    report=json.loads((path.parent/"ablation_report.json").read_text())
    fixture=report["parity_fixture"];size=report["input_size"][0]
    tensor=((np.full((size,size,3),fixture["pixel_value"],dtype=np.float32)/255-np.asarray(report["mean"],dtype=np.float32))/np.asarray(report["std"],dtype=np.float32)).transpose(2,0,1)[None].astype(np.float32)
    actual=session.run(None,{"image":tensor})[0][0]
    if np.max(np.abs(actual-np.asarray(fixture["logits"])))>fixture["tolerance"]:
        raise ValueError("Six-head runtime numeric parity failed.")
    return session


def infer_conditions(image, model_dir, report, offset=0):
    session=ablation_runtime(model_dir/"ablation-heads.onnx",report["model_sha256"])
    size=report["input_size"][0]
    pixels=np.asarray(image.convert("RGB").resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
    results=[]
    for i in [(j+offset)%6 for j in range(6)]:
        key=report["variant_order"][i];variant=report["variants"][key];start=time.perf_counter();data=pixels.copy()
        if variant["transform"]=="gray":
            gray=(data*np.asarray([.2126,.7152,.0722],dtype=np.float32)).sum(2)
            data=np.repeat(gray[:,:,None],3,2)
        elif variant["transform"] in ("red","green","blue"):
            keep=["red","green","blue"].index(variant["transform"])
            data[:,:,[c for c in range(3) if c!=keep]]=0
        tensor=((data-np.asarray(report["mean"],dtype=np.float32))/np.asarray(report["std"],dtype=np.float32)).transpose(2,0,1)[None].astype(np.float32)
        prepared=time.perf_counter();logits=session.run(None,{"image":tensor})[0][0,i];end=time.perf_counter()
        scores=np.exp(logits-logits.max());scores/=scores.sum();pred=report["class_names"][int(scores.argmax())]
        results.append({"variant":key,"condition":variant["name"],"prediction":pred,"score":float(scores.max()),"scores":scores.tolist(),"preprocessing_ms":(prepared-start)*1000,"inference_ms":(end-prepared)*1000})
    return sorted(results,key=lambda r:report["variant_order"].index(r["variant"]))


def optical_page():
    st.title("Optical reference workbench")
    st.info("This native page records single images. The browser Optical bench also supports distinct-frame bursts, temporal uncertainty, camera controls, exposure-response tests and qualified physical-band indices.")
    st.link_button("Open full optical bench", "https://spectraleaf-lab.slrichris2005.chatgpt.site/#optics")
    st.caption("Webcam output is normally processed RGB. Values below are relative camera-response ratios, not independently established spectral reflectance.")
    cols=st.columns(3)
    rig=cols[0].text_input("Rig / settings profile","RIG-001")
    band=cols[1].text_input("Physical band / LED centre","white")
    channel=cols[2].selectbox("Detector channel",[0,1,2],format_func=lambda x:["Red","Green","Blue"][x])
    coords=st.columns(4)
    x=coords[0].number_input("ROI X fraction",0.,.9,.3,.05)
    y=coords[1].number_input("ROI Y fraction",0.,.9,.3,.05)
    w=coords[2].number_input("ROI width fraction",.1,1.,.4,.05)
    h=coords[3].number_input("ROI height fraction",.1,1.,.4,.05)
    roi=(x,y,min(w,1-x),min(h,1-y))
    source=st.radio("Optical input",["Webcam photograph","Upload photograph"],horizontal=True)
    file=st.camera_input("Acquire reference or specimen") if source=="Webcam photograph" else st.file_uploader("Optical image",type=["jpg","jpeg","png","webp"])
    kind=st.selectbox("Acquisition role",["dark","white","sample"])
    profile=hashlib.sha256(json.dumps([rig,band,roi,channel]).encode()).hexdigest()
    if "native_optics" not in st.session_state:st.session_state.native_optics={}
    if file and st.button("Record selected acquisition",type="primary"):
        with Image.open(file) as im:
            pixels=roi_pixels(im,roi)
        st.session_state.native_optics.setdefault(profile,{})[kind]={"pixels":pixels.tolist(),"time":datetime.now(timezone.utc).isoformat(),"role":kind,"profile":profile,"source":"one processed image"}
        st.success(f"Recorded {kind}. Repetition uncertainty is unavailable for a single image.")
    records=st.session_state.native_optics.get(profile,{})
    st.dataframe([{"role":k,"mean_DN":float(np.asarray(v["pixels"])[:,:,channel].mean()),"timestamp":v["time"],"frames":1} for k,v in records.items()],width="stretch")
    if all(k in records for k in ["sample","dark","white"]):
        try:
            values=ratios(*(np.asarray(records[k]["pixels"]) for k in ["sample","dark","white"]),channel)
            st.dataframe([{"processing_condition":k,"value":v} for k,v in values.items()],width="stretch")
            st.caption("Raw/dark-only values have DN units; referenced ratios are dimensionless. Different processing conditions are optical correction ablations, not disease-accuracy results. No temporal uncertainty is inferred from one image.")
        except ValueError as e:st.warning(str(e))
    st.download_button("Export native optical records",json.dumps({"format":"spectraleaf.native-optics.v1","records":st.session_state.native_optics},indent=2),"spectraleaf-native-optics.json","application/json")
    st.markdown("**Radiometric model:** `DN ≈ dark + gain × exposure × ∫ irradiance × reflectance × optical transmission × detector response dλ`, before nonlinear ISP processing. Matching dark/white references, stable geometry, known target reflectance and documented linearity are required before a physical reflectance interpretation.")


def evaluation_page(model_dir):
    st.title("Live matched ablation evaluation")
    report=json.loads((model_dir/"ablation_report.json").read_text())
    st.warning("These are experimental white-light tomato models with weak external validation. Camera RGB channel ablations do not identify an optimal physical illumination wavelength.")
    if "native_trials" not in st.session_state:st.session_state.native_trials=[]
    if "native_lock" not in st.session_state:st.session_state.native_lock=None
    cols=st.columns(2)
    plant=cols[0].text_input("Evaluation plant ID",disabled=bool(st.session_state.native_lock))
    leaf=cols[1].text_input("Evaluation specimen ID",disabled=bool(st.session_state.native_lock))
    label=st.selectbox("Independent reference",["unlabelled"]+report["class_names"],disabled=bool(st.session_state.native_lock))
    evidence=st.text_input("Reference evidence ID",disabled=bool(st.session_state.native_lock))
    independent=st.checkbox("The reference label is independent of model predictions.",disabled=bool(st.session_state.native_lock))
    consent=st.checkbox("The input is a tomato leaf under white light; enable experimental comparisons.")
    if st.button("Lock reference before prediction",disabled=bool(st.session_state.native_lock)):
        if not plant.strip() or not leaf.strip():st.error("Plant and specimen IDs are required.")
        elif label!="unlabelled" and not(independent and evidence.strip()):st.error("A scored reference needs independent evidence; otherwise choose unlabelled.")
        else:st.session_state.native_lock={"plant":plant.strip(),"specimen":leaf.strip(),"reference":label,"evidence":evidence,"scored":label!="unlabelled"};st.rerun()
    if st.button("New evaluation specimen"):
        st.session_state.native_lock=None;st.rerun()
    source=st.radio("Ablation input",["Saved session capture","Live webcam"],horizontal=True)
    holder=None;capture=None
    if source=="Live webcam":
        from streamlit_webrtc import webrtc_streamer,WebRtcMode
        if "ablation_frame_holder" not in st.session_state:st.session_state.ablation_frame_holder={"lock":threading.Lock(),"frame":None,"time":0.}
        holder=st.session_state.ablation_frame_holder
        def callback(frame):
            with holder["lock"]:holder["frame"]=frame;holder["time"]=time.monotonic()
            return frame
        webrtc_streamer(key="ablation-native-live",mode=WebRtcMode.SENDRECV,rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},media_stream_constraints={"video":True,"audio":False},video_frame_callback=callback,async_processing=True)
    else:
        records=st.session_state.get("captures",[])
        if records:capture=st.selectbox("Saved image for comparison",records,format_func=lambda r:f'{r["specimen"]} · {r["wavelength_nm"] or "white"}')
        else:st.info("Add a capture first, or choose Live webcam.")
    @st.fragment(run_every=2 if source=="Live webcam" else None)
    def live_table():
        if not(consent and st.session_state.native_lock):st.info("Lock the specimen reference and confirm the input conditions first.");return
        image=None
        if holder:
            with holder["lock"]:frame,age=holder["frame"],time.monotonic()-holder["time"]
            if frame is not None and age<3:image=frame.to_image()
        elif capture:
            if capture["wavelength_nm"] is not None or capture["crop"]!="Tomato":st.error("This model comparison requires a white-light tomato capture.");return
            image=Image.open(io.BytesIO(base64.b64decode(capture["image"].split(",",1)[1]))).convert("RGB")
        if image is None:st.info("Waiting for a fresh camera frame.");return
        cycle=st.session_state.get("ablation_native_cycle",0);st.session_state.ablation_native_cycle=cycle+1
        results=infer_conditions(image,model_dir,report,cycle%6);reference=st.session_state.native_lock
        st.dataframe([{"condition":r["condition"],"prediction":r["prediction"],"score":r["score"],"reference_match":("correct" if r["prediction"]==reference["reference"] else "incorrect") if reference["scored"] else "not scored","inference_ms":r["inference_ms"]} for r in results],width="stretch")
        st.caption("Same frozen image for all six actual inferences. Live table refreshes about every two seconds; each frame is not an independent sample. Model load/compilation is excluded from displayed inference timing.")
        if st.button("Record this specimen trial"):
            key=(reference["plant"],reference["specimen"],report["model_sha256"])
            if any((t["plant"],t["specimen"],t["model_sha256"])==key for t in st.session_state.native_trials):st.warning("This plant/specimen is already recorded.")
            else:st.session_state.native_trials.append({**reference,"model_sha256":report["model_sha256"],"timestamp":datetime.now(timezone.utc).isoformat(),"results":results});st.success("One paired specimen trial recorded.")
        valid=[t for t in st.session_state.native_trials if t["scored"]]
        st.dataframe([{"condition":report["variants"][key]["name"],"labelled_specimens":len(valid),"plants":len({t["plant"] for t in valid}),"accuracy":sum(next(r for r in t["results"] if r["variant"]==key)["prediction"]==t["reference"] for t in valid)/len(valid) if valid else None} for key in report["variant_order"]],width="stretch")
    live_table()
    st.subheader("Actual fixed-dataset ablation results")
    st.dataframe([{"condition":v["name"],"held_out_accuracy":v["results"]["test"]["accuracy"],"external_accuracy":v["results"]["external_test"]["accuracy"],"external_macro_f1":v["results"]["external_test"]["macro_f1"]} for v in report["variants"].values()],width="stretch")
    st.caption("Secondary comparison on previously audited fixed sets, not a fresh external confirmation. No variant is promoted based on test results. Full paired intervals and corrected tests are in the browser report.")
    st.download_button("Export native evaluation trials",json.dumps(st.session_state.native_trials,indent=2),"spectraleaf-native-evaluation.json","application/json")


def render_lab(page,model_dir):
    if page=="Optical bench":optical_page()
    else:evaluation_page(model_dir)
