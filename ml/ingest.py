"""Download immutable, checksum-verified source images. Raw files are never modified."""
import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests

PV_REV = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
PD_REV = "5467f6012d78d1c446145d5f582da6096f852ae8"
CLASS_MAP = {"Tomato___healthy": "healthy", "Tomato___Early_blight": "early_blight", "Tomato___Late_blight": "late_blight"}
PD_MAP = {"Tomato leaf": "healthy", "Tomato Early blight leaf": "early_blight", "Tomato leaf late blight": "late_blight"}
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def get(url):
    for attempt in range(4):
        try:
            response = session().get(url, timeout=(15, 45))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(min(attempt + 1, 3))


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def ingest(root, output, workers=16):
    root, output = Path(root), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    metadata = root / "source_metadata"
    metadata.mkdir(exist_ok=True)
    pv_repo = "spMohanty/PlantVillage-Dataset"
    pd_repo = "pratikkayal/PlantDoc-Dataset"
    directory = get(f"https://api.github.com/repos/{pv_repo}/contents/raw/color?ref={PV_REV}").json()
    for entry in directory:
        if entry["name"] not in CLASS_MAP:
            continue
        tree = get(f"https://api.github.com/repos/{pv_repo}/git/trees/{entry['sha']}").json()
        if tree.get("truncated"):
            raise RuntimeError("Source class listing is truncated; refusing incomplete ingestion.")
        for item in tree["tree"]:
            if item["type"] == "blob" and Path(item["path"]).suffix.lower() in (".jpg", ".jpeg", ".png"):
                path = f"raw/color/{entry['name']}/{item['path']}"
                sources.append(dict(dataset="plantvillage", revision=PV_REV, label=CLASS_MAP[entry["name"]],
                    source_class=entry["name"], source_path=path, git_blob_sha1=item["sha"],
                    source_url=f"https://raw.githubusercontent.com/{pv_repo}/{PV_REV}/{quote(path)}"))
    pd_tree = get(f"https://api.github.com/repos/{pd_repo}/git/trees/{PD_REV}?recursive=1").json()
    if pd_tree.get("truncated"):
        raise RuntimeError("PlantDoc source listing is truncated.")
    for item in pd_tree["tree"]:
        parts = item["path"].split("/")
        if item["type"] == "blob" and len(parts) == 3 and parts[1] in PD_MAP:
            # Both original PlantDoc splits form an external-only evaluation set.
            # None of these files can enter training, selection or calibration.
            sources.append(dict(dataset="plantdoc_external", revision=PD_REV, label=PD_MAP[parts[1]],
                source_class=parts[1], source_path=item["path"], git_blob_sha1=item["sha"],
                source_url=f"https://raw.githubusercontent.com/{pd_repo}/{PD_REV}/{quote(item['path'])}"))
    leaf_bytes = get(f"https://raw.githubusercontent.com/{pv_repo}/{PV_REV}/leaf-map.json").content
    (metadata / "leaf-map.json").write_bytes(leaf_bytes)
    for dataset, repo, revision in [("plantvillage", pv_repo, PV_REV), ("plantdoc_external", pd_repo, PD_REV)]:
        # The creator's published dataset card declares the PV dataset licence.
        name = "README_HF.md" if dataset == "plantvillage" else "LICENSE.txt"
        (metadata / f"{dataset}_{name}").write_bytes(get(f"https://raw.githubusercontent.com/{repo}/{revision}/{name}").content)
    (output / "source_inventory.json").write_text(json.dumps(sources, indent=2))
    print(f"Source inventory: {len(sources)} images", flush=True)

    def download(record):
        path = root / "raw" / record["dataset"] / record["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = path.read_bytes()
            if blob_sha(data) != record["git_blob_sha1"]:
                raise RuntimeError(f"Existing raw file failed checksum: {path}")
        else:
            data = get(record["source_url"]).content
            if blob_sha(data) != record["git_blob_sha1"]:
                raise RuntimeError(f"Downloaded source checksum mismatch: {record['source_path']}")
            temp = path.with_suffix(path.suffix + ".partial")
            temp.write_bytes(data)
            temp.replace(path)
        return dict(record, raw_path=str(path.relative_to(root)), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))

    complete, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download, r): r for r in sources}
        for future in concurrent.futures.as_completed(jobs):
            try:
                complete.append(future.result())
            except Exception as error:
                failed.append(dict(jobs[future], error=str(error)))
            if (len(complete) + len(failed)) % 250 == 0:
                print(f"Verified {len(complete)}/{len(sources)}; failures {len(failed)}", flush=True)
    complete.sort(key=lambda r: (r["dataset"], r["source_path"]))
    (output / "ingested.jsonl").write_text("".join(json.dumps(r) + "\n" for r in complete))
    (output / "ingest_failures.json").write_text(json.dumps(failed, indent=2))
    (output / "provenance.json").write_text(json.dumps({"plantvillage_revision": PV_REV, "plantdoc_revision": PD_REV,
        "plantvillage_dataset_license": "CC-BY-SA-3.0", "plantdoc_license": "CC-BY-4.0",
        "leaf_map_sha256": hashlib.sha256(leaf_bytes).hexdigest(), "selected_classes": CLASS_MAP,
        "expected_images": len(sources), "verified_images": len(complete), "download_failures": len(failed)}, indent=2))
    if failed:
        raise RuntimeError("Ingestion incomplete. Retry to resume verified files; no training is allowed until complete.")
    print("Ingestion complete; every file matches its immutable Git blob checksum.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="ml/artifacts")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    ingest(a.data_root, a.output, a.workers)
