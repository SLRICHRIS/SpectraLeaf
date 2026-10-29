"""Decode, validate, group duplicates and freeze leakage-aware split assignments."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

SEED = 20261029
CLASSES = ["healthy", "early_blight", "late_blight"]


def canonical_phash(image):
    values = []
    for flipped in (image, ImageOps.mirror(image)):
        for angle in (0, 90, 180, 270):
            tiny = flipped.rotate(angle, expand=True).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
            coeff = dctn(np.asarray(tiny, dtype=float), type=2, norm="ortho")[:8, :8].flatten()
            bits = coeff > np.median(coeff[1:])
            values.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    return min(values)


def leaf_identity(filename, source_class, leaf_map):
    identifier = filename.replace("_final_masked", "").split("___")[-1].split("copy")[0]
    for suffix in (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"):
        identifier = identifier.replace(suffix, "")
    matches = [v for v in leaf_map.get(identifier.lower().strip(), []) if v.startswith(source_class + ":::")]
    if len(set(matches)) != 1:
        return None
    return matches[0]


def curate(root, output):
    root, output = Path(root), Path(output)
    provenance = json.loads((output / "provenance.json").read_text())
    if provenance["download_failures"]:
        raise RuntimeError("Complete ingestion before curation.")
    ingested = [json.loads(line) for line in (output / "ingested.jsonl").read_text().splitlines()]
    leaf_map = json.loads((root / "source_metadata/leaf-map.json").read_text())
    rows, quarantine = [], []
    for record in ingested:
        try:
            data = (root / record["raw_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("raw_checksum_changed")
            with Image.open(root / record["raw_path"]) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
            if min(image.size) < 32 or max(image.size) > 12000:
                raise ValueError("invalid_image_dimensions")
            leaf_id = leaf_identity(Path(record["source_path"]).name, record["source_class"], leaf_map) if record["dataset"] == "plantvillage" else None
            if record["dataset"] == "plantvillage" and leaf_id is None:
                raise ValueError("missing_or_ambiguous_leaf_identity")
            pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            rows.append(dict(record, width=image.width, height=image.height, pixel_sha256=pixels,
                phash=f"{canonical_phash(image):016x}", leaf_id=leaf_id, label_id=CLASSES.index(record["label"])))
        except Exception as error:
            quarantine.append(dict(record, reason=str(error)))
    print(f"Decoded: {len(rows)}; quarantined: {len(quarantine)}", flush=True)
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    seen = {}
    for index, row in enumerate(rows):
        keys = [("bytes", row["sha256"]), ("pixels", row["pixel_sha256"])]
        if row["leaf_id"]:
            keys.append(("leaf", row["leaf_id"]))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            seen[key] = index
    # Conservative perceptual grouping also checks rotations and mirror variants.
    # This is not proof that every possible near duplicate has been removed.
    hashes = [int(row["phash"], 16) for row in rows]
    near_pairs = 0
    for i, h in enumerate(hashes):
        for j in range(i):
            if (h ^ hashes[j]).bit_count() <= 4:
                union(i, j)
                near_pairs += 1
    components = collections.defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    cleaned, duplicate_rows = [], []
    for indices in components.values():
        members = [rows[i] for i in indices]
        group = hashlib.sha256("|".join(sorted(r["sha256"] for r in members)).encode()).hexdigest()[:24]
        if len({r["label"] for r in members}) > 1:
            quarantine.extend(dict(r, reason="duplicate_group_has_conflicting_labels") for r in members)
            continue
        has_internal = any(r["dataset"] == "plantvillage" for r in members)
        unique_pixels = set()
        for row in sorted(members, key=lambda r: (r["dataset"] != "plantvillage", r["source_path"])):
            if has_internal and row["dataset"] != "plantvillage":
                quarantine.append(dict(row, reason="external_duplicate_of_internal_image"))
                continue
            if row["pixel_sha256"] in unique_pixels:
                duplicate_rows.append(dict(row, reason="exact_decoded_image_duplicate"))
                continue
            unique_pixels.add(row["pixel_sha256"])
            cleaned.append(dict(row, group_id=group))
    internal = [r for r in cleaned if r["dataset"] == "plantvillage"]
    external = [r for r in cleaned if r["dataset"] != "plantvillage"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(internal)), [r["label_id"] for r in internal], [r["group_id"] for r in internal]))
    test = set(folds[0][1]); validation = set(folds[1][1])
    for i, row in enumerate(internal):
        row["split"] = "test" if i in test else "validation" if i in validation else "train"
    for row in external:
        row["split"] = "external_test"
    final = sorted(internal + external, key=lambda r: (r["split"], r["source_path"]))
    for field in ("group_id", "sha256", "pixel_sha256", "leaf_id"):
        by_split = {s: {r[field] for r in final if r["split"] == s and r.get(field)} for s in ("train", "validation", "test", "external_test")}
        for a, values in by_split.items():
            for b, other in by_split.items():
                if a != b and values & other:
                    raise RuntimeError(f"Leakage detected: {field}: {a}/{b}")
    for split in ("train", "validation", "test"):
        if {r["label"] for r in final if r["split"] == split} != set(CLASSES):
            raise RuntimeError(f"Missing class in {split}")
    manifest_bytes = "".join(json.dumps(r, sort_keys=True) + "\n" for r in final).encode()
    (output / "manifest.jsonl").write_bytes(manifest_bytes)
    (output / "quarantine.jsonl").write_text("".join(json.dumps(r) + "\n" for r in quarantine + duplicate_rows))
    report = {"seed": SEED, "ingested": len(ingested), "accepted": len(final), "quarantined": len(quarantine),
        "exact_duplicates_removed": len(duplicate_rows), "perceptual_near_pairs": near_pairs,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "class_order": CLASSES,
        "splits": {s: dict(collections.Counter(r["label"] for r in final if r["split"] == s)) for s in ("train", "validation", "test", "external_test")},
        "groups": {s: len({r["group_id"] for r in final if r["split"] == s}) for s in ("train", "validation", "test", "external_test")},
        "quarantine_reasons": dict(collections.Counter(r["reason"] for r in quarantine + duplicate_rows)),
        "leakage_checks": {"group_overlap": 0, "byte_hash_overlap": 0, "pixel_hash_overlap": 0, "known_leaf_overlap": 0},
        "limitations": ["Leaf-level grouping is available; plant identifiers are unavailable, so plant-level independence cannot be established.",
            "Conservative perceptual hashing cannot guarantee removal of all near duplicates.",
            "PlantDoc labels are inherited from the source and have not been independently confirmed by a plant pathologist.",
            "PlantVillage is controlled RGB imagery; no wavelength or real webcam generalisation is established by its internal test split."]}
    (output / "data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="ml/artifacts")
    a = p.parse_args()
    curate(a.data_root, a.output)
