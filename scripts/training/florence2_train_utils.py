"""Shared Florence-2 OMR training utilities (from Colab notebooks).

Used by train_florence2_base.py and train_florence2_large.py.
Train wrappers set module-level path/config globals before calling helpers.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback

# Config globals (overwritten by train wrappers)
GOLDEN_DIR = None
MANIFEST_PATH = None
IMAGES_DIR = None
GT_DIR = None
DRIVE_OUTPUT_DIR = None
EVAL_GOLDEN_EPOCH_DIR = None
EVAL_GOLDEN_EPOCH_HISTORY_PATH = None
EVAL_GOLDEN_EPOCH_WRITEUP_PATH = None
EVAL_GOLDEN_BEFORE_PATH = None
EVAL_GOLDEN_AFTER_PATH = None
MANIFEST_VERSION = 5
GOLDEN_BENCHMARK_COUNT = 100
STYLE_WEIGHTS = {"handwritten": 0.50, "clean": 0.25, "camera": 0.25}
STYLE_WEIGHT_CLEAN_VS_CORRUPT = 0.50
USE_CONSTRAINED_DECODE = True
EVAL_GOLDEN_EVERY_EPOCH = True
DEVICE = "cuda"
TASK_PROMPT = "<OCR>"
MODEL_ID = ""
SEED = 1337

# Injected by wrappers after importing florence2_primus
make_primus_logits_processor = None
encode_semantic_labels = None

# Set by train wrappers before StyleBalancedTrainer / GoldenEpochEvalCallback run
processor = None
golden_records = None
baseline_results = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

GROUP_IMAGE_DIRS: dict[str, tuple[str, ...]] = {
    "handwritten_original": ("original", "images", "handwritten"),
    "clean_original": ("original", "images", "clean"),
    "camera_distorted_original": ("original", "images", "camera"),
    "handwritten_corrupt": ("corrupt", "images", "handwritten"),
    "clean_corrupt": ("corrupt", "images", "clean"),
    "camera_distorted_corrupt": ("corrupt", "images", "camera"),
    "manual_handwritten": ("manual", "handwritten"),
    "manual_digital": ("manual", "digital"),
}

GROUP_ORDER = [
    "handwritten_original",
    "clean_original",
    "camera_distorted_original",
    "handwritten_corrupt",
    "clean_corrupt",
    "camera_distorted_corrupt",
    "manual_handwritten",
    "manual_digital",
]

GOLDEN_BENCHMARK_ENTRIES = [
    {"group": "handwritten_original", "filename": "000132946-1_21_1.jpg"},
    {"group": "handwritten_original", "filename": "230000659-1_3_1.jpg"},
    {"group": "handwritten_original", "filename": "220010817-1_18_1.jpg"},
    {"group": "handwritten_original", "filename": "220034811-1_2_1.jpg"},
    {"group": "handwritten_original", "filename": "190019918-1_1_1.jpg"},
    {"group": "handwritten_original", "filename": "212001709-1_4_1.jpg"},
    {"group": "handwritten_original", "filename": "150203639-1_2_1.jpg"},
    {"group": "handwritten_original", "filename": "220001519-1_1_1.jpg"},
    {"group": "handwritten_original", "filename": "000128225-1_1_1.jpg"},
    {"group": "handwritten_original", "filename": "000132422-1_1_1.jpg"},
    {"group": "clean_original", "filename": "000102395-1_1_1.png"},
    {"group": "clean_original", "filename": "000114963-1_1_1.png"},
    {"group": "clean_original", "filename": "110002051-1_1_1.png"},
    {"group": "clean_original", "filename": "220031110-1_1_1.png"},
    {"group": "clean_original", "filename": "000104407-1_1_1.png"},
    {"group": "clean_original", "filename": "190015389-1_1_1.png"},
    {"group": "clean_original", "filename": "000112762-1_1_1.png"},
    {"group": "clean_original", "filename": "000109599-1_2_1.png"},
    {"group": "clean_original", "filename": "212002605-1_1_1.png"},
    {"group": "clean_original", "filename": "000111721-1_1_1.png"},
    {"group": "camera_distorted_original", "filename": "000118903-1_1_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "000111686-1_1_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "210000061-1_4_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "230003675-1_30_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "000115397-1_1_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "220019566-1_1_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "210000094-1_7_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "190004306-1_1_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "000101628-1_1_1_distorted.jpg"},
    {"group": "camera_distorted_original", "filename": "211002852-1_1_1_distorted.jpg"},
    {"group": "handwritten_corrupt", "filename": "e3863145ef3e42afae97cd349163d0c6.jpg"},
    {"group": "handwritten_corrupt", "filename": "5e11951664a3457e9739d93a3171aa48.jpg"},
    {"group": "handwritten_corrupt", "filename": "86f558bff0ed4074ab55b765a9ba584d.jpg"},
    {"group": "handwritten_corrupt", "filename": "a8e9d04340e94054bb15e748c7927e1c.jpg"},
    {"group": "handwritten_corrupt", "filename": "4523f7f2dc624128bdcbc82bc118d335.jpg"},
    {"group": "handwritten_corrupt", "filename": "1bdd86927d97410db65cc7d1799735ae.jpg"},
    {"group": "handwritten_corrupt", "filename": "6717afa4e0144f71bd2c6300540be8d5.jpg"},
    {"group": "handwritten_corrupt", "filename": "dbd18f2c3b5144e9a937d85be0925c7a.jpg"},
    {"group": "handwritten_corrupt", "filename": "5537a1a3a4044c2997cf075c09e467fb.jpg"},
    {"group": "handwritten_corrupt", "filename": "6308c109bd69476893997da0bad95dc6.jpg"},
    {"group": "clean_corrupt", "filename": "aabc4e73b470487e92eec3a6f5f2dce8.png"},
    {"group": "clean_corrupt", "filename": "0ffd9e6dc1f742fabab24fafcfc72579.png"},
    {"group": "clean_corrupt", "filename": "95c0aa4542a8441790d08805168f182b.png"},
    {"group": "clean_corrupt", "filename": "beb2ea01c54c41d4bac1a9a7b424febf.png"},
    {"group": "clean_corrupt", "filename": "9e1d9aae9505446f8be8d8a92494c20d.png"},
    {"group": "clean_corrupt", "filename": "729021847bb641f0a7e840a64d993bf4.png"},
    {"group": "clean_corrupt", "filename": "9bc9e120643948ee8788e4295ecc22f1.png"},
    {"group": "clean_corrupt", "filename": "a0d9c58a64a74b4782cd135d353a33dc.png"},
    {"group": "clean_corrupt", "filename": "4541abd5c9d14406b27133995ce9ff23.png"},
    {"group": "clean_corrupt", "filename": "d2e5f36027ea4cb582f53c3eb3d3e59c.png"},
    {"group": "camera_distorted_corrupt", "filename": "d4eb58a8ff2640eab7a6a93bc4211a03.jpg"},
    {"group": "camera_distorted_corrupt", "filename": "ca4c0489b165467789f2146a2afeb74b.png"},
    {"group": "camera_distorted_corrupt", "filename": "37cbb51c3b2049b98db0c16fccb47e47.png"},
    {"group": "camera_distorted_corrupt", "filename": "5ea6e4a124894bb9839399772ed925f6.png"},
    {"group": "camera_distorted_corrupt", "filename": "982f0e40df14456795320bb9b26db89a.png"},
    {"group": "camera_distorted_corrupt", "filename": "e56c78c7987142b7a874540a8dab923d.png"},
    {"group": "camera_distorted_corrupt", "filename": "dcf91aef10f246a58325f20a889d7de7.png"},
    {"group": "camera_distorted_corrupt", "filename": "e6692210188a45ab8e7673d9719389d5.png"},
    {"group": "camera_distorted_corrupt", "filename": "4eb68ed427b24f0cb64c77f6656b1362.png"},
    {"group": "camera_distorted_corrupt", "filename": "2051043cf00b482aa76c122af3ab199b.png"},
    {"group": "manual_handwritten", "filename": "gs0.png"},
    {"group": "manual_handwritten", "filename": "gs1.png"},
    {"group": "manual_handwritten", "filename": "gs2.png"},
    {"group": "manual_handwritten", "filename": "gs3.png"},
    {"group": "manual_handwritten", "filename": "gs4.png"},
    {"group": "manual_handwritten", "filename": "gs5.png"},
    {"group": "manual_handwritten", "filename": "gs6.png"},
    {"group": "manual_handwritten", "filename": "gs7.png"},
    {"group": "manual_handwritten", "filename": "gs8.png"},
    {"group": "manual_handwritten", "filename": "gs9.png"},
    {"group": "manual_handwritten", "filename": "gs10.png"},
    {"group": "manual_handwritten", "filename": "gs11.png"},
    {"group": "manual_handwritten", "filename": "gs12.png"},
    {"group": "manual_handwritten", "filename": "gs13.png"},
    {"group": "manual_handwritten", "filename": "gs14.png"},
    {"group": "manual_handwritten", "filename": "gs15.png"},
    {"group": "manual_handwritten", "filename": "gs16.png"},
    {"group": "manual_handwritten", "filename": "gs17.png"},
    {"group": "manual_handwritten", "filename": "gs18.png"},
    {"group": "manual_handwritten", "filename": "gs19.png"},
    {"group": "manual_digital", "filename": "gs0.png"},
    {"group": "manual_digital", "filename": "gs1.png"},
    {"group": "manual_digital", "filename": "gs2.png"},
    {"group": "manual_digital", "filename": "gs3.png"},
    {"group": "manual_digital", "filename": "gs4.png"},
    {"group": "manual_digital", "filename": "gs5.png"},
    {"group": "manual_digital", "filename": "gs6.png"},
    {"group": "manual_digital", "filename": "gs7.png"},
    {"group": "manual_digital", "filename": "gs8.png"},
    {"group": "manual_digital", "filename": "gs9.png"},
    {"group": "manual_digital", "filename": "gs10.png"},
    {"group": "manual_digital", "filename": "gs11.png"},
    {"group": "manual_digital", "filename": "gs12.png"},
    {"group": "manual_digital", "filename": "gs13.png"},
    {"group": "manual_digital", "filename": "gs14.png"},
    {"group": "manual_digital", "filename": "gs15.png"},
    {"group": "manual_digital", "filename": "gs16.png"},
    {"group": "manual_digital", "filename": "gs17.png"},
    {"group": "manual_digital", "filename": "gs18.png"},
    {"group": "manual_digital", "filename": "gs19.png"},
]


def normalize_semantic(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def incipit_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_distorted"):
        stem = stem[: -len("_distorted")]
    return stem


def exclusion_id_for_entry(group: str, filename: str) -> str:
    if group.startswith("manual_") or group.endswith("_corrupt"):
        return Path(filename).stem
    return incipit_id_from_filename(filename)


def golden_image_path(golden_dir: Path, group: str, filename: str) -> Path:
    return golden_dir.joinpath(*GROUP_IMAGE_DIRS[group]) / filename


def golden_gt_path(golden_dir: Path, group: str, filename: str) -> Path:
    stem = Path(filename).stem
    if group.startswith("manual_"):
        return golden_dir / "manual" / "ground_truth" / f"{stem}.semantic"
    if group.endswith("_corrupt"):
        return golden_dir / "corrupt" / "ground_truth" / f"{stem}.semantic"
    return golden_dir / "original" / "ground_truth" / f"{incipit_id_from_filename(filename)}.semantic"


def load_golden_selection(golden_dir: Path) -> list[dict]:
    """Prefer Drive benchmark JSONs; fall back to inline GOLDEN_BENCHMARK_ENTRIES."""
    bench_dir = golden_dir / "benchmark"
    sel_files = [
        "selection_10.json",
        "selection_10_corrupt.json",
        "selection_manual.json",
    ]
    if bench_dir.is_dir() and all((bench_dir / name).is_file() for name in sel_files):
        selection: list[dict] = []
        for name in sel_files:
            selection.extend(json.loads((bench_dir / name).read_text(encoding="utf-8")))
        print(f"Loaded golden benchmark from {bench_dir} ({len(selection)} samples)")
        return selection

    print(f"Using inline golden benchmark list ({len(GOLDEN_BENCHMARK_ENTRIES)} samples)")
    return list(GOLDEN_BENCHMARK_ENTRIES)


def build_golden_eval_records(golden_dir: Path, selection: list[dict]) -> tuple[list[dict], set[str]]:
    records: list[dict] = []
    exclusion_ids: set[str] = set()
    missing = 0

    for entry in selection:
        group = entry["group"]
        filename = entry["filename"]
        exclusion_ids.add(exclusion_id_for_entry(group, filename))

        img_path = golden_image_path(golden_dir, group, filename)
        gt_path = golden_gt_path(golden_dir, group, filename)
        if not img_path.is_file() or not gt_path.is_file():
            missing += 1
            continue

        text = normalize_semantic(gt_path.read_text(encoding="utf-8"))
        records.append(
            {
                "group": group,
                "filename": filename,
                "stem": img_path.stem,
                "image": str(img_path),
                "gt": str(gt_path),
                "text": text,
                "exclusion_id": exclusion_id_for_entry(group, filename),
            }
        )

    if missing:
        print(
            f"Warning: {missing} golden benchmark files missing under {golden_dir}. "
            "Upload the full golden_set/ folder for eval."
        )
    return records, exclusion_ids


STYLE_GROUP_PREFIXES = {
    "handwritten": frozenset({"mashcima", "corrupt_handwritten"}),
    "clean": frozenset({"primus", "corrupt_clean", "corrupt_beamed_clean"}),
    "camera": frozenset({"camera_primus", "corrupt_camera", "corrupt_beamed_camera"}),
}


def record_style(group_prefix: str) -> str:
    for style, prefixes in STYLE_GROUP_PREFIXES.items():
        if group_prefix in prefixes:
            return style
    if "handwritten" in group_prefix:
        return "handwritten"
    if "camera" in group_prefix:
        return "camera"
    if group_prefix.endswith("_clean") or group_prefix == "primus":
        return "clean"
    raise ValueError(f"Unknown style for group_prefix={group_prefix!r}")


def record_is_corrupt(group_prefix: str) -> bool:
    return group_prefix.startswith("corrupt")


def style_corrupt_quotas(
    max_pairs: int,
    style_weights: dict[str, float],
    corrupt_fraction: float,
) -> dict[tuple[str, bool], int]:
    """Target counts per (style, is_corrupt) for stratified subsampling."""
    quotas: dict[tuple[str, bool], int] = {}
    assigned = 0
    active = [(s, w) for s, w in style_weights.items() if w > 0]
    for i, (style, weight) in enumerate(active):
        if i == len(active) - 1:
            style_total = max(0, max_pairs - assigned)
        else:
            style_total = int(round(max_pairs * weight))
            assigned += style_total
        corrupt_n = int(round(style_total * corrupt_fraction))
        orig_n = style_total - corrupt_n
        quotas[(style, False)] = orig_n
        quotas[(style, True)] = corrupt_n
    return quotas


def build_style_sample_weights(
    records: list[dict],
    style_weights: dict[str, float],
    corrupt_fraction: float = 0.5,
):
    import torch
    from collections import defaultdict

    buckets: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        gp = rec.get("group_prefix") or rec["stem"].split("__", 1)[0]
        buckets[(record_style(gp), record_is_corrupt(gp))].append(i)

    index_weights = [0.0] * len(records)
    for style, target_share in style_weights.items():
        if target_share <= 0:
            continue
        branches = []
        for corrupt in (False, True):
            idxs = buckets.get((style, corrupt), [])
            if idxs:
                branches.append(idxs)
        if not branches:
            continue
        per_branch = target_share / len(branches)
        for idxs in branches:
            w = per_branch / len(idxs)
            for i in idxs:
                index_weights[i] = w

    weights = torch.tensor(index_weights, dtype=torch.double)
    if weights.sum() <= 0:
        return torch.ones(len(records), dtype=torch.double)
    return weights / weights.sum() * len(records)


def print_style_mix_stats(records: list[dict], weights=None) -> None:
    from collections import Counter

    def bucket(rec):
        gp = rec.get("group_prefix") or rec["stem"].split("__", 1)[0]
        return (record_style(gp), record_is_corrupt(gp))

    raw = Counter(bucket(r) for r in records)
    print("Train pool by (style, corrupt):")
    for key in sorted(raw.keys()):
        print(f"  {key[0]:12s} corrupt={key[1]}: {raw[key]:,}")

    if weights is not None:
        weighted = Counter()
        for rec, wt in zip(records, weights.tolist()):
            weighted[bucket(rec)] += wt
        total = sum(weighted.values())
        print("Effective sampler mix (weights normalized):")
        for key in sorted(weighted.keys()):
            pct = 100.0 * weighted[key] / total if total else 0
            print(f"  {key[0]:12s} corrupt={key[1]}: {pct:.1f}%")


def build_pairs(images_dir: Path, gt_dir: Path) -> list[dict]:
    """Pair consolidated images with same-stem .semantic files."""
    if not images_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(
            f"Missing data dirs. Expected:\n  {images_dir}\n  {gt_dir}\n"
            "Upload data/consolidated/ to Drive first (see README_colab.md)."
        )

    pairs: list[dict] = []
    missing_gt = 0
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        gt_path = gt_dir / f"{img_path.stem}.semantic"
        if not gt_path.is_file():
            missing_gt += 1
            continue
        text = normalize_semantic(gt_path.read_text(encoding="utf-8"))
        if not text:
            continue
        match_stem = img_path.stem.split("__", 1)[-1] if "__" in img_path.stem else img_path.stem
        pairs.append(
            {
                "stem": img_path.stem,
                "match_stem": match_stem,
                "image": str(img_path),
                "gt": str(gt_path),
                "text": text,
                "group_prefix": img_path.stem.split("__", 1)[0],
            }
        )

    print(f"Built {len(pairs):,} pairs ({missing_gt:,} images missing GT)")
    return pairs


def pair_is_leaked(pair: dict, exclusion_ids: set[str]) -> bool:
    if pair["stem"].startswith("golden_"):
        return True
    return pair.get("match_stem", pair["stem"]) in exclusion_ids


def split_pairs(pairs: list[dict], exclusion_ids: set[str]) -> dict[str, list[dict]]:
    leaked = [p for p in pairs if pair_is_leaked(p, exclusion_ids)]
    train = [p for p in pairs if not pair_is_leaked(p, exclusion_ids)]
    return {"train": train, "leaked_from_consolidated": leaked}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")


def load_or_build_manifest() -> dict:
    if MANIFEST_PATH.is_file():
        cached = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if cached.get("version") == MANIFEST_VERSION:
            print(f"Loaded cached manifest from {MANIFEST_PATH} (version {MANIFEST_VERSION})")
            print(
                f"train={len(cached['splits']['train']):,}  "
                f"eval_golden={len(cached['splits']['eval_golden']):,}"
            )
            return cached
        print(f"Cached manifest version {cached.get('version')!r} != {MANIFEST_VERSION} -- rebuilding")

    selection = load_golden_selection(GOLDEN_DIR)
    eval_golden, exclusion_ids = build_golden_eval_records(GOLDEN_DIR, selection)
    pairs = build_pairs(IMAGES_DIR, GT_DIR)
    splits = split_pairs(pairs, exclusion_ids)

    manifest = {
        "version": MANIFEST_VERSION,
        "images_dir": str(IMAGES_DIR),
        "gt_dir": str(GT_DIR),
        "golden_dir": str(GOLDEN_DIR),
        "train_pairs": len(splits["train"]),
        "eval_golden": len(eval_golden),
        "splits": {"train": splits["train"], "eval_golden": eval_golden},
    }
    save_manifest(manifest)
    print(
        f"train={len(splits['train']):,}  eval_golden={len(eval_golden):,}  "
        f"leaked_excluded={len(splits['leaked_from_consolidated']):,}"
    )
    return manifest


def normalize_tokens(text: str) -> list[str]:
    return [p for p in re.split(r"[\s\t]+", text.strip()) if p]


# Relative major/minor share the same accidentals; canonicalize to the major form.
_RELATIVE_MINOR_TO_MAJOR = {
    "Am": "CM",
    "Em": "GM",
    "Bm": "DM",
    "F#m": "AM",
    "C#m": "EM",
    "G#m": "BM",
    "D#m": "F#M",
    "A#m": "C#M",
    "Dm": "FM",
    "Gm": "BbM",
    "Cm": "EbM",
    "Fm": "AbM",
    "Bbm": "DbM",
    "Ebm": "GbM",
    "Abm": "CbM",
}

# Common time ≡ 4/4; cut time ≡ 2/2.
_TIME_SIGNATURE_CANON = {
    "timeSignature-4/4": "timeSignature-C",
    "timeSignature-C": "timeSignature-C",
    "timeSignature-2/2": "timeSignature-C/",
    "timeSignature-C/": "timeSignature-C/",
}


def canonicalize_token(token: str) -> str:
    """Map notation-equivalent tokens to a shared canonical form for scoring."""
    if token.startswith("keySignature-"):
        key = token[len("keySignature-") :]
        key = _RELATIVE_MINOR_TO_MAJOR.get(key, key)
        return f"keySignature-{key}"
    return _TIME_SIGNATURE_CANON.get(token, token)


def canonicalize_tokens(tokens: list[str]) -> list[str]:
    return [canonicalize_token(t) for t in tokens]


def token_edit_distance(pred: list[str], gt: list[str]) -> int:
    """Token-level Levenshtein (same metric as scripts/omr_benchmark_score.py).

    Relative major/minor key signatures and common/cut-time aliases count as equal.
    """
    pred = canonicalize_tokens(pred)
    gt = canonicalize_tokens(gt)
    m, n = len(gt), len(pred)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        gt_tok = gt[i - 1]
        for j in range(1, n + 1):
            if gt_tok == pred[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def strip_trailing_barline(tokens: list[str]) -> list[str]:
    """Drop a single trailing 'barline' token if present (trailing-barline-agnostic eval)."""
    if tokens and tokens[-1] == "barline":
        return tokens[:-1]
    return tokens


def barline_forgiven_distance(pred: list[str], gt: list[str]) -> int:
    """Token edit distance that ignores a trailing-barline-only difference and never penalizes.

    The min over {plain, strip-pred, strip-gt, strip-both} is always <= the plain distance,
    so forgiving a trailing barline can only help or be neutral, while any difference that is
    solely a trailing barline (on either side) is neutralized. Only one trailing 'barline' per
    side is ever removed; internal barlines stay counted.
    """
    sp, sg = strip_trailing_barline(pred), strip_trailing_barline(gt)
    return min(
        token_edit_distance(pred, gt),
        token_edit_distance(sp, gt),
        token_edit_distance(pred, sg),
        token_edit_distance(sp, sg),
    )


def score_tokens(pred: list[str], gt: list[str]) -> dict:
    n_gt = len(gt)
    errors = barline_forgiven_distance(pred, gt)
    if n_gt == 0:
        return {
            "errors": errors,
            "gt_tokens": 0,
            "error_rate_pct": 100.0 if errors else 0.0,
            "accuracy_pct": 0.0 if errors else 100.0,
        }
    error_rate_pct = errors / n_gt * 100.0
    accuracy_pct = max(0.0, 100.0 - error_rate_pct)
    return {
        "errors": errors,
        "gt_tokens": n_gt,
        "error_rate_pct": round(error_rate_pct, 2),
        "accuracy_pct": round(accuracy_pct, 2),
    }


def group_sort_key(group: str) -> int:
    try:
        return GROUP_ORDER.index(group)
    except ValueError:
        return len(GROUP_ORDER)


def predict_semantic(
    model,
    processor,
    image_path: str,
    *,
    device: str,
    task_prompt: str,
    max_new_tokens: int = 512,
) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    cast_inputs = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            cast_inputs[k] = v
            continue
        v = v.to(device)
        if k == "pixel_values":
            v = v.to(dtype=model_dtype)
        cast_inputs[k] = v
    inputs = cast_inputs

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=1,  # beam search + constrained processor collapses to zero tokens; greedy is equivalent here
        do_sample=False,
        use_cache=False,
    )
    if USE_CONSTRAINED_DECODE:
        gen_kwargs["logits_processor"] = [make_primus_logits_processor(processor.tokenizer)]

    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(**gen_kwargs)

    # Fine-tuned PrIMuS semantics are plain token strings -- do NOT run the native
    # <OCR> post-processor (it strips spaces and parses bbox-tagged OCR regions).
    prompt_ids = inputs["input_ids"][0]
    gen_ids = generated_ids[0]
    if (
        gen_ids.shape[0] >= prompt_ids.shape[0]
        and torch.equal(gen_ids[: prompt_ids.shape[0]], prompt_ids)
    ):
        new_ids = gen_ids[prompt_ids.shape[0] :]
    else:
        new_ids = gen_ids
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    if text.startswith(task_prompt):
        text = text[len(task_prompt):]
    return normalize_semantic(text)


def summarize_golden_scores(sample_rows: list[dict]) -> dict:
    from collections import defaultdict

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        by_group[row["group"]].append(row)

    group_summary = {}
    for group in sorted(by_group, key=group_sort_key):
        rows = by_group[group]
        group_summary[group] = {
            "n": len(rows),
            "exact_match": sum(1 for r in rows if r["errors"] == 0),
            "mean_error_rate_pct": round(sum(r["error_rate_pct"] for r in rows) / len(rows), 2),
            "mean_accuracy_pct": round(sum(r["accuracy_pct"] for r in rows) / len(rows), 2),
        }

    return {
        "exact_match": sum(1 for r in sample_rows if r["errors"] == 0),
        "mean_error_rate_pct": round(
            sum(r["error_rate_pct"] for r in sample_rows) / max(len(sample_rows), 1), 2
        ),
        "mean_accuracy_pct": round(
            sum(r["accuracy_pct"] for r in sample_rows) / max(len(sample_rows), 1), 2
        ),
    }, group_summary


def run_golden_eval(
    model,
    processor,
    golden_records: list[dict],
    *,
    phase: str,
    model_id: str,
    task_prompt: str,
    device: str,
    use_lora: bool = False,
    show_progress: bool = True,
) -> dict:
    from tqdm.auto import tqdm

    if not golden_records:
        print(
            "No golden eval samples found. Upload golden_set/ to Drive "
            f"(expected under {GOLDEN_DIR})."
        )
        return {
            "phase": phase,
            "model_id": model_id,
            "task_prompt": task_prompt,
            "use_lora": use_lora,
            "tokenizer": "primus_closed_vocab",
            "barline_normalization": "trailing_barline_ignored",
            "vocab_size": processor.tokenizer.vocab_size,
            "n_samples": 0,
            "expected_samples": GOLDEN_BENCHMARK_COUNT,
            "overall": {},
            "by_group": {},
            "samples": [],
        }

    model.to(device)
    sample_rows = []
    iterator = tqdm(golden_records, desc=f"golden eval ({phase})") if show_progress else golden_records

    for rec in iterator:
        gt_tokens = normalize_tokens(rec["text"])
        pred_text = predict_semantic(
            model,
            processor,
            rec["image"],
            device=device,
            task_prompt=task_prompt,
        )
        pred_tokens = normalize_tokens(pred_text)
        scored = score_tokens(pred_tokens, gt_tokens)
        sample_rows.append(
            {
                "group": rec["group"],
                "filename": rec["filename"],
                "gt_tokens": scored["gt_tokens"],
                "pred_tokens": len(pred_tokens),
                "errors": scored["errors"],
                "error_rate_pct": scored["error_rate_pct"],
                "accuracy_pct": scored["accuracy_pct"],
                "prediction": pred_text,
            }
        )

    overall, by_group = summarize_golden_scores(sample_rows)
    return {
        "phase": phase,
        "model_id": model_id,
        "task_prompt": task_prompt,
        "use_lora": use_lora,
        "tokenizer": "primus_closed_vocab",
        "barline_normalization": "trailing_barline_ignored",
        "vocab_size": processor.tokenizer.vocab_size,
        "n_samples": len(sample_rows),
        "expected_samples": GOLDEN_BENCHMARK_COUNT,
        "overall": overall,
        "by_group": by_group,
        "samples": sample_rows,
    }


def print_golden_eval_summary(results: dict, title: str) -> None:
    if not results.get("samples"):
        print(f"{title}: no samples scored")
        return

    n = results["n_samples"]
    overall = results["overall"]
    print(f"{title}: {n}/{results['expected_samples']} samples")
    print(
        f"Exact match: {overall['exact_match']}/{n} "
        f"({overall['exact_match'] / n * 100:.1f}%)"
    )
    print(f"Mean token accuracy: {overall['mean_accuracy_pct']:.2f}%")
    for group in sorted(results["by_group"], key=group_sort_key):
        g = results["by_group"][group]
        print(
            f"  {group}: n={g['n']}  acc={g['mean_accuracy_pct']:.2f}%  "
            f"err={g['mean_error_rate_pct']:.2f}%"
        )


def compare_golden_eval_results(before: dict, after: dict) -> None:
    if not before.get("samples") or not after.get("samples"):
        print("Cannot compare -- missing before or after results")
        return

    b_overall = before["overall"]["mean_accuracy_pct"]
    a_overall = after["overall"]["mean_accuracy_pct"]
    print("\n=== Before vs after (golden 100) ===")
    print(f"Overall accuracy: {b_overall:.2f}% -> {a_overall:.2f}%  (delta {a_overall - b_overall:+.2f} pp)")

    groups = sorted(
        set(before["by_group"]) | set(after["by_group"]),
        key=group_sort_key,
    )
    for group in groups:
        b_acc = before["by_group"].get(group, {}).get("mean_accuracy_pct")
        a_acc = after["by_group"].get(group, {}).get("mean_accuracy_pct")
        if b_acc is None or a_acc is None:
            continue
        print(f"  {group}: {b_acc:.2f}% -> {a_acc:.2f}%  (delta {a_acc - b_acc:+.2f} pp)")



def golden_eval_epoch_path(epoch: int) -> Path:
    return EVAL_GOLDEN_EPOCH_DIR / f"eval_golden_epoch_{epoch:02d}.json"


def format_golden_eval_writeup_section(
    results: dict,
    *,
    epoch: int,
    baseline: dict | None = None,
    previous: dict | None = None,
) -> str:
    lines: list[str] = []
    overall = results.get("overall") or {}
    n = results.get("n_samples", 0)
    mean_acc = overall.get("mean_accuracy_pct")
    exact = overall.get("exact_match")
    lines.append(f"## Epoch {epoch}")
    lines.append("")
    if mean_acc is None:
        lines.append("No samples scored.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"- **Mean token accuracy:** {mean_acc:.2f}%  "
        f"({exact}/{n} exact match)"
    )
    if baseline and baseline.get("overall"):
        b = baseline["overall"]["mean_accuracy_pct"]
        lines.append(f"- **vs baseline:** {mean_acc - b:+.2f} pp")
    if previous and previous.get("overall"):
        p = previous["overall"]["mean_accuracy_pct"]
        lines.append(f"- **vs previous epoch:** {mean_acc - p:+.2f} pp")
    lines.append("")
    lines.append("| Group | n | Accuracy | Error rate |")
    lines.append("|---|---:|---:|---:|")
    for group in sorted(results.get("by_group") or {}, key=group_sort_key):
        g = results["by_group"][group]
        lines.append(
            f"| {group} | {g['n']} | {g['mean_accuracy_pct']:.2f}% | "
            f"{g['mean_error_rate_pct']:.2f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def save_golden_epoch_eval(
    results: dict,
    *,
    epoch: int,
    baseline: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Persist per-epoch golden JSON + rolling markdown writeup + history index."""
    EVAL_GOLDEN_EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_eval_epoch_path(epoch)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if history is None:
        if EVAL_GOLDEN_EPOCH_HISTORY_PATH.is_file():
            history = json.loads(
                EVAL_GOLDEN_EPOCH_HISTORY_PATH.read_text(encoding="utf-8")
            )
        else:
            history = []

    history = [h for h in history if h.get("epoch") != epoch]
    summary = {
        "epoch": epoch,
        "phase": results.get("phase"),
        "path": str(path),
        "n_samples": results.get("n_samples"),
        "mean_accuracy_pct": (results.get("overall") or {}).get("mean_accuracy_pct"),
        "exact_match": (results.get("overall") or {}).get("exact_match"),
        "by_group": {
            g: {"mean_accuracy_pct": v["mean_accuracy_pct"], "n": v["n"]}
            for g, v in (results.get("by_group") or {}).items()
        },
    }
    history.append(summary)
    history.sort(key=lambda h: h["epoch"])
    EVAL_GOLDEN_EPOCH_HISTORY_PATH.write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    sections = [
        "# Golden benchmark — per-epoch results",
        "",
        f"Output dir: `{DRIVE_OUTPUT_DIR}`",
        "",
    ]
    if baseline and baseline.get("overall"):
        b = baseline["overall"]
        sections.append(
            f"**Baseline (before training):** {b['mean_accuracy_pct']:.2f}% "
            f"mean token accuracy "
            f"({b.get('exact_match', '?')}/{baseline.get('n_samples', '?')} exact)."
        )
        sections.append("")

    previous = None
    for entry in history:
        ep = int(entry["epoch"])
        ep_path = Path(entry["path"]) if entry.get("path") else golden_eval_epoch_path(ep)
        if not ep_path.is_file():
            ep_path = golden_eval_epoch_path(ep)
        if ep_path.is_file():
            ep_results = json.loads(ep_path.read_text(encoding="utf-8"))
        else:
            ep_results = {
                "overall": {
                    "mean_accuracy_pct": entry.get("mean_accuracy_pct"),
                    "exact_match": entry.get("exact_match"),
                },
                "n_samples": entry.get("n_samples"),
                "by_group": {},
            }
        sections.append(
            format_golden_eval_writeup_section(
                ep_results,
                epoch=ep,
                baseline=baseline,
                previous=previous,
            )
        )
        previous = ep_results

    EVAL_GOLDEN_EPOCH_WRITEUP_PATH.write_text(
        "\n".join(sections).rstrip() + "\n", encoding="utf-8"
    )
    print(f"Wrote {path}")
    print(f"Wrote {EVAL_GOLDEN_EPOCH_HISTORY_PATH}")
    print(f"Wrote {EVAL_GOLDEN_EPOCH_WRITEUP_PATH}")
    return summary


def patch_florence2_transformers5_compat(*, model_id: str | None = None) -> None:
    """Florence-2 shims for transformers 5.x (from Colab load cell)."""
    import transformers.configuration_utils as config_utils
    import transformers.tokenization_utils_base as tok_base

    _flag = "_florence2_tf5_compat_patched"

    if not getattr(config_utils.PretrainedConfig, _flag, False):
        _orig_init = config_utils.PretrainedConfig.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, "forced_bos_token_id"):
                self.forced_bos_token_id = kwargs.get("forced_bos_token_id", None)

        config_utils.PretrainedConfig.__init__ = _patched_init
        setattr(config_utils.PretrainedConfig, _flag, True)

    ga_flag = "_florence2_tf5_getattr_patched"
    if not getattr(config_utils.PretrainedConfig, ga_flag, False):
        _orig_ga = config_utils.PretrainedConfig.__getattribute__

        def _patched_ga(self, key):
            if key == "forced_bos_token_id":
                try:
                    return _orig_ga(self, key)
                except AttributeError:
                    return None
            return _orig_ga(self, key)

        config_utils.PretrainedConfig.__getattribute__ = _patched_ga
        setattr(config_utils.PretrainedConfig, ga_flag, True)

    tok_flag = "_florence2_tf5_tok_getattr_patched"
    if not getattr(tok_base.PreTrainedTokenizerBase, tok_flag, False):
        if hasattr(tok_base.PreTrainedTokenizerBase, "__getattr__"):
            _orig_getattr = tok_base.PreTrainedTokenizerBase.__getattr__

            def _patched_getattr(self, name):
                if name == "additional_special_tokens":
                    try:
                        return _orig_getattr(self, name)
                    except AttributeError:
                        try:
                            added = _orig_getattr(self, "added_tokens_decoder")
                        except AttributeError:
                            return []
                        return [
                            tok.content
                            for tok in added.values()
                            if getattr(tok, "special", False)
                        ]
                return _orig_getattr(self, name)

            tok_base.PreTrainedTokenizerBase.__getattr__ = _patched_getattr
            setattr(tok_base.PreTrainedTokenizerBase, tok_flag, True)

    if model_id:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                model_id,
                allow_patterns=[
                    "configuration_florence2.py",
                    "processing_florence2.py",
                    "modeling_florence2.py",
                ],
            )
        except Exception:
            pass

    search_roots = [
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    patches = [
        (
            "configuration_florence2.py",
            "if self.forced_bos_token_id is None and kwargs.get(",
            "if getattr(self, 'forced_bos_token_id', None) is None and kwargs.get(",
        ),
        (
            "processing_florence2.py",
            "tokenizer.additional_special_tokens + \\",
            "getattr(tokenizer, 'additional_special_tokens', []) + \\",
        ),
        (
            "modeling_florence2.py",
            "        return self.language_model._supports_sdpa",
            "        if not hasattr(self, \"language_model\"):\n"
            "            return False\n"
            "        return self.language_model._supports_sdpa",
        ),
        (
            "modeling_florence2.py",
            "        return self.language_model._supports_flash_attn_2",
            "        if not hasattr(self, \"language_model\"):\n"
            "            return False\n"
            "        return self.language_model._supports_flash_attn_2",
        ),
    ]

    patched_any = False
    for root in search_roots:
        if not root.is_dir():
            continue
        for src in root.rglob("*.py"):
            for filename, old, new in patches:
                if src.name != filename:
                    continue
                try:
                    text = src.read_text(encoding="utf-8")
                except OSError:
                    continue
                if old in text and new not in text:
                    src.write_text(text.replace(old, new), encoding="utf-8")
                    patched_any = True

    modules_cache = search_roots[0]
    if patched_any and modules_cache.is_dir():
        for d in modules_cache.glob("*lorence*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


# Dataset / collator

_CORRUPT_WARNED: set[str] = set()



class OMRDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_rgb(path: str) -> "Image.Image":
        image = Image.open(path).convert("RGB")
        image.load()  # force full decode now so any error surfaces here, not mid-batch
        return image

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        try:
            image = self._load_rgb(rec["image"])
        except (OSError, SyntaxError, ValueError) as exc:
            # Undecodable image: warn once, then substitute the next valid sample so the
            # batch stays intact and training does not crash. A few bad files out of 136k
            # are harmless; run the corruption scan to find/remove them permanently.
            if rec["image"] not in _CORRUPT_WARNED:
                _CORRUPT_WARNED.add(rec["image"])
                print(f"[OMRDataset] skipping unreadable image: {rec['image']} ({exc})", flush=True)
            n = len(self.records)
            for off in range(1, n):
                alt = self.records[(idx + off) % n]
                try:
                    image = self._load_rgb(alt["image"])
                    rec = alt
                    break
                except (OSError, SyntaxError, ValueError):
                    continue
            else:
                raise RuntimeError("No decodable images in dataset") from exc
        return {"image": image, "text": rec["text"], "stem": rec["stem"]}



def make_collate_fn(processor, task_prompt: str, max_label_length: int):
    """Florence-2 is an ENCODER-DECODER model.

    The encoder consumes (task prompt + image); the decoder target is the answer.
    Do NOT concatenate prompt+answer into input_ids (that is the causal-LM recipe
    and makes the model echo an answer it can already see in the encoder input --
    at inference the encoder only sees the prompt, so it emits EOS => empty output).

    Pass the answer as `labels`; the model builds decoder_input_ids internally via
    shift_tokens_right(labels, ...). See modeling_florence2.py forward().
    """

    def collate(batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]

        # Encoder inputs: task prompt + image.
        enc = processor(
            text=[task_prompt] * len(batch),
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Decoder targets: one token ID per PrIMuS semantic type.
        labels = encode_semantic_labels(
            processor.tokenizer,
            texts,
            max_length=max_label_length,
        )

        # Ignore pad positions in the loss.
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        enc["labels"] = labels
        return enc

    return collate


# Trainer helpers

class GoldenEpochEvalCallback(TrainerCallback):
    """Run the 100-image golden benchmark after each full training epoch."""

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not EVAL_GOLDEN_EVERY_EPOCH:
            return
        if model is None:
            print("Golden epoch eval skipped — no model in callback kwargs")
            return

        epoch_num = int(round(float(state.epoch)))
        if epoch_num < 1:
            return

        print(f"\n=== Golden eval after epoch {epoch_num} ===")
        was_training = model.training
        try:
            results = run_golden_eval(
                model,
                processor,
                golden_records,
                phase=f"after_epoch_{epoch_num}",
                model_id=MODEL_ID,
                task_prompt=TASK_PROMPT,
                device=DEVICE,
                use_lora=True,
            )
            print_golden_eval_summary(results, f"After epoch {epoch_num}")

            baseline = globals().get("baseline_results")
            if baseline is None and EVAL_GOLDEN_BEFORE_PATH.is_file():
                baseline = json.loads(
                    EVAL_GOLDEN_BEFORE_PATH.read_text(encoding="utf-8")
                )
            if baseline is not None:
                compare_golden_eval_results(baseline, results)

            save_golden_epoch_eval(results, epoch=epoch_num, baseline=baseline)
        finally:
            if was_training:
                model.train()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()



class StyleBalancedTrainer(Trainer):
    """WeightedRandomSampler so each epoch sees ~50/25/25 handwritten/clean/camera."""

    def __init__(self, *args, train_sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sample_weights = train_sample_weights

    def _get_train_sampler(self, train_dataset=None):
        # Newer transformers pass the dataset; older ones call with no args.
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_sample_weights is not None and dataset is self.train_dataset:
            from torch.utils.data import WeightedRandomSampler

            return WeightedRandomSampler(
                weights=self._train_sample_weights,
                num_samples=len(dataset),
                replacement=True,
            )
        try:
            return super()._get_train_sampler(train_dataset)
        except TypeError:
            return super()._get_train_sampler()

