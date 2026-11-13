"""Golden-set evaluation helpers for Qwen2-VL OMR training."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXPECTED_SYNTHETIC = 60
EXPECTED_MANUAL_PER_STYLE = 20
EXPECTED_GOLDEN_TOTAL = 100

SYSTEM_PROMPT = "You are an optical music recognition system. Return only PrIMuS semantic tokens."
USER_PROMPT = (
    "Transcribe this monophonic sheet-music image into PrIMuS semantic format. "
    "Return only one space-separated token string with no explanation or markdown."
)

GROUP_DIRS = {
    "handwritten_original": ("original", "images", "handwritten"),
    "clean_original": ("original", "images", "clean"),
    "camera_distorted_original": ("original", "images", "camera"),
    "handwritten_corrupt": ("corrupt", "images", "handwritten"),
    "clean_corrupt": ("corrupt", "images", "clean"),
    "camera_distorted_corrupt": ("corrupt", "images", "camera"),
}

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
_TIME_SIGNATURE_CANON = {
    "timeSignature-4/4": "timeSignature-C",
    "timeSignature-C": "timeSignature-C",
    "timeSignature-2/2": "timeSignature-C/",
    "timeSignature-C/": "timeSignature-C/",
}


def normalized_target(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def find_image(folder: Path, filename_or_stem: str) -> Path | None:
    filename = filename_or_stem.replace("\\", "/").rsplit("/", 1)[-1]
    exact = folder / filename
    if exact.is_file() and exact.suffix.lower() in IMAGE_EXTS:
        return exact
    stem = Path(filename).stem
    return next(
        (
            path
            for path in sorted(folder.glob(f"{stem}.*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        None,
    )


def load_all_golden_records(golden_root: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    missing_details: list[dict] = []
    manifest_path = golden_root / "manifest.jsonl"
    manifest_rows = 0

    def add_record(record: dict) -> None:
        key = (str(record["group"]), str(record["id"]))
        if key in seen:
            return
        seen.add(key)
        records.append(record)

    if manifest_path.is_file():
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            manifest_rows += 1
            try:
                row = json.loads(line)
                record_id = str(row["id"])
                group = str(row["group"])
                source = str(row.get("source", "original"))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                missing_details.append(
                    {"kind": "synthetic", "line": line_number, "reason": f"invalid manifest row: {exc}"}
                )
                continue
            group_parts = GROUP_DIRS.get(group)
            if group_parts is None:
                missing_details.append(
                    {"kind": "synthetic", "id": record_id, "group": group, "reason": "unknown group"}
                )
                continue
            image_path = find_image(
                golden_root.joinpath(*group_parts),
                str(row.get("image_path", record_id)),
            )
            gt_path = golden_root / source / "ground_truth" / f"{record_id}.semantic"
            if image_path is None or not gt_path.is_file():
                missing_details.append(
                    {
                        "kind": "synthetic",
                        "id": record_id,
                        "group": group,
                        "missing_image": image_path is None,
                        "missing_ground_truth": not gt_path.is_file(),
                    }
                )
                continue
            target = normalized_target(gt_path)
            if not target:
                continue
            add_record(
                {
                    "id": record_id,
                    "group": group,
                    "kind": "synthetic",
                    "image": str(image_path),
                    "target": target,
                }
            )
    else:
        missing_details.append(
            {"kind": "synthetic", "reason": f"manifest missing: {manifest_path}", "expected": EXPECTED_SYNTHETIC}
        )

    manual_gt_dir = golden_root / "manual" / "ground_truth"
    for style in ("handwritten", "digital"):
        group = f"manual_{style}"
        image_dir = golden_root / "manual" / style
        for index in range(EXPECTED_MANUAL_PER_STYLE):
            record_id = f"gs{index}"
            image_path = find_image(image_dir, record_id)
            gt_path = manual_gt_dir / f"{record_id}.semantic"
            if image_path is None or not gt_path.is_file():
                missing_details.append(
                    {
                        "kind": "manual",
                        "id": record_id,
                        "group": group,
                        "missing_image": image_path is None,
                        "missing_ground_truth": not gt_path.is_file(),
                    }
                )
                continue
            target = normalized_target(gt_path)
            if not target:
                continue
            add_record(
                {
                    "id": record_id,
                    "group": group,
                    "kind": "manual",
                    "image": str(image_path),
                    "target": target,
                }
            )

    coverage = {
        "expected_total": EXPECTED_GOLDEN_TOTAL,
        "expected_synthetic": EXPECTED_SYNTHETIC,
        "expected_manual": 2 * EXPECTED_MANUAL_PER_STYLE,
        "synthetic_manifest_rows": manifest_rows,
        "available_total": len(records),
        "missing_total": max(0, EXPECTED_GOLDEN_TOTAL - len(records)),
        "available_by_group": dict(sorted(Counter(record["group"] for record in records).items())),
        "missing_details": missing_details,
    }
    return records, coverage


def canonicalize_token(token: str) -> str:
    if token.startswith("keySignature-"):
        key = token[len("keySignature-") :]
        return f"keySignature-{_RELATIVE_MINOR_TO_MAJOR.get(key, key)}"
    return _TIME_SIGNATURE_CANON.get(token, token)


def canonicalize_tokens(tokens: list[str]) -> list[str]:
    return [canonicalize_token(token) for token in tokens]


def token_edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def strip_one_trailing_barline(tokens: list[str]) -> list[str]:
    return tokens[:-1] if tokens and tokens[-1] == "barline" else tokens


def project_token_distance(prediction_tokens: list[str], target_tokens: list[str]) -> int:
    pred = canonicalize_tokens(prediction_tokens)
    target = canonicalize_tokens(target_tokens)
    stripped_pred = strip_one_trailing_barline(pred)
    stripped_target = strip_one_trailing_barline(target)
    return min(
        token_edit_distance(pred, target),
        token_edit_distance(stripped_pred, target),
        token_edit_distance(pred, stripped_target),
        token_edit_distance(stripped_pred, stripped_target),
    )


def score_prediction(prediction: str, target: str) -> dict:
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    edits = project_token_distance(prediction_tokens, target_tokens)
    gt_tokens = len(target_tokens)
    if gt_tokens:
        error_rate_pct = edits / gt_tokens * 100.0
        accuracy_pct = max(0.0, 100.0 - error_rate_pct)
    else:
        error_rate_pct = 100.0 if edits else 0.0
        accuracy_pct = 0.0 if edits else 100.0
    return {
        "token_edits": edits,
        "gt_tokens": gt_tokens,
        "prediction_tokens": len(prediction_tokens),
        "error_rate_pct": error_rate_pct,
        "accuracy_pct": accuracy_pct,
        "exact_match": edits == 0,
    }


def clean_prediction(text: str) -> str:
    text = re.sub(r"```(?:\w+)?", "", text or "").replace("```", "")
    return " ".join(text.split())


def make_messages() -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]


def aggregate_rows(rows: list[dict]) -> tuple[dict, dict]:
    successful = [row for row in rows if row.get("status") == "ok"]
    total_edits = sum(int(row["token_edits"]) for row in successful)
    total_gt_tokens = sum(int(row["gt_tokens"]) for row in successful)
    mean_accuracy = (
        sum(float(row["accuracy_pct"]) for row in successful) / len(successful) if successful else None
    )
    exact_matches = sum(bool(row["exact_match"]) for row in successful)
    micro_accuracy = (
        max(0.0, 100.0 * (1.0 - total_edits / total_gt_tokens)) if total_gt_tokens else 0.0
    )
    overall = {
        "n_available": len(rows),
        "n_evaluated": len(successful),
        "n_inference_errors": len(rows) - len(successful),
        "mean_accuracy_pct": round(mean_accuracy, 4) if mean_accuracy is not None else None,
        "exact_matches": exact_matches,
        "exact_match_pct": round(100.0 * exact_matches / len(successful), 4) if successful else 0.0,
        "total_token_edits": total_edits,
        "total_gt_tokens": total_gt_tokens,
        "micro_token_accuracy_pct": round(micro_accuracy, 4),
    }
    grouped: dict[str, dict] = {}
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        group_ok = [row for row in group_rows if row.get("status") == "ok"]
        group_edits = sum(int(row["token_edits"]) for row in group_ok)
        group_gt = sum(int(row["gt_tokens"]) for row in group_ok)
        group_exact = sum(bool(row["exact_match"]) for row in group_ok)
        group_mean = (
            sum(float(row["accuracy_pct"]) for row in group_ok) / len(group_ok) if group_ok else None
        )
        group_micro = max(0.0, 100.0 * (1.0 - group_edits / group_gt)) if group_gt else 0.0
        grouped[group] = {
            "n_available": len(group_rows),
            "n_evaluated": len(group_ok),
            "n_inference_errors": len(group_rows) - len(group_ok),
            "mean_accuracy_pct": round(group_mean, 4) if group_mean is not None else None,
            "exact_matches": group_exact,
            "exact_match_pct": round(100.0 * group_exact / len(group_ok), 4) if group_ok else 0.0,
            "total_token_edits": group_edits,
            "total_gt_tokens": group_gt,
            "micro_token_accuracy_pct": round(group_micro, 4),
        }
    return overall, grouped


def run_golden_eval(model, processor, golden_root: Path, step: int, model_id: str) -> dict:
    import gc

    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    records, coverage = load_all_golden_records(golden_root)
    if not records:
        raise RuntimeError(f"No golden records found under {golden_root}")

    was_training = model.training
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    rows: list[dict] = []
    for record in tqdm(records, desc=f"Golden eval step {step}"):
        row = {
            "id": record["id"],
            "group": record["group"],
            "kind": record["kind"],
            "image": record["image"],
            "target": record["target"],
        }
        try:
            with Image.open(record["image"]) as source:
                image = source.convert("RGB")
                image.load()
            prompt_text = processor.apply_chat_template(
                make_messages(),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt_text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            model_device = getattr(model, "device", None) or next(model.parameters()).device
            inputs = inputs.to(model_device)
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            generated = output_ids[:, prompt_length:]
            prediction = clean_prediction(
                processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            )
            row.update({"status": "ok", "prediction": prediction, **score_prediction(prediction, record["target"])})
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "prediction": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)

    metrics, per_group = aggregate_rows(rows)
    if was_training:
        model.train()
        if hasattr(model, "config"):
            model.config.use_cache = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "schema_version": 1,
        "checkpoint_step": step,
        "model_id": model_id,
        "golden_coverage": coverage,
        "metrics": metrics,
        "per_group": per_group,
        "samples": rows,
    }
