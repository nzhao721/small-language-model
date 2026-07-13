"""Re-score a Florence golden eval JSON with key/time signature canonicalization."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from omr_benchmark_score import (  # noqa: E402
    group_sort_key,
    load_gt,
    normalize_tokens,
    token_edit_distance,
)


def strip_trailing_barline(tokens: list[str]) -> list[str]:
    if tokens and tokens[-1] == "barline":
        return tokens[:-1]
    return tokens


def barline_forgiven_distance(pred: list[str], gt: list[str]) -> int:
    sp, sg = strip_trailing_barline(pred), strip_trailing_barline(gt)
    return min(
        token_edit_distance(pred, gt),
        token_edit_distance(sp, gt),
        token_edit_distance(pred, sg),
        token_edit_distance(sp, sg),
    )


def score_sample(pred: list[str], gt: list[str]) -> dict:
    n_gt = len(gt)
    errors = barline_forgiven_distance(pred, gt)
    if n_gt == 0:
        return {
            "errors": errors,
            "gt_tokens": 0,
            "error_rate_pct": 100.0 if errors else 0.0,
            "accuracy_pct": 0.0 if errors else 100.0,
            "exact_match": errors == 0,
        }
    error_rate_pct = errors / n_gt * 100.0
    accuracy_pct = max(0.0, 100.0 - error_rate_pct)
    return {
        "errors": errors,
        "gt_tokens": n_gt,
        "error_rate_pct": round(error_rate_pct, 2),
        "accuracy_pct": round(accuracy_pct, 2),
        "exact_match": errors == 0,
    }

GOLDEN = ROOT / "golden_set"


def incipit_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_distorted"):
        stem = stem[: -len("_distorted")]
    return stem


def ground_truth_path(group: str, filename: str) -> Path:
    stem = Path(filename).stem
    if group.startswith("manual_"):
        return GOLDEN / "manual" / "ground_truth" / f"{stem}.semantic"
    if group.endswith("_corrupt"):
        return GOLDEN / "corrupt" / "ground_truth" / f"{stem}.semantic"
    return GOLDEN / "original" / "ground_truth" / f"{incipit_id_from_filename(filename)}.semantic"


def rescore_eval(eval_path: Path) -> dict:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    by_group: dict[str, list[dict]] = defaultdict(list)

    for sample in data["samples"]:
        gt_path = ground_truth_path(sample["group"], sample["filename"])
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing ground truth: {gt_path}")

        gt_tokens = load_gt(gt_path)
        pred_tokens = normalize_tokens(sample["prediction"])
        scored = score_sample(pred_tokens, gt_tokens)

        sample["gt_tokens"] = len(gt_tokens)
        sample["pred_tokens"] = len(pred_tokens)
        sample["errors"] = scored["errors"]
        sample["error_rate_pct"] = scored["error_rate_pct"]
        sample["accuracy_pct"] = scored["accuracy_pct"]
        sample["exact_match"] = scored["exact_match"]
        by_group[sample["group"]].append(sample)

    samples = data["samples"]
    data["overall"] = {
        "exact_match": sum(1 for s in samples if s.get("exact_match")),
        "mean_error_rate_pct": round(
            sum(s["error_rate_pct"] for s in samples) / len(samples), 2
        ),
        "mean_accuracy_pct": round(
            sum(s["accuracy_pct"] for s in samples) / len(samples), 2
        ),
    }
    data["metric_policy"] = {
        "relative_major_minor_equivalent": True,
        "time_signature_equivalence": ["4/4 <-> C", "2/2 <-> C/"],
        "trailing_barline": "at most one trailing barline forgiven on either side",
    }
    data["by_group"] = {}
    for group in sorted(by_group.keys(), key=group_sort_key):
        rows = by_group[group]
        data["by_group"][group] = {
            "n": len(rows),
            "exact_match": sum(1 for r in rows if r.get("exact_match")),
            "mean_error_rate_pct": round(
                sum(r["error_rate_pct"] for r in rows) / len(rows), 2
            ),
            "mean_accuracy_pct": round(
                sum(r["accuracy_pct"] for r in rows) / len(rows), 2
            ),
        }

    eval_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return data


def main() -> None:
    eval_path = ROOT / "florence2-base_eval.json"
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1])
    data = rescore_eval(eval_path)
    print(f"Updated {eval_path}")
    print("overall:", data["overall"])


if __name__ == "__main__":
    main()
