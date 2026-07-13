"""Score OMR benchmark predictions against ground truth."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "examples" / "omr_benchmark"

MODELS = [
    "claude-opus-4-8-thinking-high",
    "gemini-3.1-pro",
    "composer-2.5",
    "grok-4.5-xhigh",
    "gpt-5.6-sol-medium",
]

VLM_SHORTLIST_MODELS = [
    "florence-2-base",
    "donut-base",
    "florence-2-large",
    "qwen2-vl-2b",
    "qwen2-vl-7b",
]

MODEL_SHORT = {
    "claude-opus-4-8-thinking-high": "opus",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "composer-2.5": "composer-2.5",
    "grok-4.5-xhigh": "grok-4.5-high",
    "gpt-5.6-sol-medium": "gpt-5.6-sol",
    "florence-2-base": "florence-2-base",
    "donut-base": "donut-base",
    "florence-2-large": "florence-2-large",
    "qwen2-vl-2b": "qwen2-vl-2b",
    "qwen2-vl-7b": "qwen2-vl-7b",
}


def normalize_tokens(text: str) -> list[str]:
    parts = re.split(r"[\s\t]+", text.strip())
    return [p for p in parts if p]


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


def load_gt(path: Path) -> list[str]:
    return normalize_tokens(path.read_text(encoding="utf-8"))


def token_edit_distance(pred: list[str], gt: list[str]) -> int:
    """Minimum single-token insert/delete/replace edits (Levenshtein at token level).

    Relative major/minor key signatures and common/cut-time aliases are treated as equal.
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


def score_tokens(pred: list[str], gt: list[str]) -> dict:
    n_gt = len(gt)
    errors = token_edit_distance(pred, gt)
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


def load_predictions_jsonl(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    if not path.exists():
        return preds
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            row = json.loads(line)
            fname = row.get("filename") or row.get("image") or ""
            preds[fname] = row.get("prediction", "")
    return preds


def load_predictions_from_batches(
    predictions_dir: Path,
    model: str,
    batches_dir: Path,
) -> dict[tuple[str, str], str]:
    """Map (group, filename) -> prediction using batch JSON order."""
    preds: dict[tuple[str, str], str] = {}
    model_dir = predictions_dir / model
    if not model_dir.is_dir():
        return preds
    for entries_path in sorted(batches_dir.glob("batch_*.json")):
        batch_name = entries_path.stem
        entries = json.loads(entries_path.read_text(encoding="utf-8"))
        pred_path = model_dir / f"{batch_name}.jsonl"
        if not pred_path.is_file():
            continue
        pred_lines = [
            ln.strip().lstrip("\ufeff")
            for ln in pred_path.read_text(encoding="utf-8-sig").splitlines()
            if ln.strip().lstrip("\ufeff")
        ]
        for entry, pred_line in zip(entries, pred_lines, strict=False):
            row = json.loads(pred_line)
            key = (entry["group"], entry["filename"])
            preds[key] = row.get("prediction", "")
    return preds


def load_prediction_for_entry(
    entry: dict,
    model: str,
    predictions_dir: Path,
    batches_dir: Path | None,
    flat_preds: dict[str, str],
    batch_preds: dict[tuple[str, str], str],
) -> str:
    key = (entry["group"], entry["filename"])
    if key in batch_preds:
        return batch_preds[key]
    return flat_preds.get(entry["filename"], "")


def score_selection(
    selection: list[dict],
    model: str,
    predictions_dir: Path,
    batches_dir: Path | None,
) -> list[dict]:
    flat_preds = load_predictions_jsonl(predictions_dir / f"{model}.jsonl")
    batch_preds: dict[tuple[str, str], str] = {}
    if batches_dir and batches_dir.is_dir():
        batch_preds = load_predictions_from_batches(predictions_dir, model, batches_dir)

    rows: list[dict] = []
    for entry in selection:
        pred_text = load_prediction_for_entry(
            entry, model, predictions_dir, batches_dir, flat_preds, batch_preds
        )
        gt_tokens = load_gt(Path(entry["ground_truth_semantic_path"]))
        pred_tokens = normalize_tokens(pred_text)
        s = score_tokens(pred_tokens, gt_tokens)
        rows.append(
            {
                "group": entry["group"],
                "model": model,
                "filename": entry["filename"],
                "has_prediction": bool(pred_text.strip()),
                **s,
            }
        )
    return rows


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


def group_sort_key(group: str) -> int:
    try:
        return GROUP_ORDER.index(group)
    except ValueError:
        return len(GROUP_ORDER)


def pivot_accuracies(
    all_scores: list[dict],
    models: list[str],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    aggregates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_scores:
        aggregates[(row["group"], row["model"])].append(row)

    groups = sorted({r["group"] for r in all_scores}, key=group_sort_key)
    matrix: dict[str, dict[str, float]] = {}
    for group in groups:
        matrix[group] = {}
        for model in models:
            rows = aggregates.get((group, model), [])
            scored = [r for r in rows if r.get("has_prediction")]
            if scored:
                matrix[group][model] = round(
                    sum(r["accuracy_pct"] for r in scored) / len(scored), 2
                )

    model_avg: dict[str, float] = {}
    for model in models:
        model_rows = [
            r for r in all_scores if r["model"] == model and r.get("has_prediction")
        ]
        if model_rows:
            model_avg[model] = round(
                sum(r["accuracy_pct"] for r in model_rows) / len(model_rows), 2
            )

    return groups, matrix, model_avg


def pivot_exact_match_pct(
    all_scores: list[dict],
    models: list[str],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    aggregates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_scores:
        aggregates[(row["group"], row["model"])].append(row)

    groups = sorted({r["group"] for r in all_scores}, key=group_sort_key)
    matrix: dict[str, dict[str, float]] = {}
    for group in groups:
        matrix[group] = {}
        for model in models:
            rows = aggregates.get((group, model), [])
            scored = [r for r in rows if r.get("has_prediction")]
            if scored:
                exact = sum(1 for r in scored if r.get("exact_match"))
                matrix[group][model] = round(exact / len(scored) * 100.0, 2)

    model_avg: dict[str, float] = {}
    for model in models:
        model_rows = [
            r for r in all_scores if r["model"] == model and r.get("has_prediction")
        ]
        if model_rows:
            exact = sum(1 for r in model_rows if r.get("exact_match"))
            model_avg[model] = round(exact / len(model_rows) * 100.0, 2)

    return groups, matrix, model_avg


def render_pivot_table(
    groups: list[str],
    matrix: dict[str, dict[str, float]],
    model_avg: dict[str, float],
    models: list[str],
) -> str:
    headers = ["Dataset", *[MODEL_SHORT[m] for m in models]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for group in groups:
        row = [group, *[str(matrix[group].get(m, "—")) for m in models]]
        lines.append("| " + " | ".join(row) + " |")
    avg_row = ["**Average**", *[str(model_avg.get(m, "—")) for m in models]]
    lines.append("| " + " | ".join(avg_row) + " |")
    return "\n".join(lines)


def render_pivot_markdown(
    all_scores: list[dict], total_samples: int, models: list[str]
) -> str:
    groups, matrix, model_avg = pivot_accuracies(all_scores, models)
    _, exact_matrix, exact_model_avg = pivot_exact_match_pct(all_scores, models)
    lines = [
        f"Samples scored: {total_samples}.",
        "",
        "## Token accuracy %",
        "",
        render_pivot_table(groups, matrix, model_avg, models),
        "",
        "## Exact match %",
        "",
        "Exact match % = share of samples with zero token edits after canonicalization.",
        "",
        render_pivot_table(groups, exact_matrix, exact_model_avg, models),
    ]
    return "\n".join(lines) + "\n"


def render_pivot_html_table(
    groups: list[str],
    matrix: dict[str, dict[str, float]],
    model_avg: dict[str, float],
    models: list[str],
) -> str:
    model_headers = "".join(f"<th>{MODEL_SHORT[m]}</th>" for m in models)
    body_rows = []
    for group in groups:
        cells = "".join(f"<td>{matrix[group].get(m, '—')}</td>" for m in models)
        body_rows.append(f"<tr><td>{group}</td>{cells}</tr>")
    avg_cells = "".join(f"<td>{model_avg.get(m, '—')}</td>" for m in models)
    body_rows.append(f"<tr><td><strong>Average</strong></td>{avg_cells}</tr>")
    return "\n    ".join(body_rows), model_headers


def render_pivot_html(
    all_scores: list[dict], total_samples: int, rubric: str, models: list[str]
) -> str:
    groups, matrix, model_avg = pivot_accuracies(all_scores, models)
    _, exact_matrix, exact_model_avg = pivot_exact_match_pct(all_scores, models)
    accuracy_rows, model_headers = render_pivot_html_table(
        groups, matrix, model_avg, models
    )
    exact_rows, _ = render_pivot_html_table(
        groups, exact_matrix, exact_model_avg, models
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OMR Benchmark Scores</title>
</head>
<body>
  <h1>OMR Benchmark Scores</h1>
  <p>{rubric}</p>
  <p>Samples scored: {total_samples}.</p>
  <h2>Token accuracy %</h2>
  <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>Dataset</th>{model_headers}</tr>
    {accuracy_rows}
  </table>
  <h2>Exact match %</h2>
  <p>Exact match % = share of samples with zero token edits after canonicalization.</p>
  <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>Dataset</th>{model_headers}</tr>
    {exact_rows}
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=BENCH_DIR / "selection_100.json",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=BENCH_DIR / "predictions_100",
    )
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=None,
        help="batch JSON folder (required when filenames repeat, e.g. manual set)",
    )
    parser.add_argument(
        "--scores-json",
        type=Path,
        default=BENCH_DIR / "scores_100.json",
    )
    parser.add_argument(
        "--scores-html",
        type=Path,
        default=BENCH_DIR / "scores_for_gdocs.html",
    )
    parser.add_argument(
        "--scores-md",
        type=Path,
        default=BENCH_DIR / "scores.md",
    )
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Score predictions_10, predictions_10_corrupt, and predictions_manual",
    )
    parser.add_argument(
        "--vlm-shortlist",
        action="store_true",
        help="Score VLM shortlist models across all benchmark suites",
    )
    args = parser.parse_args()

    rubric = (
        "Token-level edit distance against ground-truth semantic tokens; each single-token "
        "insert, delete, or replace counts as one edit, with alignment chosen to minimize "
        "total edits. Relative major/minor key signatures (same accidentals) and "
        "common/cut-time aliases (4/4↔C, 2/2↔C/) count as equal. "
        "Accuracy % = max(0, (1 − edits ÷ ground-truth token count) × 100)."
    )

    if args.all_benchmarks or args.vlm_shortlist:
        models = VLM_SHORTLIST_MODELS if args.vlm_shortlist else MODELS
        if args.vlm_shortlist:
            args.scores_json = BENCH_DIR / "scores_vlm_shortlist.json"
            args.scores_md = BENCH_DIR / "scores_vlm_shortlist.md"
            args.scores_html = BENCH_DIR / "scores_vlm_shortlist.html"
        suites = [
            (
                "Original holdout (10 per style)",
                BENCH_DIR / "selection_10.json",
                BENCH_DIR / "predictions_10",
                BENCH_DIR / "batches_10",
            ),
            (
                "Corrupt holdout (10 per style)",
                BENCH_DIR / "selection_10_corrupt.json",
                BENCH_DIR / "predictions_10_corrupt",
                BENCH_DIR / "batches_10_corrupt",
            ),
            (
                "Manual golden set (20 handwritten + 20 digital)",
                BENCH_DIR / "selection_manual.json",
                BENCH_DIR / "predictions_manual",
                BENCH_DIR / "batches_manual",
            ),
        ]
        all_scores: list[dict] = []
        total_samples = 0
        for _title, sel_path, pred_dir, batch_dir in suites:
            selection = json.loads(sel_path.read_text(encoding="utf-8"))
            total_samples += len(selection)
            for model in models:
                all_scores.extend(
                    score_selection(selection, model, pred_dir, batch_dir)
                )

        title = (
            "# OMR VLM Shortlist Benchmark Scores"
            if args.vlm_shortlist
            else "# OMR Benchmark Scores"
        )
        env_notes = ""
        if args.vlm_shortlist:
            env_notes = "\n".join(
                [
                    "## Environment",
                    "",
                    "- Zero-shot pretrained weights only (no fine-tuning).",
                    "- Machine: CPU-only, 16 GB RAM, no CUDA.",
                    "- `transformers==4.49.0` (Florence-2 remote code incompatible with 5.x).",
                    "- `qwen2-vl-7b` skipped on CPU (requires GPU or >16 GB RAM).",
                    "- All 100 benchmark images present on disk (camera originals restored).",
                    "",
                ]
            )
        md_content = "\n".join(
            [
                title,
                "",
                rubric,
                "",
                env_notes,
                render_pivot_markdown(all_scores, total_samples, models).strip(),
                "",
            ]
        )
        html_content = render_pivot_html(all_scores, total_samples, rubric, models)

        args.scores_json.parent.mkdir(parents=True, exist_ok=True)
        args.scores_json.write_text(
            json.dumps(all_scores, indent=2), encoding="utf-8"
        )
        args.scores_md.write_text(md_content, encoding="utf-8")
        args.scores_html.write_text(html_content, encoding="utf-8")
        print(f"Wrote {args.scores_json} ({len(all_scores)} rows)")
        print(f"Wrote {args.scores_md}")
        print(f"Wrote {args.scores_html}")
        return

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    batches_dir = args.batches_dir
    all_scores = []
    for model in MODELS:
        all_scores.extend(
            score_selection(selection, model, args.predictions_dir, batches_dir)
        )

    args.scores_json.parent.mkdir(parents=True, exist_ok=True)
    args.scores_json.write_text(json.dumps(all_scores, indent=2), encoding="utf-8")

    md_content = "\n".join(
        [
            "# OMR Benchmark Scores",
            "",
            rubric,
            "",
            render_pivot_markdown(all_scores, len(selection), MODELS).strip(),
            "",
        ]
    )
    args.scores_md.write_text(md_content, encoding="utf-8")
    args.scores_html.write_text(
        render_pivot_html(all_scores, len(selection), rubric, MODELS),
        encoding="utf-8",
    )
    print(md_content)


if __name__ == "__main__":
    main()
