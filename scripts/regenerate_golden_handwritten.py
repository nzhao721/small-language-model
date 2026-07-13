"""Regenerate golden_set handwritten images with per-score MUSCIMA++ writers."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from mashcima.primus_adapter import convert_primus_annotation_to_mashcima_annotation
from mashcima.vocabulary import parse_annotation_into_token_groups
from PIL import Image
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_layout import read_primus_agnostic, read_primus_semantic  # noqa: E402
from demo_corrupted_gt_examples import (  # noqa: E402
    corrupt_tokens,
    extract_aligned_pairs,
    is_sem_alignable,
    mashcima_annotation_from_semantic,
    prepare_mashcima_for_render,
    rebuild_mashcima_tokens,
    rebuild_semantic_body,
    split_semantic,
)
from demo_mashcima_camera_distort import camera_like_distort, seed_for_name  # noqa: E402
from generate_mashcima_batch import (  # noqa: E402
    DEFAULT_MUSCIMA_DIR,
    DEFAULT_PRIMUS_TGZ,
    ensure_muscima_cache,
    image_to_uint8,
    resize_long_edge,
    synthesize_handwritten,
)
from generate_corrupt_batch import (  # noqa: E402
    DEFAULT_PRIMUS_DIR,
    load_manifest_records,
    render_corrupt_image,
)
from image_compress import write_jpeg_bgr_under_kb  # noqa: E402
from mashcima_appearance import apply_mashcima_appearance  # noqa: E402
from mashcima_writers import DEFAULT_WRITER_MODE, configure_mashcima_writers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_MANIFEST = ROOT / "golden_set" / "manifest.jsonl"
CORRUPT_MANIFEST = ROOT / "data" / "corrupt" / "manifest.jsonl"
LOG_DIR = ROOT / "data" / "logs"
DEFAULT_MAX_DIM = 768
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def write_handwritten_jpeg(img: Image.Image, out_path: Path, max_dim: int | None) -> Path:
    """Match optimized mashcima/corrupt regen: 768px long edge + fast JPEG."""
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    bgr = resize_long_edge(bgr, max_dim)
    out_path = Path(out_path)
    for stale in (out_path, *(out_path.with_suffix(ext) for ext in IMAGE_EXTS)):
        stale.unlink(missing_ok=True)
    return write_jpeg_bgr_under_kb(bgr, out_path.with_suffix(".png"))


def norm_semantic(text: str) -> str:
    return " ".join(text.split())


def find_corruption_replay(
    incipit_id: str,
    target_semantic: str,
    *,
    max_seed: int = 25_000_000,
) -> tuple[int, float, str]:
    sem_orig = read_primus_semantic(incipit_id, DEFAULT_PRIMUS_DIR)
    agnostic = read_primus_agnostic(incipit_id, DEFAULT_PRIMUS_DIR, DEFAULT_PRIMUS_TGZ)
    target = norm_semantic(target_semantic)
    for seed in range(max_seed):
        rng = random.Random(seed)
        fraction = rng.uniform(0.33, 0.5)
        replayed, mash = corrupt_tokens(sem_orig, agnostic, rng, fraction)
        if norm_semantic(replayed) == target:
            return seed, fraction, mash
    raise RuntimeError(
        f"Could not replay corruption for {incipit_id} within {max_seed:,} seeds"
    )


def manifest_replay(
    corrupt_id: str,
    incipit_id: str,
    target_semantic: str,
) -> tuple[int, float, str] | None:
    """Replay corruption using seed stored in data/corrupt/manifest.jsonl."""
    row = load_manifest_records(CORRUPT_MANIFEST).get(corrupt_id)
    if not row or row.get("status") != "ok":
        return None
    seed = row.get("corruption_seed")
    fraction = row.get("corrupt_fraction")
    if seed is None or fraction is None:
        return None
    sem_orig = read_primus_semantic(incipit_id, DEFAULT_PRIMUS_DIR)
    agnostic = read_primus_agnostic(incipit_id, DEFAULT_PRIMUS_DIR, DEFAULT_PRIMUS_TGZ)
    rng = random.Random(int(seed))
    rng.uniform(0.33, 0.5)
    replayed, mash = corrupt_tokens(sem_orig, agnostic, rng, float(fraction))
    if norm_semantic(replayed) == norm_semantic(target_semantic):
        return int(seed), float(fraction), mash
    rng2 = random.Random(int(seed))
    replayed2, mash2 = corrupt_tokens(sem_orig, agnostic, rng2, float(fraction))
    if norm_semantic(replayed2) == norm_semantic(target_semantic):
        return int(seed), float(fraction), mash2
    return None


def infer_mash_for_corrupted_semantic(incipit_id: str, corrupted_semantic: str) -> str:
    """Build a Mashcima annotation from corrupted semantic when RNG replay is unavailable."""
    sem_orig = read_primus_semantic(incipit_id, DEFAULT_PRIMUS_DIR)
    agnostic = read_primus_agnostic(incipit_id, DEFAULT_PRIMUS_DIR, DEFAULT_PRIMUS_TGZ)
    mashcima = convert_primus_annotation_to_mashcima_annotation(agnostic)
    if mashcima is None:
        raise ValueError(f"agnostic conversion failed for {incipit_id}")

    _, orig_body = split_semantic(sem_orig.split())
    _, corr_body = split_semantic(corrupted_semantic.split())
    pairs = extract_aligned_pairs(orig_body, mashcima.split())

    orig_sem = [t for t in orig_body if is_sem_alignable(t)]
    corr_sem = [t for t in corr_body if is_sem_alignable(t)]
    matcher = difflib.SequenceMatcher(None, orig_sem, corr_sem)
    new_pairs: list[tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                new_pairs.append((corr_sem[j1 + offset], pairs[i1 + offset][1]))
            continue
        if tag == "replace":
            for offset in range(j2 - j1):
                ref = min(i1 + offset, len(pairs) - 1) if pairs else 0
                mash = pairs[ref][1] if pairs else "qr"
                new_pairs.append((corr_sem[j1 + offset], mash))
            continue
        if tag == "insert":
            ref = max(i1 - 1, 0)
            mash = pairs[ref][1] if pairs else "qr"
            for offset in range(j2 - j1):
                new_pairs.append((corr_sem[j1 + offset], mash))
            continue

    rebuilt_body = rebuild_semantic_body(corr_body, new_pairs)
    header, _ = split_semantic(corrupted_semantic.split())
    rebuilt_semantic = "\t".join(header + rebuilt_body) + "\t"
    if norm_semantic(rebuilt_semantic) != norm_semantic(corrupted_semantic):
        raise ValueError(f"inferred semantic mismatch for {incipit_id}")
    return " ".join(rebuild_mashcima_tokens(mashcima.split(), new_pairs))


def resolve_corrupt_mash(
    corrupt_id: str,
    incipit_id: str,
    corrupted_semantic: str,
    *,
    corruption_seed: int | None,
    corrupt_fraction: float | None,
    max_seed: int,
) -> tuple[int, float, str]:
    if corruption_seed is not None and corrupt_fraction is not None:
        sem_orig = read_primus_semantic(incipit_id, DEFAULT_PRIMUS_DIR)
        agnostic = read_primus_agnostic(incipit_id, DEFAULT_PRIMUS_DIR, DEFAULT_PRIMUS_TGZ)
        rng = random.Random(corruption_seed)
        replayed, mash = corrupt_tokens(sem_orig, agnostic, rng, corrupt_fraction)
        if norm_semantic(replayed) == norm_semantic(corrupted_semantic):
            return corruption_seed, corrupt_fraction, mash

    replayed = manifest_replay(corrupt_id, incipit_id, corrupted_semantic)
    if replayed is not None:
        return replayed

    try:
        return find_corruption_replay(incipit_id, corrupted_semantic, max_seed=min(max_seed, 10_000))
    except RuntimeError:
        pass

    mash = infer_mash_for_corrupted_semantic(incipit_id, corrupted_semantic)
    seed = zlib.crc32(corrupt_id.encode("utf-8")) % 2_147_483_647
    return seed, 0.4, mash


def render_original_handwritten(
    incipit_id: str,
    out_path: Path,
    *,
    max_dim: int | None = DEFAULT_MAX_DIM,
) -> None:
    agnostic = read_primus_agnostic(incipit_id, DEFAULT_PRIMUS_DIR, DEFAULT_PRIMUS_TGZ)
    converted = convert_primus_annotation_to_mashcima_annotation(agnostic)
    if converted is None:
        raise ValueError(f"agnostic conversion failed for {incipit_id}")
    warnings = parse_annotation_into_token_groups(converted)[1]
    if warnings:
        raise ValueError(f"annotation warnings for {incipit_id}: {warnings[:3]}")
    img = image_to_uint8(synthesize_handwritten(converted, incipit_id, max_dim))
    img = resize_long_edge(img, max_dim)
    img = apply_mashcima_appearance(img, key=incipit_id)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = camera_like_distort(pil, seed=seed_for_name(incipit_id))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img = resize_long_edge(img, max_dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jpeg_bgr_under_kb(img, out_path)


def render_corrupt_handwritten(
    corrupt_id: str,
    incipit_id: str,
    semantic_path: Path,
    out_path: Path,
    *,
    corruption_seed: int | None,
    corrupt_fraction: float | None,
    max_seed: int,
    max_dim: int | None = DEFAULT_MAX_DIM,
) -> tuple[int, float]:
    corrupted_semantic = semantic_path.read_text(encoding="utf-8")
    corruption_seed, corrupt_fraction, mash = resolve_corrupt_mash(
        corrupt_id,
        incipit_id,
        corrupted_semantic,
        corruption_seed=corruption_seed,
        corrupt_fraction=corrupt_fraction,
        max_seed=max_seed,
    )

    corrupted_mashcima = mashcima_annotation_from_semantic(
        corrupted_semantic,
        layout_annotation=mash,
    )
    render_annotation, _ = prepare_mashcima_for_render(corrupted_mashcima)
    img = render_corrupt_image(
        "handwritten",
        corrupted_semantic,
        render_annotation,
        corruption_seed,
        writer_key=corrupt_id,
        apply_appearance=True,
        max_dim=max_dim,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_handwritten_jpeg(img, out_path, max_dim)
    return corruption_seed, corrupt_fraction


def load_replay_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=GOLDEN_MANIFEST,
    )
    parser.add_argument(
        "--writers",
        default=DEFAULT_WRITER_MODE,
        choices=["per-score", "mixed", "single"],
    )
    parser.add_argument(
        "--replay-cache",
        type=Path,
        default=LOG_DIR / "golden_corrupt_replay.json",
    )
    parser.add_argument(
        "--max-seed",
        type=int,
        default=25_000_000,
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help="Cap output long edge (default: 768; use 0 to disable)",
    )
    args = parser.parse_args()
    max_dim = args.max_dim or None

    cv2.setNumThreads(1)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_muscima_cache(DEFAULT_PRIMUS_TGZ, DEFAULT_MUSCIMA_DIR)
    configure_mashcima_writers(args.writers)
    replay_cache = load_replay_cache(args.replay_cache)

    rows: list[dict] = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    targets = [
        r
        for r in rows
        if r.get("group") in {"handwritten_original", "handwritten_corrupt"}
    ]

    for row in tqdm(targets, desc="Golden handwritten", unit="image"):
        group = row["group"]
        image_path = Path(row["image_path"])
        label = row["id"]
        print(f"[golden] start {group} {label}", flush=True)
        if group == "handwritten_original":
            incipit_id = row["id"]
            render_original_handwritten(incipit_id, image_path, max_dim=max_dim)
            print(f"[golden] done {group} {label}", flush=True)
            continue

        corrupt_id = row["id"]
        incipit_id = row["incipit_id"]
        semantic_path = Path(row["gt_path"])
        cached = replay_cache.get(corrupt_id, {})
        seed, fraction = render_corrupt_handwritten(
            corrupt_id,
            incipit_id,
            semantic_path,
            image_path,
            corruption_seed=cached.get("corruption_seed"),
            corrupt_fraction=cached.get("corrupt_fraction"),
            max_seed=args.max_seed,
            max_dim=max_dim,
        )
        print(f"[golden] done {group} {label}", flush=True)
        if corrupt_id not in replay_cache:
            replay_cache[corrupt_id] = {
                "corruption_seed": seed,
                "corrupt_fraction": fraction,
            }
            args.replay_cache.write_text(
                json.dumps(replay_cache, indent=2), encoding="utf-8"
            )

    print(f"Regenerated {len(targets)} golden handwritten images", flush=True)


if __name__ == "__main__":
    main()
