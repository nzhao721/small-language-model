"""Paper texture and handwriting-ink appearance for Mashcima renders.

Applied after Mashcima synthesis (float ink-on-black → uint8 black-on-white).
Staff-line geometry/darkness is varied earlier in ``generate_staff_lines``.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import cv2
import numpy as np


def rng_for_key(key: str, salt: str = "mashcima-appearance") -> np.random.Generator:
    """Stable per-sample RNG so the same incipit gets a consistent look."""
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _to_gray(img: np.ndarray) -> tuple[np.ndarray, bool]:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), True
    return img, False


def _from_gray(gray: np.ndarray, was_color: bool) -> np.ndarray:
    if was_color:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def apply_ink_darkness(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    strength_range: tuple[float, float] = (0.42, 1.0),
) -> np.ndarray:
    """Vary how dark the handwriting / staff ink is (black-on-white uint8).

    ``strength`` near 1.0 keeps near-black strokes; lower values lift ink toward gray
    without washing out to near-invisible pale strokes.
    """
    gray, was_color = _to_gray(img)
    h, w = gray.shape
    strength = float(rng.uniform(*strength_range))

    # Coarse pen-pressure map (wide swings across the page).
    lh, lw = max(3, h // 12), max(3, w // 12)
    coarse = rng.uniform(0.65, 1.12, size=(lh, lw)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    # Finer stroke-level mottling so individual glyphs aren't uniformly gray.
    fh, fw = max(6, h // 4), max(6, w // 4)
    fine = rng.uniform(0.8, 1.15, size=(fh, fw)).astype(np.float32)
    fine = cv2.resize(fine, (w, h), interpolation=cv2.INTER_LINEAR)

    local = np.clip(coarse * fine * strength, 0.32, 1.0)

    f = gray.astype(np.float32)
    out = np.clip(255.0 - (255.0 - f) * local, 0, 255).astype(np.uint8)
    return _from_gray(out, was_color)


def apply_paper_texture(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    grain_sigma_range: tuple[float, float] = (2.5, 7.0),
    blotch_amp_range: tuple[float, float] = (2.0, 7.5),
) -> np.ndarray:
    """Add fine fiber grain and small-scale shade variation to paper regions."""
    gray, was_color = _to_gray(img)
    h, w = gray.shape
    paper = gray.astype(np.float32)

    # Prefer applying texture on paper; leave dark ink mostly alone.
    paper_mask = np.clip((paper - 35.0) / 180.0, 0.0, 1.0)

    # Build all texture at reduced resolution, then one upsample (same look, less work).
    sh, sw = max(h // 2, 32), max(w // 2, 64)
    grain = rng.normal(0.0, float(rng.uniform(*grain_sigma_range)), size=(sh, sw)).astype(
        np.float32
    )

    bh, bw = max(h // 3, 24), max(w // 3, 48)
    blotch = rng.normal(0.0, 1.0, size=(bh, bw)).astype(np.float32)
    blotch = cv2.GaussianBlur(blotch, (0, 0), sigmaX=0.6, sigmaY=0.6)
    blotch = cv2.resize(blotch, (sw, sh), interpolation=cv2.INTER_LINEAR)
    blotch *= float(rng.uniform(*blotch_amp_range))

    mid = rng.normal(0.0, float(rng.uniform(1.5, 4.0)), size=(sh, sw)).astype(np.float32)

    if rng.random() < 0.7:
        sth, stw = max(8, sh // 2), max(16, sw // 2)
        streak = rng.normal(0.0, 1.0, size=(sth, stw)).astype(np.float32)
        streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=1.2, sigmaY=0.35)
        streak = cv2.resize(streak, (sw, sh), interpolation=cv2.INTER_LINEAR)
        blotch = blotch + streak * float(rng.uniform(1.0, 3.0))

    texture = cv2.resize(grain + blotch + mid, (w, h), interpolation=cv2.INTER_LINEAR)
    paper = paper + texture * paper_mask
    paper = paper + float(rng.uniform(-4.0, 3.0)) * paper_mask

    out = np.clip(paper, 0, 255).astype(np.uint8)
    return _from_gray(out, was_color)


def apply_mashcima_appearance(
    img: np.ndarray,
    key: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Full appearance pass: ink darkness then paper texture (single gray round-trip)."""
    if rng is None:
        rng = rng_for_key(key or "default")

    gray, was_color = _to_gray(img)
    # Run both stages on gray to avoid BGR↔GRAY churn between passes.
    inked = apply_ink_darkness(gray, rng)
    textured = apply_paper_texture(inked, rng)
    return _from_gray(textured, was_color)
