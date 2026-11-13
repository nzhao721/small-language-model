"""Canonical on-disk layout for PrIMuS-family datasets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIMUS_DIR = ROOT / "data" / "primus"
MASHCIMA_DIR = ROOT / "data" / "mashcima"
CAMERA_PRIMUS_DIR = ROOT / "data" / "camera-primus"
DEFAULT_PRIMUS_TGZ = ROOT / "data" / "archives" / "primusCalvoRizoAppliedSciences2018.tgz"


def primus_images_dir(primus_dir: Path = PRIMUS_DIR) -> Path:
    return primus_dir / "images"


def primus_ground_truth_dir(primus_dir: Path = PRIMUS_DIR) -> Path:
    return primus_dir / "ground_truth"


def primus_agnostic_dir(primus_dir: Path = PRIMUS_DIR) -> Path:
    return primus_dir / "agnostic"


def primus_semantic_path(incipit_id: str, primus_dir: Path = PRIMUS_DIR) -> Path:
    flat = primus_ground_truth_dir(primus_dir) / f"{incipit_id}.semantic"
    if flat.is_file():
        return flat
    # legacy nested layout
    for pkg in ("package_aa", "package_ab"):
        legacy = primus_dir / pkg / incipit_id / f"{incipit_id}.semantic"
        if legacy.is_file():
            return legacy
    return flat


def primus_png_path(incipit_id: str, primus_dir: Path = PRIMUS_DIR) -> Path:
    flat = primus_images_dir(primus_dir) / f"{incipit_id}.png"
    if flat.is_file():
        return flat
    for pkg in ("package_aa", "package_ab"):
        legacy = primus_dir / pkg / incipit_id / f"{incipit_id}.png"
        if legacy.is_file():
            return legacy
    return flat


def primus_agnostic_path(incipit_id: str, primus_dir: Path = PRIMUS_DIR) -> Path:
    flat = primus_agnostic_dir(primus_dir) / f"{incipit_id}.agnostic"
    if flat.is_file():
        return flat
    for pkg in ("package_aa", "package_ab"):
        legacy = primus_dir / pkg / incipit_id / f"{incipit_id}.agnostic"
        if legacy.is_file():
            return legacy
    return flat


def mashcima_images_dir(mashcima_dir: Path = MASHCIMA_DIR) -> Path:
    return mashcima_dir / "images"


def mashcima_ground_truth_dir(mashcima_dir: Path = MASHCIMA_DIR) -> Path:
    return mashcima_dir / "ground_truth"


def mashcima_image_path(incipit_id: str, mashcima_dir: Path = MASHCIMA_DIR) -> Path:
    flat = mashcima_images_dir(mashcima_dir) / f"{incipit_id}.jpg"
    if flat.is_file():
        return flat
    legacy = mashcima_dir / f"{incipit_id}.jpg"
    if legacy.is_file():
        return legacy
    legacy_png = mashcima_dir / f"{incipit_id}.png"
    if legacy_png.is_file():
        return legacy_png
    return flat


def camera_images_dir(camera_dir: Path = CAMERA_PRIMUS_DIR) -> Path:
    return camera_dir / "images"


def camera_ground_truth_dir(camera_dir: Path = CAMERA_PRIMUS_DIR) -> Path:
    return camera_dir / "ground_truth"


def camera_distorted_path(incipit_id: str, camera_dir: Path = CAMERA_PRIMUS_DIR) -> Path:
    flat = camera_images_dir(camera_dir) / f"{incipit_id}_distorted.jpg"
    if flat.is_file():
        return flat
    legacy = (
        camera_dir / "Corpus" / incipit_id / f"{incipit_id}_distorted.jpg"
    )
    if legacy.is_file():
        return legacy
    return flat


def read_primus_agnostic(
    incipit_id: str,
    primus_dir: Path = PRIMUS_DIR,
    primus_tgz: Path = DEFAULT_PRIMUS_TGZ,
) -> str:
    path = primus_agnostic_path(incipit_id, primus_dir)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    if primus_tgz.is_file():
        import tarfile

        suffix = f"/{incipit_id}/{incipit_id}.agnostic"
        with tarfile.open(primus_tgz, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(suffix) or member.name.endswith(
                    f"{incipit_id}.agnostic"
                ):
                    with tar.extractfile(member) as f:
                        return f.read().decode("utf-8")
    raise FileNotFoundError(f"agnostic not found for {incipit_id}")


def read_primus_semantic(incipit_id: str, primus_dir: Path = PRIMUS_DIR) -> str:
    path = primus_semantic_path(incipit_id, primus_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")
