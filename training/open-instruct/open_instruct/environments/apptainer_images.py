"""Resolve container references against an operator-provided local SIF pool."""

import os
import re
from functools import cache
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_HASH_TOKEN = re.compile(r"[0-9a-f]{6,64}")
_LOGGED_IMAGES: set[str] = set()


def _strip_scheme(image: str) -> str:
    return _SCHEME.sub("", image).strip().strip("/")


class MissingLocalSifError(RuntimeError):
    """A configured local SIF pool did not contain a requested image."""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sif_name_for_image(image: str) -> str | None:
    name = re.sub(r"[^A-Za-z0-9._-]+", "__", _strip_scheme(image)).strip("._-")
    return f"{name}.sif" if name else None


@cache
def _sif_hash_index(sif_dir: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted(Path(sif_dir).iterdir(), key=lambda item: item.name):
        if path.suffix != ".sif":
            continue
        token = path.stem.rsplit("__", 1)[-1].lower()
        if _HASH_TOKEN.fullmatch(token):
            index.setdefault(token, path.name)
    return index


def _image_hash_candidates(image: str) -> list[str]:
    ref = _strip_scheme(image)
    path, tag = ref, None
    match = re.match(r"^(.+?):([A-Za-z0-9._-]+)$", ref)
    if match:
        path, tag = match.group(1), match.group(2)
    candidates: list[str] = []
    for token in (tag, path.rsplit("/", 1)[-1]):
        normalized = token.lower() if token else ""
        if _HASH_TOKEN.fullmatch(normalized) and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _log_resolution(image: str, path: Path, *, hash_match: bool) -> None:
    if image in _LOGGED_IMAGES:
        return
    qualifier = " (hash match)" if hash_match else ""
    logger.info("Using local Apptainer SIF%s for %s: %s", qualifier, image, path)
    _LOGGED_IMAGES.add(image)


def prefer_local_sif(image: str) -> str:
    """Return a matching local SIF when the operator configures a SIF pool."""

    if image.startswith(("/", "./")) or image.endswith(".sif"):
        return image
    configured_dir = os.environ.get("SWERL_APPTAINER_SIF_DIR")
    if not configured_dir:
        return image

    sif_dir = Path(configured_dir)
    hash_index = _sif_hash_index(str(sif_dir))
    sif_name = _sif_name_for_image(image)
    if sif_name:
        exact = sif_dir / sif_name
        if _is_nonempty_file(exact):
            _log_resolution(image, exact, hash_match=False)
            return str(exact)
    for token in _image_hash_candidates(image):
        hashed_name = hash_index.get(token)
        if hashed_name:
            candidate = sif_dir / hashed_name
            if _is_nonempty_file(candidate):
                _log_resolution(image, candidate, hash_match=True)
                return str(candidate)

    # Missing images in a configured local pool must fail before remote conversion.
    if _env_flag("SWERL_ALLOW_REMOTE_IMAGE_FALLBACK", False):
        logger.warning(
            "No local SIF for %s in %s; falling back to a remote ref because "
            "SWERL_ALLOW_REMOTE_IMAGE_FALLBACK is set. This triggers worker-side "
            "OCI->SIF conversion.",
            image,
            configured_dir,
        )
        return image
    raise MissingLocalSifError(
        f"No local SIF for {image!r} in {configured_dir!r} "
        f"(expected {sif_name!r} or a hash match). Refusing to fall back to a "
        f"remote image reference: worker-side OCI->SIF conversion is slow and "
        f"fails under restricted egress. Build the missing image into the pool, "
        f"or set SWERL_ALLOW_REMOTE_IMAGE_FALLBACK=1 to allow remote pulls."
    )
