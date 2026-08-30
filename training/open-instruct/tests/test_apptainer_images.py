"""Tests for resolving container references against a local SIF pool."""

import pytest

from open_instruct.environments import apptainer_images


@pytest.fixture(autouse=True)
def clear_resolver_state(monkeypatch):
    monkeypatch.delenv("SWERL_APPTAINER_SIF_DIR", raising=False)
    apptainer_images._sif_hash_index.cache_clear()
    apptainer_images._LOGGED_IMAGES.clear()


def test_unconfigured_pool_leaves_image_reference_unchanged():
    image = "docker://registry.example/team/image:latest"

    assert apptainer_images.prefer_local_sif(image) == image


def test_exact_sanitized_reference_resolves_to_nonempty_sif(tmp_path, monkeypatch):
    image = "docker://registry.example/team/image:latest"
    sif = tmp_path / "registry.example__team__image__latest.sif"
    sif.write_bytes(b"sif")
    monkeypatch.setenv("SWERL_APPTAINER_SIF_DIR", str(tmp_path))

    assert apptainer_images.prefer_local_sif(image) == str(sif)


def test_mirror_reference_resolves_by_trailing_hash(tmp_path, monkeypatch):
    image = "registry.example/mirror/image:abc123def"
    sif = tmp_path / "upstream__image__abc123def.sif"
    sif.write_bytes(b"sif")
    monkeypatch.setenv("SWERL_APPTAINER_SIF_DIR", str(tmp_path))

    assert apptainer_images.prefer_local_sif(image) == str(sif)


def test_zero_byte_match_and_unmatched_reference_are_not_used(tmp_path, monkeypatch):
    image = "registry.example/team/image:latest"
    (tmp_path / "registry.example__team__image__latest.sif").touch()
    monkeypatch.setenv("SWERL_APPTAINER_SIF_DIR", str(tmp_path))

    assert apptainer_images.prefer_local_sif(image) == image


def test_hash_index_is_scoped_to_configured_directory(tmp_path, monkeypatch):
    token = "abc123def"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_sif = first_dir / f"first__{token}.sif"
    second_sif = second_dir / f"second__{token}.sif"
    first_sif.write_bytes(b"first")
    second_sif.write_bytes(b"second")
    image = f"registry.example/image:{token}"

    monkeypatch.setenv("SWERL_APPTAINER_SIF_DIR", str(first_dir))
    assert apptainer_images.prefer_local_sif(image) == str(first_sif)

    monkeypatch.setenv("SWERL_APPTAINER_SIF_DIR", str(second_dir))
    assert apptainer_images.prefer_local_sif(image) == str(second_sif)


def test_missing_configured_directory_is_an_error(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    monkeypatch.setenv("SWERL_APPTAINER_SIF_DIR", str(missing))

    with pytest.raises(FileNotFoundError):
        apptainer_images.prefer_local_sif("registry.example/image:latest")
