"""Terminus-2 → SWE-agent format conversion pipeline."""


def run_pipeline(**kwargs):
    from preprocessing.pipeline import run_pipeline as _run
    return _run(**kwargs)


__all__ = ["run_pipeline"]
