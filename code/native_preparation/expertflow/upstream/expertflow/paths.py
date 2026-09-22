"""Utilities for resolving model and artifact paths.

ExpertFlow experiments often need direct access to sharded model weight files.
Callers may pass either a local directory, a glob, or a Hugging Face repo id.
"""

import os
from glob import glob

from huggingface_hub import snapshot_download


def resolve_model_state_path(model_ref, allow_download=True):
    """Return a local filesystem path for a model reference.

    Args:
        model_ref: Local directory, glob expression, or Hugging Face repo id.
        allow_download: If true, resolve repo ids through the Hugging Face cache.
    """
    if not model_ref:
        raise ValueError("model_ref must be a local path, glob, or Hugging Face repo id")

    expanded = os.path.expandvars(os.path.expanduser(model_ref))
    if os.path.exists(expanded):
        return expanded

    matches = sorted(glob(expanded))
    if matches:
        return matches[0]

    looks_like_local_path = (
        os.path.isabs(expanded)
        or expanded.startswith(".")
        or expanded.startswith("~")
    )
    if looks_like_local_path:
        raise FileNotFoundError(f"Model path does not exist: {model_ref}")

    if not allow_download:
        raise FileNotFoundError(f"Model path is not local and downloads are disabled: {model_ref}")

    return snapshot_download(repo_id=model_ref)


def model_config_ref(model_name, state_path):
    """Prefer a local config.json when the resolved state path has one."""
    if state_path:
        config_path = os.path.join(state_path, "config.json")
        if os.path.exists(config_path):
            return state_path
    return model_name
