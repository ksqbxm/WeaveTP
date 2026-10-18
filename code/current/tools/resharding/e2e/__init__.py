"""Unified serving driver shared by the fail-stop and the scale-up/down end-to-end experiments.

Pure-Python control plane (requests, ledger, admission, anchors, plan executor, timeline,
results) plus block-level KV, paged attention and a step loop whose GPU-facing pieces are
injected through RuntimeFactories; a fake backend drives everything on CPU.
"""
