"""Persist attempt history, retry decisions and idempotent progress; never launch jobs."""

import csv
import time
from pathlib import Path

if __package__:
    from . import artifacts, trace_metrics
else:
    import artifacts
    import trace_metrics


def validate_attempt(attempt, method):
    if (attempt / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("nonzero launch exit")
    # Require the complete evidence set, not merely eight filenames.
    artifacts.read_json(attempt / "run/routes.json")
    return trace_metrics.summarize_trace(trace_metrics.load_traces(attempt / "run"), method, "cycle", artifacts.read_json(attempt / "run/result.json"))


def history(root, row):
    items = []
    for path in sorted((root / "attempts" / row["name"]).glob("attempt_*")):
        if not (path / "intent.json").exists():
            continue  # No child can start until this intent has been persisted.
        # intent is persisted before any child starts; a missing outcome is an interrupted attempt.
        entry = artifacts.read_json(path / "intent.json")
        if (path / "outcome.json").is_file():
            entry.update(artifacts.read_json(path / "outcome.json"))
        else:
            entry.update(state="interrupted", contaminated=None)
        entry["path"] = str(path)
        entry["valid"] = False
        if entry["state"] == "success":
            try:
                validate_attempt(path, row["method"])
                entry["valid"] = True
            except (OSError, ValueError, KeyError, TypeError, StopIteration):
                pass
        items.append(entry)
    return items


def next_attempt(items, rerun_contaminated=False):
    last = items[-1] if items else None
    if rerun_contaminated:
        if last and last.get("contaminated") is True:
            return dict(generation=last["generation"] + 1, retry=0, rerun_of=last["path"])
        if not last or not last.get("rerun_of"):
            return None
    if last and last["valid"]:
        return None
    if not last:
        return dict(generation=0, retry=0, rerun_of=None)
    retry = last["retry"] + (last["state"] != "blocked")
    if retry > 1:
        raise RuntimeError("automatic retry exhausted (original attempt plus one retry); evidence retained")
    return dict(generation=last["generation"], retry=retry, rerun_of=last.get("rerun_of"))


def attempt_directory(root, row, intent):
    parent = root / "attempts" / row["name"]
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"attempt_{len(list(parent.glob('attempt_*'))) + 1:03d}"
    path.mkdir()
    artifacts.save_json(path / "intent.json", {**intent, "created_ns": time.time_ns()})
    # Current view is compatible with Step 2. Every old attempt remains intact.
    current = root / row["name"]
    if current.exists() and not current.is_symlink():
        raise ValueError(f"refusing to replace a non-symlink launch directory: {current}")
    link = root / f".{row['name']}_{path.name}.tmp"
    link.symlink_to(path.relative_to(root), target_is_directory=True)
    link.replace(current)
    return path


def progress(root, row, path, outcome):
    metrics = outcome.get("metrics") or {}
    target = root / "progress.tsv"
    fields = ["method", "level", "repeat", "raw_max_gap_s", "wave_count", "plan_variant", "contaminated", "attempt", "exit_code", "ended_ns"]
    exists = target.exists()
    relative = str(path.relative_to(root))
    if exists:
        with target.open(newline="", encoding="utf-8") as stream:
            if any(r["attempt"] == relative for r in csv.DictReader(stream, delimiter="\t")):
                return
    with target.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({**row, **metrics, **{k: outcome[k] for k in ("contaminated", "exit_code", "ended_ns")}, "attempt": relative})
