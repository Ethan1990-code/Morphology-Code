"""Restore hash-verified manuscript results into the scripts' expected locations.

Run only for the quick figure rebuild. Use a separate clone for raw-data reruns.
Different existing results are never overwritten.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def restore(root=ROOT):
    root = Path(root).resolve()
    records = json.loads((root / "provenance/source_manifest.json").read_text("utf-8"))
    jobs = []
    for record in records:
        rel = Path(record["file"])
        if rel.parts[0] != "reference_results":
            continue
        source = root / rel
        target = root / Path(*rel.parts[1:])
        if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
            raise ValueError("Manifest path leaves the repository")
        expected = record["published_sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed reference result: {rel}")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise FileExistsError(f"Refusing to overwrite different results: {target.relative_to(root)}")
        jobs.append((source, target))
    for source, target in jobs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    print(f"Restored/verified {len(jobs)} reference files; no analysis was rerun.")
    return len(jobs)


if __name__ == "__main__":
    restore()

