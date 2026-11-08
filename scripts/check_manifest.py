#!/usr/bin/env python3
"""Verify the distributed release files before regenerating any outputs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "MANIFEST_SHA256.json").read_text())
failures = []
for name, expected in manifest.items():
    path = (ROOT / name).resolve()
    if ROOT not in path.parents or not path.is_file():
        failures.append(f"Missing or invalid path: {name}")
        continue
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        failures.append(f"Changed: {name}")
if failures:
    raise SystemExit("\n".join(failures) + "\nGenerated outputs change after a rerun; see docs/DATA.md.")
print(f"Verified {len(manifest)} release file hashes.")
