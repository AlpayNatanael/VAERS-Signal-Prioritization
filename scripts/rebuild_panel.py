#!/usr/bin/env python3
"""Optionally reconstruct an aggregate panel without replacing frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "source"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--require-original-snapshot", action="store_true",
                        help="Fail unless every raw-file hash and the rebuilt panel match the archive.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    frozen = (ROOT / "analysis" / "inputs").resolve()
    if output == frozen or frozen in output.parents:
        parser.error("Choose an output directory outside the frozen analysis/inputs directory.")

    expected = {}
    for line in (ROOT / "docs" / "raw_data_sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[Path(name).name] = digest
    missing = [name for name in expected if not (args.raw_dir / name).is_file()]
    if missing:
        parser.error("Missing raw CSV files: " + ", ".join(missing))
    actual = {name: sha256(args.raw_dir / name) for name in expected}
    mismatches = [name for name in expected if expected[name] != actual[name]]
    if mismatches and args.require_original_snapshot:
        parser.error("Raw snapshot differs from the archive: " + ", ".join(mismatches))
    if mismatches:
        print(f"Using a different raw snapshot: {len(mismatches)} file hashes differ.", flush=True)

    import pandas as pd
    from vaers_pipeline import build_project_data

    print("Building the 2022–2026 panel through receipt quarter 2026Q2...", flush=True)
    long, audit, panel, candidates = build_project_data(args.raw_dir)
    unique_reports = int(long["VAERS_ID"].nunique())
    del long
    archived = pd.read_pickle(frozen / "quarterly_panel.pkl")
    difference = None
    try:
        pd.testing.assert_frame_equal(panel, archived, check_exact=True)
    except AssertionError as error:
        difference = str(error)[:1200]
    if difference and args.require_original_snapshot:
        raise RuntimeError("Rebuilt panel differs from the frozen panel:\n" + difference)

    output.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(output / "quarterly_panel.pkl")
    candidates.to_pickle(output / "emerging_signal_candidates.pkl")
    audit.to_csv(output / "raw_data_audit.csv", index=False)
    report = {
        "raw_sha256": actual,
        "raw_files_differing_from_archive": mismatches,
        "panel_matches_frozen_values": difference is None,
        "panel_difference": difference,
        "unique_reports": unique_reports,
        "panel_rows": len(panel),
        "candidate_rows": len(candidates),
        "complete_through": "2026Q2",
    }
    (output / "rebuild_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "raw_sha256"}, indent=2))
    print(f"Saved separate rebuilt files to {output}")


if __name__ == "__main__":
    main()
