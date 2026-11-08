# Data and provenance

## Included snapshot

The repository includes aggregate inputs sufficient to rerun the notebook
without downloading raw VAERS files:

| File in `analysis/inputs/` | Contents |
|---|---|
| `quarterly_panel.pkl` | 228,652 pair-quarter rows, 2022Q1–2026Q2; counts, report-context summaries, and statistical scores |
| `reference_final_scored.pkl` | Archived final-quarter candidate features and predictions used for numerical comparison |
| `reference_rolling.csv` | Archived rolling results used for numerical comparison |

The panel has no report IDs or free-text narratives. It contains vaccine-type
and event-term identifiers plus aggregate statistics. The main panel is about
40 MiB uncompressed. It is committed as an ordinary Git file, so no separate
data transfer or Git LFS setup is required.

`analysis/FILE_SHA256.json` records the original verification archive. Before
every presentation run, the code checks the hashes of the frozen inputs,
scientific pipeline, verification scripts, and protocol. Some historical hashes
for generated result files will differ after a rerun; that file is archival
provenance, not the current release manifest.

The root `MANIFEST_SHA256.json` describes the current distributed repository.
Run `python scripts/check_manifest.py` immediately after cloning or unpacking to
check those bytes. Generated-file hashes can change when you rerun the notebook.

## Data dictionary

| Fields | Meaning |
|---|---|
| `vax_type`, `meddra_pt` | Vaccine-type and event-term pair |
| `quarter`, `q_int` | Receipt quarter and its integer representation |
| `a`, `b`, `c`, `d` | Pair-specific 2-by-2 report counts |
| `n_vax`, `n_symptom`, `N` | Vaccine, term, and total report counts |
| `serious_share`, `er_urgent_share`, `office_visit_share` | Fractions of pair reports with the specified indicators |
| `mean_age`, `female_share` | Pair-level demographic summaries; may be missing |
| `pharmacy_share`, `senior_facility_share`, `unknown_admin_share` | Vaccine-administration setting summaries |
| `prr`, `ror`, `ic` | Corrected disproportionality statistics |
| `prr_lcb95`, `ror_lcb95` | Lower 95% bounds |
| `expected`, `gp_shrinkage` | Expected co-report count and simple shrinkage proxy |
| `strong_signal`, `threshold_proximity` | Screening-rule indicator and baseline score |

Candidate features and exact next-quarter outcomes are rebuilt from the panel
on each verification run. They are not learned from the saved reference scores.

## Optional raw-data reconstruction

Source: [VAERS data](https://vaers.hhs.gov/data.html) and
[interpretation guidance](https://vaers.hhs.gov/data/dataguide.html).
Place the 2022–2026 `VAERSDATA`, `VAERSVAX`, and `VAERSSYMPTOMS` CSVs in
`data/raw/`, keeping their original filenames. The code requires the expanded
format containing `ORDER` in all three tables. Then run:

```bash
python scripts/rebuild_panel.py --require-original-snapshot
```

This checks all 15 raw-file hashes against [raw_data_sha256.txt](raw_data_sha256.txt),
calls the same corrected scientific pipeline, and compares the rebuilt panel
with the frozen panel. It writes a separate cache and audit under
`data/processed/`; it never overwrites `analysis/inputs/`.

For a newer download, omit `--require-original-snapshot`. The script records
hash mismatches and treats the result as a different snapshot. A matching
calendar range alone does not imply matching data: VAERS files change over time.
The raw snapshot's exact download date is not recorded in the available
provenance, so no download date is claimed here.

The initial repository verification reruns the frozen-panel experiment. Raw
ingestion is a separate optional operation and is not part of that numerical
verification. Raw data, rebuilt caches, and report-level data are excluded from
Git by default.
