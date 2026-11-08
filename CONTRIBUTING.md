# Contributing

For a numerical issue, include the commit, Python and package versions, the
command run, and the failing check or result. Please avoid attaching raw report
narratives or unrelated personal information.

Before proposing a code change, run:

```bash
python run_presentation.py --in-process
```

The run refits the models and checks predictions, chronological labels, history,
ties, and presentation values. Explain any intended change to the candidate
definition, labels, fitting windows, or metrics explicitly.

Keep the original frozen inputs in `analysis/inputs/` unchanged. New data
snapshots belong in a separately documented experiment. Preserve the distinction
between a statistical threshold crossing and a clinically validated finding.

The release manifest records the distributed files before execution. A rerun
changes timestamps, some binary metadata, and generated outputs. If a release is
updated, regenerate `MANIFEST_SHA256.json` for the intended tracked files and
record the reason; do not change hashes merely to hide a failing comparison.

See [NOTICE.md](NOTICE.md) for the current reuse status.
