# Research poster

[AIPM_VAERS_Poster_2026_36x42.pdf](AIPM_VAERS_Poster_2026_36x42.pdf) is the
accepted print master: **36 inches wide × 42 inches tall**, with the compact
contact footer and UCI logos at the top. Print at actual size (100%).

The [source directory](source/) contains the recovered editable LaTeX, the two
transparent logos, vector graphs, and numeric macros. The accepted PDF is
preserved unchanged. The source was not recompiled for this repository because
the available TeX installation lacks the required format/packages.

With a full TeX Live installation, or in Overleaf using pdfLaTeX, compile twice:

```bash
cd poster/source
pdflatex -interaction=nonstopmode -halt-on-error AIPM_VAERS_Poster_2026_36x42.tex
pdflatex -interaction=nonstopmode -halt-on-error AIPM_VAERS_Poster_2026_36x42.tex
```

Required packages include `geometry`, `lmodern`, `helvet`, `amsmath`, `amssymb`,
`graphicx`, `xcolor`, `booktabs`, `tabularx`, `ragged2e`, `tikz`, `tcolorbox`,
`textpos`, `qrcode`, and `hyperref`. The second pass positions the background.
Inspect the compiled result before replacing the accepted print master.

The poster's row-bootstrap interval and feature-ablation figures are archived
poster analyses; they are not refitted by the presentation verification command.
Available supporting tables are preserved in `source/archive_tables/`.
The presentation notebook adds the history baseline, explicit tie handling,
and more detail on first-observed versus returning crossings. See the notebook
for the fullest interpretation of the study.
