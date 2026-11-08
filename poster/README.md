# Research poster

[AIPM_VAERS_Poster_2026_42x36.pdf](AIPM_VAERS_Poster_2026_42x36.pdf) is the
current print master supplied by Natanael Alpay on 8 November 2026:
**42 inches wide × 36 inches tall (landscape)**, with the compact contact footer
and UCI logos at the top. Print at actual size (100%). The PDF is preserved
exactly as uploaded.

The [source directory](source/) contains the archived LaTeX for the earlier
**36 × 42 portrait poster**, the two transparent logos, vector graphs, and
numeric macros. It does not reproduce the current landscape PDF; updated
LaTeX for the supplied PDF has not been provided.

To build the archived portrait version with a full TeX Live installation, or
in Overleaf using pdfLaTeX, compile twice:

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
