# Publish this prepared repository

Suggested repository: **AlpayNatanael/VAERS-Signal-Prioritization**.
The delivered Git ZIP includes the files, a `main` branch, and an initial commit.
Extract it and open a terminal in the `VAERS-Signal-Prioritization` folder.

If GitHub CLI is installed, sign in to your own account and create the repository:

```bash
gh auth login
gh repo create AlpayNatanael/VAERS-Signal-Prioritization --public --source=. --remote=origin --push --description "Reproducible VAERS signal prioritization with chronological machine-learning evaluation, an executed notebook, and a research poster."
```

If the GitHub repository already exists and is empty, add it as the remote and
push instead of creating it again:

```bash
git remote add origin https://github.com/AlpayNatanael/VAERS-Signal-Prioritization.git
git push -u origin main
```

If `origin` is already configured, inspect `git remote -v` and use the existing
correct remote. Do not force-push over an existing repository's history.

Before publishing updates, verify the release files with
`python scripts/check_manifest.py`. The recorded numerical checks are documented
in the main README and `execution_report.json`. Recommended repository topics
are `vaers`, `pharmacovigilance`, `machine-learning`, and
`reproducible-research`.

The software license remains unselected; see [NOTICE.md](../NOTICE.md).
The accepted print poster is distributed as a PDF in `poster/`.
The optional raw-data rebuild is separate from the verified presentation command.

The main README links to the executed notebook through Jupyter nbviewer for
browser reading. GitHub's HTML file view displays source code; readers should
use **Download raw file** and open the saved HTML locally to read the version
with code hidden.
