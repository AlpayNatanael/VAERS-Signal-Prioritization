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

After publication, check the **Actions** tab for the first verification run.
The local numerical checks have passed; the hosted workflow still needs to run
in GitHub's environment. Recommended repository topics are `vaers`,
`pharmacovigilance`, `machine-learning`, and `reproducible-research`.

The software license remains unselected; see [NOTICE.md](../NOTICE.md).
The accepted print poster is in `poster/`, and its editable source is in
`poster/source/`. The raw-data rebuild and poster recompilation are optional
paths, separate from the verified presentation command.

If your ZIP extractor omits dotfiles, enable hidden-file extraction so it keeps
`.git`, `.github`, `.gitignore`, and `.gitattributes`.
