---
description: Stage all changes, draft a commit message from the diff, fix any pre-commit hook failures, and repeat until the commit succeeds.
---

Commit all pending changes in the PhysioMotion4D repository.

$ARGUMENTS

Instructions:

0. Run `git branch --show-current` to check the active branch.
   If the branch is `main`, stop immediately and report:
   "ERROR: Refusing to commit directly to main. Please switch to a feature branch first."
   Do not proceed further.

1. Run `git diff HEAD` and `git status` to understand what has changed.
   - Read any modified source files that are not self-explanatory from the diff alone.
   - Do NOT commit files that look like secrets, large binaries, or generated artefacts that belong in .gitignore.
   - Do NOT add any untracked files to the commit
   - ONLY perform the equivalent to a `git commit -a`

2. Draft a commit message following the project convention (match style of recent `git log --oneline -10`):
   - Subject line: `<TAG>: <imperative summary>` (≤72 chars), where TAG is one of:
     `ENH` (new feature / enhancement), `FIX` (bug fix), `REF` (refactor),
     `TST` (tests only), `DOC` (docs/comments only), `MNT` (maintenance / config).
   - Optional body: 1–3 sentences explaining *why*, not *what*.

3. Attempt the commit:
   ```
   git commit -a -m "<subject>" -m "<body with co-author line>"
   ```

4. If the commit fails because a pre-commit hook rejected it:
   a. Read the hook output carefully.
   b. Fix every reported issue (formatting, lint errors, type errors, test failures, etc.).
      - For `ruff` formatting/lint: run `ruff check . --fix && ruff format .`
      - For `mypy` errors: fix the type annotations in the flagged files.
      - For other hook failures: diagnose and fix the root cause; do NOT use `--no-verify`.
   c. Return to step 3 and retry — repeat until the commit succeeds or you have exhausted reasonable fixes.
   d. If an issue cannot be fixed automatically (e.g. a failing test unrelated to the current changes), report it to the user and stop.

5. After a successful commit, print the one-line commit summary (`git log --oneline -1`).
