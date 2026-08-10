# Add Windows to Continuous Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the YAPA test suite on both `windows-latest` and `ubuntu-latest` in GitHub Actions CI so platform-specific regressions (like the JSON-session overwrite failure) are caught before release.

**Architecture:** Convert the single `runs-on: ubuntu-latest` test job into a job using an OS matrix (`windows-latest`, `ubuntu-latest`), and switch dependency installs to the lockfile-enforcing `--frozen` mode. The `test` job is also referenced via `workflow_call` from `release.yml`, so the matrix must continue to work when called. Mainline/PR-merge triggers and the existing `name: Test` stay unchanged so badge/release references keep working.

**Tech Stack:** GitHub Actions, `uv` (`astral-sh/setup-uv@v5`), `uv sync --frozen`, Python 3.13, pytest + pytest-cov.

## Global Constraints

- Test runner OS matrix MUST be `['windows-latest', 'ubuntu-latest']` (issue #39).
- Use `uv sync --frozen` for test/lint dependency installs (project has a committed `uv.lock`; lockfile-enforcing install is requested by the issue and makes CI reproducible).
- Keep `python-version: "3.13"` everywhere (per existing workflows and `requires-python = ">=3.13"`).
- Keep `actions/checkout@v4` and `astral-sh/setup-uv@v5` (existing workflow versions).
- Do NOT add SQLite/sqlmodel dependencies (AGENTS.md constraint).
- Do NOT change the lint job to a matrix — lint tooling (`ruff`, `ty`) is platform-independent; only the test suite needs the OS matrix.
- Do not let the per-job coverage gate interact badly with the matrix — coverage config in `pytest.ini` (`--cov-fail-under=80`) stays untouched; each matrix leg runs the full suite independently and already reaches ~94%.

---

## File Structure

- `.github/workflows/test.yml` — add OS matrix + `--frozen` install. This is the primary deliverable.
- `.github/workflows/lint.yml` — add `--frozen` install (consistency with test; not a matrix).
- `.github/workflows/release.yml` — depends on `test.yml` via `workflow_call`; no code change expected, but verified.
- `CONTRIBUTING.md:165-166` — update CI description to mention Windows.

These files change together (they describe the same CI pipeline) and are small, so the plan uses bite-sized verification steps rather than one big task.

---

### Task 1: Add OS matrix to the test workflow

**Files:**
- Modify: `.github/workflows/test.yml:16` (the `test` job)
- Verify: `.github/workflows/release.yml` (workflow_call consumer)

**Interfaces:**
- Consumes: existing `test` job referenced by `release.yml` via `needs: [lint, test]` and `uses: ./.github/workflows/test.yml`.
- Produces: a `test` job that runs on `['windows-latest', 'ubuntu-latest']`. Because release calls this workflow via `workflow_call`, the job id (`test`) and workflow `name: Test` are preserved verbatim.

- [ ] **Step 1: Edit the test job to use an OS matrix**

Change `.github/workflows/test.yml` job `test` so the runner and matrix look like this:

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: uv sync --dev --frozen

      - name: Test
        run: uv run pytest tests/ -v
```

Keep the `name: Test` header, the `on:` triggers, `concurrency` block, and the `workflow_call:` trigger exactly as-is.

- [ ] **Step 2: Validate YAML syntax**

Run:
```powershell
python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml')); print('OK')"
```
Expected: prints `OK`. If PyYAML is unavailable in the base interpreter, validate inside the project venv:
```powershell
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml')); print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 3: Confirm the workflow_call consumer is intact**

Read `.github/workflows/release.yml` and confirm the `test` job still calls `./.github/workflows/test.yml` with no extra job-id expectations. No edit is expected here; the matrix lives inside the called workflow and each matrix leg is a separate runner instance under the single `test` job. Record the finding (e.g. in the PR description) rather than changing the file.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: run test suite on Windows and Ubuntu OS matrix"
```

---

### Task 2: Use lockfile-enforcing install in the lint workflow

**Files:**
- Modify: `.github/workflows/lint.yml:26`

**Interfaces:**
- Consumes: nothing from Task 1; independent change for consistency.
- Produces: lint install using `uv sync --dev --frozen`, matching the test workflow and ensuring repeatable lint across environments.

- [ ] **Step 1: Update the lint install command**

In `.github/workflows/lint.yml`, change the "Install dependencies" step:

```yaml
      - name: Install dependencies
        run: uv sync --dev --frozen
```

- [ ] **Step 2: Validate YAML syntax**

Run:
```powershell
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/lint.yml')); print('OK')"
```
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/lint.yml
git commit -m "ci: enforce lockfile install in lint workflow"
```

---

### Task 3: Update CI documentation in CONTRIBUTING.md

**Files:**
- Modify: `CONTRIBUTING.md:165-166`

**Interfaces:**
- Consumes: Task 1's outcome (test suite now runs on Windows and Ubuntu).
- Produces: accurate developer-facing description of the CI pipeline. No test code depends on this.

- [ ] **Step 1: Update the CI paragraph**

Replace the current text at `CONTRIBUTING.md:165-166`:

```markdown
The CI runs lint and test workflows on every push and pull request to
`master`. The remote CI also runs the test suite on Windows and Ubuntu to
catch platform-specific regressions. The release workflow builds and
drafts a GitHub release when you push a tag matching `v*`.
```

Keep surrounding prose intact.

- [ ] **Step 2: Review the rendered paragraph**

Read `CONTRIBUTING.md` lines 155-168 to confirm the paragraph reads cleanly and references are accurate.

- [ ] **Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: note Windows coverage in CI description"
```

---

## Post-Plan Validation

After Tasks 1-3, exercise the workflow locally where possible and stage the PR:

- [ ] Run the full quality gate on the machine (this already exercises the Windows code path, since the dev environment here is `win32`):
  ```powershell
  uv run ruff check src/ tests/
  uv run ty check src/
  uv run pytest tests/
  ```
  Expected: ruff clean, ty clean, `343 passed`, coverage `>= 80%` (observed ~94%).
- [ ] Create a feature branch from `master`, push it, and open a PR to `development` per `CONTRIBUTING.md` so the GitHub Actions matrix (both OSes) actually runs and turns green.
- [ ] Re-title/link the PR to issue #39 (e.g. "Closes #39") so the enhancement is tracked.

## Self-Review

**1. Spec coverage:** Issue #39 asks for (a) an OS matrix incl. `windows-latest` + `ubuntu-latest` → Task 1; (b) tests running unchanged on both → Task 1 runs the full existing pytest suite per matrix leg, no test code changed; (c) lockfile-enforcing install where appropriate → Tasks 1 & 2 use `--frozen`. No gaps.

**2. Placeholder scan:** No "TBD"/"similar to above". All file paths, code, and commands are spelled out inline. The `workflow_call` check in Task 1 Step 3 is a concrete verification step with an explicit expected outcome, not an open-ended directive.

**3. Type consistency:** The `test` job id and workflow `name: Test` stay unchanged across the plan so `release.yml`'s `needs: [lint, test]` and `uses: ./.github/workflows/test.yml` continue to resolve. No function/method names are invented. Matrix key (`os`) and values match across Steps 1-3 references.