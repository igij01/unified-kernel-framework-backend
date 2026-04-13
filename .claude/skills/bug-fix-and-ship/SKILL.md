---
name: bug-fix-and-ship
description: End-to-end bug fix workflow for kernel-pipeline-backend. Reads a structured bug report (issues/*.md), fixes the affected source file, adds regression tests, runs the full test suite inside a Docker slot container, then commits and pushes. Use when the user says "fix the bug in issues/NNN-*.md", "resolve issue NNN", or provides a path to a bug report and wants the fix shipped.
---

# Bug Fix and Ship

End-to-end workflow: bug report → fix → regression tests → container test run → commit → push.

## Arguments

```
/bug-fix-and-ship <path-to-bug-report> [<container-slot>]
```

- `<path-to-bug-report>` — path to the markdown bug report (e.g. `issues/001-kernel-hasher-triton-jitfunction.md`)
- `<container-slot>` — Docker slot to run tests in (default: `cuda130-torch`)

## Instructions

Work through these steps in order. Complete each fully before moving to the next.

### Step 1 — Read the bug report

Read the bug report file at the path given by the user. Extract:

- **Component**: file path and function name that is broken
- **Severity**: blocker / major / minor
- **Root cause**: the specific incorrect assumption or missing branch
- **Suggested fix**: the approach recommended in the report (treat as a starting point, not gospel — verify it is correct and complete before applying)

### Step 2 — Read the affected source

Read the full source file identified in the bug report. Understand the surrounding logic so the fix fits the existing code style and does not break adjacent behaviour.

### Step 3 — Apply the fix

Edit only the affected file(s). Follow these constraints:

- Do not modify unrelated code
- Keep the fix minimal — change only what is broken
- Do not add error handling for impossible cases
- Preserve existing docstrings unless the fix changes the documented behaviour
- If the suggested fix in the bug report is incomplete (e.g. handles one level of nesting but not two), correct it — document the deviation in a comment

### Step 4 — Write regression tests

Open the existing test file for the affected module (e.g. `tests/versioning/test_hasher.py` for `versioning/hasher.py`). Add a new test class named `Test<ShortIssueName>` (e.g. `TestTritonJITFunctionUnwrapping`).

Test class requirements:

- Docstring citing the issue number and one-line summary
- One test that directly reproduces the crash/failure described in the report (`test_<thing>_does_not_raise` or similar)
- One test per additional wrapper depth or variant mentioned in the report
- One determinism/equivalence test (same input → same output as the plain case)
- One negative test (different inputs → different outputs still holds)
- Use fakes/stubs instead of real third-party imports when possible (duck-typing contract only)
- Do not import the actual external library (e.g. triton) unless the test is explicitly a GPU integration test

### Step 5 — Run the full test suite in the container

Determine the correct container slot (from the user's argument or default `cuda130-torch`).

Check whether the image is already built:

```bash
docker images cuda-playground/dev:<slot> --format "{{.ID}}"
```

If empty, build it first:

```bash
bash cuda_docker_env/scripts/build-env.sh <slot>
```

Run the full test suite (install the package and any required extras first):

```bash
docker compose -f cuda_docker_env/docker-compose.yml --profile <slot> \
    run --rm dev-<slot> bash -c \
    "pip install -e /workspace -q && \
     pip install cupy-cuda13x pytest-asyncio -q && \
     pytest /workspace/tests/ -v"
```

**Important notes on the container environment:**
- The `unified-kernel-framework-backend/` directory is bind-mounted as `/workspace` inside the container — use `/workspace` as the package root, not a subdirectory
- The container uses conda; wrap all commands in a single `bash -c "..."` string
- Run all docker commands from the repo root (where `cuda_docker_env/` lives)

Evaluate the results:

- **All pass** → proceed to Step 6
- **Pre-existing failures unrelated to the fix** → note them, proceed to Step 6 and call them out in the commit message
- **New failures introduced by the fix** → diagnose and iterate on the fix before proceeding

### Step 6 — Commit and push

Stage only the files changed by the fix (source file + test file). Do not stage unrelated changes.

Write a commit message in this format:

```
fix <short description of what was broken> (issue #NNN)

<1-2 sentences explaining the root cause and what the fix does differently.
Call out any deviation from the suggested fix in the bug report.>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Push to the current branch's remote tracking branch:

```bash
git push
```

If the push fails due to authentication, report the exact error and stop — do not attempt workarounds.

### Step 7 — Report back

Summarise what was done:

- Which file was fixed and what changed (link to line range)
- Which test class was added and how many tests (link to file)
- Container used and test count (pass / fail)
- Commit SHA

## Edge cases

**Bug report suggests a fix that is incomplete**: Apply the corrected version. Document in a comment why the single-level fix was insufficient (e.g. `# Walk the full chain, not just one level — Autotuner wraps JITFunction wraps fn`).

**Multiple files affected**: Fix all of them in the same commit. List each in the commit message body.

**No existing test file for the module**: Create one following the naming convention `tests/<module>/test_<file>.py` with an `__init__.py`.

**Container image not built**: Build it with `build-env.sh` before running tests. This can take several minutes — inform the user.

**Push fails (SSH not configured)**: Report the remote URL and the exact error. Ask the user to fix credentials rather than switching strategies.

## Example invocation

```
/bug-fix-and-ship issues/001-kernel-hasher-triton-jitfunction.md cuda130-torch
```

This reads `issues/001-kernel-hasher-triton-jitfunction.md`, fixes `kernel_pipeline_backend/versioning/hasher.py`, adds `TestTritonJITFunctionUnwrapping` to `tests/versioning/test_hasher.py`, runs all 678 tests in `cuda130-torch`, then commits and pushes.
