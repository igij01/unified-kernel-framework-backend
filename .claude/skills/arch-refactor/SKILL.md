---
name: arch-refactor
description: Drives architectural refactors of kernel-pipeline-backend through paired ADRs and per-step sub-agent dispatch with hard gates between discussion, ADR, plan, and implementation. Use when the user types /arch-refactor, OR when the user proposes a major architectural change that should be documented by an ADR (signals: splitting a class along conceptual lines, introducing a Protocol or pluggable boundary, extracting an abstraction, language like "refactor", "extract", "abstract", "modularize", "swap out", or mentioning an existing ADR that needs revision in step with code changes).
---

# Arch Refactor

Codifies the rhythm this repo's architectural changes have followed: discuss the conceptual split, write the ADR(s), plan the work, then implement step-by-step under user review. Reach for this skill whenever a change reshapes responsibilities across modules — not for bug fixes (use `bug-fix-and-ship`) or local cleanups.

The skill enforces hard gates between phases. It never advances on its own. If the user is silent or ambiguous at a gate, ask — do not guess.

## Phases

### Phase 1 — Discussion

Read the affected modules. Name the conceptual responsibilities being re-cut. Surface trade-offs as **options**, not decisions. Identify where the seams should land.

Do **not**:
- Draft an ADR.
- Write or edit code.
- Touch `docs/adr/`.

**Exit gate:** the user explicitly confirms the conceptual split. Phrases like "yes, split it that way" or "go ahead and write the ADR" count. Anything ambiguous — ask.

### Phase 2 — ADR drafting

Invoke the `architecture-decision-records` skill via the Skill tool for ADR mechanics (numbering, format, status lifecycle, index update). Do not duplicate that skill's content here.

Write the ADR(s) reflecting Phase 1's conclusions. Update `docs/adr/README.md` to list the new ADR(s).

**Paired ADRs are a judgement call.** The criterion is "two cleanly separable concerns, each warranting its own scope." If the discussion produced one decision with one set of consequences, write one ADR. If it produced two decisions whose consequences pull in different directions (e.g. an abstract Protocol *and* a concrete orchestrator), write paired ADRs that land together. There is no hard rule.

**Exit gate:** the user approves the drafted ADR(s).

### Phase 3 — Plan

Produce a numbered todo list. Two sections:

1. **Main-code items.** Each sized to a single coherent reviewable change (one module split, one Protocol introduced, one consumer migrated). Do not bundle unrelated edits into one item.
2. **Tests.** A single bundle at the end — all new tests, all updated tests, dispatched together in one final agent run.

**Exit gate:** the user approves the plan.

### Phase 4 — Implementation

For each main-code item, in order:

1. Dispatch a sub-agent with a self-contained prompt (see template below).
2. When the agent returns, **independently verify** its claims. The agent's report is a summary, not ground truth. Run grep, Read the touched files, run import smoke checks via the `run-in-container` skill against the `cuda130-torch` slot. Do not trust the agent blindly.
3. Stop. Report verified results to the user. Wait for review of the diff before moving to the next item.

After all main-code items are merged and reviewed, dispatch the **test bundle** as a single agent run. No per-step review for tests — review happens once at the end.

## Off-ramps

**Skip the ADR phase.** If discussion reveals the change is small, local, and reversible (renaming a private method, moving a helper, inlining a one-call function), recommend skipping Phase 2. Surface the option; the user decides.

**Abandon mid-flight.** If discussion or a draft ADR reveals the proposed split is wrong, return to Phase 1. Do not sunk-cost-defend the prior draft. "This isn't right, let's reconsider" is a legitimate signal at any gate.

## Project conventions

Most conventions (module dependency order, Protocol-over-ABC, backend isolation, pre-publication no-shims policy, build/test commands) live in `CLAUDE.md` and are already in context. Cross-reference rather than restate.

Conventions this skill must encode explicitly (not in `CLAUDE.md`):

- **Container.** All Python execution — imports, pytest, smoke checks — happens inside the `cuda130-torch` slot via the `run-in-container` skill. Sub-agent prompts must say so.
- **Test-time installs.** Inside the container: `pip install pytest-asyncio` for async tests; `pip install cupy-cuda13x` for CUDA-backend tests. Bundle these with `pip install -e /workspace -q` in one `bash -c` chain.
- **Verify, don't trust.** Sub-agent reports are summaries. After every agent run, independently grep / Read / run import smokes before reporting to the user.

## Sub-agent prompt template

Every implementation-phase sub-agent prompt must include:

- A directive to read the relevant ADR(s) at `docs/adr/NNNN-*.md` first.
- Concrete task description tied to specific files (and line numbers when stable).
- Constraints list: no compat shims, no behavior change unless specified, no unrelated edits, follow project Protocol-over-ABC guidance per `CLAUDE.md`.
- Sanity-check commands the agent must run before reporting — typically import smoke checks via the `run-in-container` skill against `cuda130-torch`, e.g. `python -c "from kernel_pipeline_backend.<module> import <symbol>"`.
- A bounded report length (under 200 words) listing files touched, smokes run, and any deviation from the spec.

Minimal example body:

```
Read docs/adr/00NN-<slug>.md before starting.

Task: <one-paragraph concrete change tied to specific files>.

Constraints:
- No backwards-compat shims or deprecation aliases (pre-publication).
- No behavior change beyond what the ADR specifies.
- Use typing.Protocol for new abstractions per CLAUDE.md.
- Touch only the files listed; flag if more are needed.

Sanity checks (run via the run-in-container skill, slot cuda130-torch):
- pip install -e /workspace -q
- python -c "from kernel_pipeline_backend.<mod> import <sym>"

Report under 200 words: files touched, smokes run, deviations.
```

If the template grows beyond this, extract to `.claude/skills/arch-refactor/templates/agent-prompt.md` and reference from here. For now it stays inline.

## Composition

This skill invokes the `architecture-decision-records` skill during Phase 2 and the `run-in-container` skill within sub-agent prompts and verification steps.
