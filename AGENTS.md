# Repository Agent Instructions

## Project

Project: `Stroke Lesion Segmentation`
Repository root: `.`
Integration branch: `master`

Primary source:
- `standalone_nnunet2d/`
- project-level training, evaluation, and data-processing scripts in the repository root

Primary tests:
- `standalone_nnunet2d/tests/`
- `tests/`

Primary validation:
`python -m pytest -q`

Real nnU-Net data may exist only on the training server. Do not assume real
patient data exists in the local checkout.

## Instruction hierarchy

Apply instructions in this order:

1. this root `AGENTS.md`;
2. the closest applicable subdirectory `AGENTS.md`, if present;
3. explicit user instructions.

Lower-level instructions may add stricter requirements but must not silently
weaken repository-level safety rules.

## Workflow skills

Global reusable skills:

- `subagent-orchestration`: implementation-mode selection, scoped normal
  subagents, independent Worktree Chat/tasks for substantial implementation,
  ordinary independent reviewer conversations, worker-profile selection,
  monitoring, escalation, fixer flow, and evidence-based acceptance.
- `project-memory`: concise repository-level state for cross-session,
  cross-terminal, branch, and worktree continuity under `.project-memory/`.
- `level3-review`: correctness-sensitive independent read-only review for
  medical/scientific preprocessing, geometry, leakage, metrics,
  checkpoint/runtime contracts, cross-module interfaces, destructive behavior,
  or other high-risk changes. Prefer an ordinary independent reviewer
  conversation; do not create a reviewer worktree merely to satisfy
  independence.
- `test-validation`: focused/affected/full validation, TDD, trustworthy test
  evidence, and temporary-test-artifact discipline.
- `scientific-experiment-integrity`: experiment classification, leakage
  prevention, preflight/formal separation, reproducibility, and result integrity.

Project-local skills:

- `standalone-nnunet-testing`: `standalone_nnunet2d` tests, checkpoint-heavy
  tests, synthetic fixtures, model-loading sentinels, pytest temp storage, and
  project-specific validation details.
- `medical-experiment-integrity`: Dataset501/Dataset502/CI-1 safety, fixed splits,
  OOF rules, medical-data leakage constraints, source-space evaluation, and
  project-specific scientific reporting.

Load only the skill(s) needed for the current task.

## Project memory

Use the global `project-memory` skill for repository-level continuity.

Canonical integration branch:

`master`

Memory layout:

`.project-memory/{STATUS.md,GOALS.md,NEXT.md,LOG.md}`

On `master`, local `.project-memory/` is the canonical repository-level memory.
In another branch or independent worktree, treat checked-out local
`.project-memory/` as a branch-local snapshot and prefer canonical `master`
memory according to the `project-memory` skill, for example:

- `git show master:.project-memory/STATUS.md`
- `git show master:.project-memory/GOALS.md`
- `git show master:.project-memory/NEXT.md`

Project-specific content rules:

- `STATUS.md`: verified pipeline/model/data-interface state, accepted baselines,
  materially relevant active workstreams, and key scientific invariants.
- `GOALS.md`: stable segmentation/research goal and current milestone.
- `NEXT.md`: only immediate engineering/experimental actions and concrete
  blockers.
- `LOG.md`: short accepted-task entries with affected component, material change,
  and validation/preflight/formal-run status.

Never store patient data, patient identifiers, raw medical-image contents,
secrets, training logs, full tracebacks, or large metric tables in project
memory. Numerical results may be summarized only when provenance and experiment
status are clear. Preserve the distinction between synthetic, smoke, preflight,
baseline, tuning, and formal evaluation; a preflight result is never a formal
result merely because it appears in memory.

Independent worker worktrees must not edit canonical project memory unless their
task brief explicitly grants memory ownership. They return HANDOFF evidence; the
parent/main agent synchronizes only accepted, validated/integrated state.

## Global scientific invariants

- Dataset501 is the established DWI-only baseline.
- Preserve the existing patient-level 5-fold `splits_final.json`; do not
  regenerate or randomly replace it without explicit authorization.
- Formal segmentation metrics use the original full-volume patient space unless
  an experiment explicitly defines otherwise.
- Real medical images and labels are read-only source data.
- Store crops, ROIs, predictions, checkpoints, reports, and other derived data
  separately from raw data.
- GT must never influence inference-time ROI or derived-input construction.
- Prediction-guided Stage 2 training must use out-of-fold Stage 1 predictions.
- Synthetic, smoke, and preflight results must never be presented as formal
  clinical or experimental results.

## Agent roles and default implementation ownership

The main agent is coordinator, architect/planner, integration manager, acceptance
decision maker, reviewer/fixer coordinator, Git coordinator, and final reporter.

Preserve main-agent context for architecture, protocol, integration, acceptance,
and user-facing decisions.

For repository tasks that modify production source code or tests, implementation
is delegated by default according to `subagent-orchestration`.

Default ownership:

- localized implementation -> normal `Luna xhigh` subagent;
- substantial or multi-module implementation -> independent `LunaMax` Worktree
  Chat/task;
- focused fixer or validator -> `Luna xhigh`;
- Level 3 independent review -> separate ordinary read-only `LunaMax` reviewer
  conversation by default.

Task smallness alone does not justify direct main-agent source/test
implementation.

The main agent may directly modify coordination artifacts such as `AGENTS.md`,
plans, task briefs, review notes, validation checklists, and canonical project
memory when appropriate.

Direct main-agent production source/test implementation is allowed only when:

- the user explicitly requests it;
- no supported worker interface is available;
- supported worker channels have failed and documented fallback is justified;
- only minimal coordination/integration glue remains and delegation would create
  more risk than value.

If direct main-agent source/test implementation occurs, record the reason in the
final report.

Normal subagents are leaf workers: they must not create other subagents,
reviewers, validators, fixers, or independent tasks. Independent review, fixer
creation, validation-worker creation, and implementation-mode escalation remain
main-agent responsibilities.

An independent Worktree Chat/task is the top-level implementation worker for its
core task. It must not recursively hand the same core implementation to another
independent task. It may use only small scoped normal subagents for focused
investigation, targeted tests, narrow review, or small fixer work; those normal
subagents remain leaf workers.

Do not duplicate the same implementation concurrently between parent, normal
subagents, and independent workers.

## Independent review policy

Choose review strength separately from implementation mode.

A small implementation can still require Level 3 review if it affects sensitive
geometry, preprocessing, leakage, metrics, data safety, checkpoint/runtime
contracts, public interfaces, destructive behavior, or other correctness-
sensitive logic.

For Level 3 review, prefer a separate ordinary reviewer conversation using
`LunaMax` with read-only scope.

Reviewer independence means:

- the reviewer did not implement the change;
- the reviewer has a separate context;
- the exact final diff/commit/snapshot is identified;
- the reviewer independently inspects the risky code path and evidence;
- the reviewer returns `PASS`, `BLOCKING`, and optional `NON-BLOCKING` findings.

A separate reviewer Git worktree is **not** required merely to count as
independent.

Use a reviewer worktree only when an isolated checkout is materially necessary,
for example when the reviewer must run commands against its own filesystem state
or the intended final state cannot otherwise be exposed reliably through an
exact commit, diff, patch, or snapshot.

Main-agent inspection and implementation-worker self-review do not count as
independent Level 3 review.

## Scope and implementation economy

Do not modify unrelated files.

Prefer the smallest implementation that satisfies the requested behavior.

Reuse existing modules/helpers/interfaces/pipelines before adding new ones.

Do not add speculative extensibility, duplicate pipelines, unnecessary
factories/registries/adapters/wrappers/feature flags, broad refactors,
reformatting, renaming, or unrelated cleanup.

## Working-tree and Git safety

Assume uncommitted changes may belong to the user.

Do not reset, overwrite, restore, stash, delete, or commit unrelated work.

Do not create nested Git repositories, force-push, or rewrite history without
authorization.

Do not push unless requested.

Before commit inspect:
- `git status`
- `git diff`
- `git diff --check`

Stage explicit intended files when unrelated changes may exist.

After staging inspect:
- `git diff --cached`
- `git diff --cached --check`

Do not commit real datasets, secrets, large generated artifacts, or temp files.

## Data and filesystem safety

Treat real datasets/user data as read-only by default.

Do not overwrite, rename, convert in place, delete, or silently repair raw data.

Do not place large generated artifacts directly in source directories.

Enforce generated-output boundaries with normalized resolved paths.

Observation/logging/telemetry must not change model, optimizer, seed, split,
preprocessing, stopping, checkpoint selection, or metric semantics.

## Production and tests

Do not weaken production behavior to make tests pass.

Tests should exercise real production logic whenever practical.

When fixing a bug:
1. reproduce it;
2. add/update the relevant test;
3. apply the minimal fix;
4. rerun the focused test;
5. rerun the affected suite;
6. run broader validation when risk justifies it.

Use `standalone-nnunet-testing` for project-specific test details.

## External-source fidelity

When implementing from a paper/spec/API/protocol/reference project, prioritize
primary sources.

Distinguish source-specified settings, project assumptions, deliberate
deviations, and unresolved details.

Do not silently replace ambiguity with common practice.

## Dependencies and security

Do not casually install or broadly upgrade dependencies.

Never expose or commit credentials, API keys, tokens, private keys, passwords,
or private user data.

Do not bypass access restrictions or run destructive commands without clear need
and authorization.

## Final acceptance

Before completion verify, as applicable:

- requested behavior implemented;
- trustworthy tests pass;
- reviewer blockers resolved;
- no unrelated files changed;
- generated outputs isolated;
- raw data unchanged;
- intended files only committed;
- unrelated pre-existing user changes untouched;
- documentation reflects verified behavior;
- canonical project memory is synchronized when the task changed accepted
  repository state.

## Final report

Keep the report concise and evidence-dense: what changed, files, validation and
results, reviewer result when required, unresolved issues, commit hash, final Git
status, whether direct main-agent source/test implementation occurred and why,
and whether canonical project memory was synchronized when applicable.
