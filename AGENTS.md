# Repository Agent Instructions

## Project

Project: `Stroke Lesion Segmentation`
Repository root: `.`

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

- `subagent-orchestration`: execution-mode selection, scoped normal subagents,
  automatic independent Worktree Chat/task creation, LunaMax long-task workers,
  automatic result collection when supported, manual top-level-thread fallback,
  monitoring, fixer flow, and evidence-based acceptance.
- `level3-review`: correctness-sensitive independent review for medical/scientific
  preprocessing, geometry, leakage, metrics, checkpoint/runtime contracts,
  cross-module interfaces, destructive behavior, or other high-risk changes.
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

## Agent roles

The main agent is coordinator, architect, integration manager, acceptance
decision maker, Git coordinator, and final reporter.

Preserve main-agent context for architecture, protocol, integration, and
acceptance.

Before delegation, use `subagent-orchestration`:

- normal scoped subagent for localized/short tasks;
- automatically created independent Codex Worktree Chat/task using LunaMax for
  substantial or long-running implementation/review work when supported;
- user-created top-level LunaMax thread only as fallback when automatic creation
  or result retrieval is unavailable;
- persistent external process for multi-hour training or similar execution.

When the main agent creates an independent task, preserve its task/thread/worktree
IDs and retrieve the final HANDOFF automatically when supported.

Do not require manual HANDOFF copy/paste when the result can be fetched reliably.

An independent worker must not recursively hand the same core task to another
independent worker. It may use only small scoped normal subagents for focused
investigation, tests, review, or fixer work.

Do not duplicate the same implementation concurrently between parent, normal
subagents, and independent workers.

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
- documentation reflects verified behavior.

## Final report

Keep the report concise and evidence-dense: what changed, files, validation and
results, reviewer result when required, unresolved issues, commit hash, and final
Git status.
