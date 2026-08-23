# Repository Agent Instructions

## Project configuration

Project name:

`Stroke Lesion Segmentation`

Repository root:

`.`

Primary source directories:

- `standalone_nnunet2d/`
- project-level training, evaluation, and data-processing scripts in the repository root

Primary test directories:

- `standalone_nnunet2d/tests/`
- `tests/`

Local data directories, if applicable:

- nnU-Net raw, preprocessed, and training data are external/runtime data and may exist only on the training server
- real patient data must not be assumed to exist in the local checkout

Generated-output directories, if applicable:

- `results/`
- `trainer_eval/`
- `outputs/`
- nnU-Net result directories supplied at runtime

Primary validation command:

`python -m pytest -q`

The rules below are repository-wide unless a more specific `AGENTS.md`
exists in a subdirectory.

## Required project context

For tasks involving the stroke-segmentation project, nnU-Net, datasets,
training, evaluation, server execution, experiment results, or server paths:

1. Read `SERVER_PROJECT_STRUCTURE.md` before planning or modifying anything.
2. Treat `SERVER_PROJECT_STRUCTURE.md` as the project record of known server-side
   structure and resource locations.
3. Distinguish verified server information from entries marked unverified,
   planned, historical, or pending confirmation.
4. Do not invent missing server files, checkpoints, datasets, predictions,
   paths, or experiment results.
5. If required server resources are unavailable locally, prepare the necessary
   code and server-side commands instead of substituting fake data.
6. `AGENTS.md` defines repository workflow and safety rules;
   `SERVER_PROJECT_STRUCTURE.md` provides project/server context.

## Project-specific scientific constraints

Current primary task:

- acute ischemic stroke lesion segmentation from brain MRI;
- Dataset501 is the established DWI-only baseline dataset;
- patient-level 5-fold splits are fixed by the existing `splits_final.json`;
- do not regenerate or randomly replace the established splits without explicit authorization;
- formal segmentation metrics must be computed in the original full-volume patient space unless a specific experiment explicitly defines otherwise;
- real medical images and labels are read-only source data;
- ROI, cropped datasets, predictions, checkpoints, and evaluation reports are derived/generated data and must be stored separately;
- GT masks must never be used to derive inference-time ROI locations in formal coarse-to-fine experiments;
- Stage 2 coarse-to-fine training must use out-of-fold Stage 1 predictions when prediction-guided ROIs are required.


# Repository scope

This repository is the root workspace for the project.

The repository root is the only Git repository unless explicitly documented
otherwise.

Do not create nested Git repositories inside project subdirectories.

Keep independent components, models, services, experiments, or packages in
clearly separated directories.

Do not mix unrelated implementations into the same module solely for
convenience.


# Instruction hierarchy

Repository-level instructions apply to the entire repository.

More specific `AGENTS.md` files may exist in subdirectories.

When working inside a subdirectory:

1. follow this root `AGENTS.md`;
2. then follow the closest applicable subdirectory `AGENTS.md`;
3. then follow the explicit user instruction for the current task.

A lower-level instruction may add stricter project-specific requirements but
must not silently weaken repository-level safety rules.


# Main-agent role

The main agent acts primarily as:

- project coordinator;
- planner;
- task decomposer;
- integration manager;
- validator;
- reviewer coordinator;
- Git coordinator;
- final reporter.

The main agent must not directly modify implementation files by default.

Concrete file modifications must be delegated to clearly scoped subagents.

This includes modifications to:

- source code;
- tests;
- configuration;
- scripts;
- documentation;
- experiment definitions;
- build files;
- adapters;
- data-processing code.

If a modification is required, create a scoped subagent assignment instead of
silently editing the file from the main agent.


# Main-agent responsibilities

The main agent is responsible for:

- reading repository instructions;
- inspecting current repository state;
- understanding the requested goal;
- decomposing work into safe units;
- determining task dependencies;
- assigning scoped work to subagents;
- monitoring subagent completion;
- collecting subagent reports;
- inspecting diffs;
- running or coordinating tests;
- coordinating independent review;
- detecting integration problems;
- creating follow-up fixer tasks;
- deciding whether acceptance criteria are satisfied;
- staging only intended files;
- creating verified commits;
- summarizing final results for the user.

The main agent may execute read-only or validation operations such as:

- `git status`;
- `git diff`;
- `git log`;
- repository searches;
- file inspection;
- test execution;
- linting;
- type checking;
- build validation;
- read-only data inspection;
- process/status inspection.


# Subagent completion and monitoring

Launching a subagent does not complete the main agent's task.

When the current phase depends on a subagent result, the main agent must remain
responsible until that subagent reaches a terminal state.

The default workflow is:

delegate
→ monitor
→ wait
→ collect result
→ inspect
→ validate
→ continue

The main agent must distinguish:

- `RUNNING`: continue monitoring;
- `COMPLETED`: collect and validate the result;
- `FAILED`: collect failure evidence and follow the failure policy;
- `BLOCKED`: collect concrete blocker evidence and follow the decision gate.

The main agent must not return control to the user merely because a subagent
has been launched.

Do not end a task with statements such as:

“the subagent has been started”

when the current phase requires that subagent's result.

Unless the user explicitly requests asynchronous execution, wait for all
subagents required by the current gate before producing the phase result.

If a phase contains sequential work such as:

implementation
→ tests
→ reviewer
→ fixer
→ validation

the main agent should coordinate the complete sequence without requesting
additional user input unless a genuine decision gate is reached.


# Decision gates

Stop and request user input only when:

- the task explicitly requires user approval;
- multiple materially different implementation choices remain;
- a blocker requires expanding authorized scope;
- a destructive action requires approval;
- experimental protocol must be chosen;
- credentials, permissions, or external access are required;
- the next step would materially change the requested goal.

Do not introduce unnecessary approval gates for routine implementation,
testing, review, or validation.


# Subagent scope requirements

Every subagent assignment must define:

1. files or directories it may read;
2. exact files or directories it may modify;
3. files or directories it must not modify;
4. expected deliverable;
5. acceptance criteria.

A subagent must not expand its own scope.

If additional files must be modified, the subagent must stop and report why.

Do not allow two subagents to modify the same file concurrently.

If two tasks depend on the same file, execute them sequentially.


# Task decomposition economy

Do not create additional subagents merely to make the workflow appear more
structured.

For a small, localized implementation, prefer:

one scoped implementer
→ validation
→ one read-only reviewer when appropriate

Use additional subagents only when work is genuinely independent, requires
different expertise, or cannot safely share the same modification scope.

Do not split one small code change across multiple implementers.

Do not create parallel alternative implementations unless the user explicitly
requests comparison of those implementations.

Agent decomposition should reduce risk or execution time, not increase process
overhead.


# Recommended subagent roles

Use specialized roles when useful.

Examples include:

- auditor:
  read-only inspection of code, data, documentation, or external evidence;

- implementer:
  modifies only explicitly authorized implementation files;

- test implementer:
  modifies only explicitly authorized tests;

- documentation implementer:
  modifies documentation only;

- reviewer:
  read-only independent inspection;

- fixer:
  performs narrowly scoped corrections identified by tests or review;

- experiment runner:
  executes a frozen configuration without modifying implementation;

- result analyst:
  reads completed results and computes summaries without changing the run.

Roles should be scoped by responsibility rather than by arbitrary agent count.


# Reviewer policy

Reviewers should normally be read-only.

A reviewer must distinguish:

- `BLOCKING`
- `NON-BLOCKING`

A blocking finding must include:

- a clear issue;
- concrete evidence;
- why the issue invalidates the current acceptance criteria;
- the smallest reasonable fix scope.

Evidence should include at least one of:

- file and function;
- relevant code location;
- failing test;
- reproducible command;
- runtime artifact;
- deterministic behavior.

Do not classify the following as blocking by themselves:

- style preferences;
- optional extra tests;
- logging improvements;
- minor performance optimizations;
- documentation polish;
- harmless warnings;
- speculative concerns without evidence.

A reviewer must not return `BLOCKED` without identifying at least one
reproducible blocking issue.

If a previous blocker cannot be reproduced and no new blocker exists, report
`PASS` explicitly.


# Fixer policy

When tests or review identify a defect, the main agent should assign a new
scoped fixer.

The fixer should modify only the minimum files required to correct the
identified problem.

Do not use a blocker as justification for unrelated refactoring.

After a fix, rerun:

relevant test
→ affected test group
→ full validation when appropriate

Then repeat independent review if the blocker involved correctness or safety.


# Production code versus test code

Do not weaken production behavior merely to make tests pass.

When a production contract becomes stricter, update outdated test fixtures to
respect that contract.

Tests should exercise real production logic whenever practical.

Do not bypass important guards by mocking the guard itself to always succeed.

Prefer dependency substitution around environment-specific roots or resources
while allowing the actual validation logic to execute.


# Modification boundaries

Do not modify unrelated files.

Do not perform broad refactors unless explicitly required.

Do not reformat unrelated code.

Do not rename files or directories without need.

Do not delete user files unless explicitly authorized.

When assigned to one component, primarily modify only that component.

Repository-level files should be changed only when the change genuinely
applies repository-wide.


# Implementation economy

Prefer the smallest implementation that fully satisfies the requested behavior.

Before creating new code, inspect whether existing modules, interfaces, helpers,
training loops, data loaders, metrics, configuration, or CLI logic can be reused.

Prefer:

- reusing an existing module over copying it;
- extending a narrow existing interface over duplicating a subsystem;
- one clear implementation over multiple alternative implementations;
- short, focused functions over unnecessary abstraction layers;
- shallow control flow over deeply nested branching;
- configuration only for parameters that actually need to vary.

Do not add speculative extensibility.

Do not introduce factories, registries, adapters, wrapper layers, base classes,
feature flags, fallback implementations, or generalized frameworks unless they
are required by the current task or clearly reduce existing duplication.

When comparing model architectures, keep the surrounding data, training,
evaluation, and experiment pipeline shared whenever practical and isolate the
architectural difference to the smallest module possible.

Do not duplicate data loading, preprocessing, split handling, metrics,
evaluation, logging, or output-path logic merely to support another model.

Avoid unrelated cleanup and opportunistic refactoring.

A smaller diff is preferred when it provides the same verified behavior.


# Git rules

The repository root is the only Git repository unless explicitly documented.

Never run `git init` inside project subdirectories.

Do not:

- create nested Git repositories;
- use `git push --force`;
- rewrite remote history without explicit approval;
- reset unrelated user work;
- discard unknown working-tree changes;
- commit secrets;
- commit real datasets unless explicitly intended;
- commit large generated artifacts;
- commit temporary files.

Before editing, inspect repository state when relevant:

`git status`

Before committing, inspect:

`git status`

`git diff`

`git diff --check`

After staging, inspect:

`git diff --cached`

`git diff --cached --check`

Avoid:

`git add .`

and:

`git add -A`

when unrelated working-tree changes may exist.

Prefer staging explicit files.

Use clear, scoped commit messages.

Examples:

- `feat(component): add ...`
- `fix(component): correct ...`
- `test(component): add ...`
- `docs(repo): update ...`
- `refactor(component): ...`
- `exp(component): record ...`

Do not push unless requested or clearly authorized.

Keep commit subjects short and limited to the implemented functional change.

Do not mention agent orchestration, review process, test counts, planning
details, or implementation chatter in a commit message unless those are the
actual subject of the commit.

Do not create additional Git branches or worktrees unless explicitly requested
or required to isolate independent work.


# Working-tree safety

Assume existing uncommitted changes may belong to the user.

Do not:

- reset them;
- overwrite them;
- restore them;
- stash them;
- commit them;

unless they are explicitly part of the current task.

If an unexpected modified file appears, inspect its diff before deciding what
to do.

Do not infer that an untracked or modified file is safe to delete.


# Data safety

Treat real datasets and user data as read-only by default.

Do not:

- overwrite raw data;
- rename raw files;
- convert raw data in place;
- delete raw data;
- silently repair raw data;
- commit private data.

Preserve untouched raw data whenever possible.

Derived data should be stored separately from raw data.

Synthetic data may be used for:

- unit tests;
- parser fixtures;
- smoke tests;
- engineering validation.

Synthetic results must not be presented as real experimental results.


# Generated artifacts

Do not place large generated files directly in source directories.

Generated artifacts may include:

- checkpoints;
- logs;
- predictions;
- cached arrays;
- `.npy`;
- `.npz`;
- plots;
- temporary analysis files;
- training histories;
- build outputs;
- experiment manifests.

Store them under dedicated generated-output directories.

Examples:

- `results/generated/`
- `outputs/`
- `runs/`
- `logs/`
- `checkpoints/`

Generated-output paths should be enforced by directory purpose rather than by
blindly ignoring every file extension.

Do not add repository-wide rules such as:

`*.npy`

or:

`*.npz`

solely to hide poorly organized generated files.

Small reproducible test fixtures may legitimately use these formats and may
be tracked when appropriate.


# Output-directory safety

When a tool or experiment is required to write only under a generated-output
root, enforce the boundary using normalized filesystem paths.

Do not rely solely on string-prefix comparisons.

Resolve paths before checking containment.

Reject path traversal such as:

`generated/../other_directory`

when it escapes the authorized generated root.

Observation or logging functionality must never silently write into source,
data, test, or tracked-result directories.


# Long-running processes

Distinguish between ordinary subagent tasks and long-running external
processes.

Short or moderate tasks such as:

- implementation;
- tests;
- review;
- audits;

should normally be monitored by the main agent until completion.

Long-running tasks such as:

- multi-hour training;
- large simulations;
- long builds;
- external services;

should not depend on a subagent remaining alive unless the execution
environment guarantees process persistence.

When appropriate:

- launch a persistent OS process;
- record PID;
- redirect stdout/stderr;
- record run identity;
- monitor process state separately.

Do not assume a subagent remaining `RUNNING` is equivalent to the underlying
OS process remaining alive.


# Experiment integrity

For experimental or scientific workflows, distinguish:

- engineering smoke test;
- preflight;
- baseline experiment;
- tuning experiment;
- final evaluation.

A smoke test or preflight is not a formal result.

Do not aggregate incomplete runs into formal results.

Do not combine folds, trials, or outputs from different aborted runs unless the
protocol explicitly allows resumable execution.

Do not cherry-pick favorable runs.

Do not rerun poor results with different seeds unless the new run is explicitly
declared as a separate experiment.

If a formal run is aborted:

- preserve the aborted run when practical;
- mark it as aborted;
- exclude it from final aggregation;
- record the reason if known.


# Observation-only instrumentation

Logging, visualization, telemetry, and monitoring must be observation-only.

They must not change:

- model parameters;
- optimizer behavior;
- learning rate;
- random seed;
- data split;
- preprocessing;
- number of epochs;
- stopping criteria;
- checkpoint selection;
- final metric semantics.

Telemetry must be best-effort.

Failure to read optional information such as:

- GPU name;
- memory telemetry;
- environment metadata;

must not block computation.

Do not swallow genuine computation failures such as:

- CUDA OOM;
- forward errors;
- backward errors;
- optimizer failures;
- invalid data.


# Preflight policy

When a formal workflow needs a preflight mode, implement it explicitly.

Do not simulate preflight by silently overriding formal configuration.

For example, prefer:

`--preflight`

over:

`--run --epochs 1 --folds 1`

if the latter could accidentally mutate formal semantics.

A preflight should:

- use the same production data path when appropriate;
- use the same model path;
- use the same preprocessing;
- use the same split logic;
- use the same optimizer path;
- differ only in explicitly documented engineering limits.

Preflight outputs must be clearly marked:

- non-formal;
- not eligible for aggregation;
- not a reported result.


# Configuration

Important runtime parameters should be configuration-driven rather than
scattered as hard-coded constants.

When reproducing an external system, paper, benchmark, or reference
implementation, distinguish:

- source-specified settings;
- project assumptions;
- deliberate deviations;
- unresolved details.

Do not silently present an assumption as an externally specified fact.

Record important assumptions in appropriate project documentation.


# Reproducibility

When reproducibility matters, record enough information to reconstruct a run.

Useful metadata may include:

- Git commit;
- configuration snapshot;
- random seed;
- environment;
- device;
- dataset version;
- excluded samples;
- split policy;
- timestamps;
- run identifier.

Generated run metadata should not influence the experiment itself.


# Testing

Tests should verify behavior, not merely execution.

Where applicable, test:

- expected input/output shape;
- failure behavior;
- deterministic behavior;
- boundary conditions;
- malformed inputs;
- numerical finiteness;
- state isolation;
- data leakage;
- split correctness;
- serialization;
- generated-output safety;
- failure propagation;
- end-to-end smoke execution.

When fixing a bug:

1. reproduce it;
2. add or update the relevant test;
3. apply the minimal fix;
4. rerun the focused test;
5. rerun the affected suite;
6. run full validation when appropriate.


# Test evidence

A test command is considered successful only when the result is trustworthy.

Do not treat an empty output with an apparently successful exit code as strong
evidence if the tool normally prints a test summary.

Prefer obtaining both:

- exit code;
- visible test summary.

Examples:

`74 passed`

`2 failed`

If tool-wrapper behavior is suspicious, use an equivalent command in the
confirmed environment rather than inventing success.


# Validation sequence

For non-trivial changes, prefer:

implementation
→ focused tests
→ affected test group
→ preflight or smoke test
→ independent review
→ full test suite
→ Git diff inspection
→ commit

Not every task requires every stage, but correctness-sensitive work should not
skip validation merely for speed.


# Documentation

Documentation must reflect verified behavior.

Do not document planned behavior as completed behavior.

Do not claim:

- numerical reproduction;
- benchmark success;
- deployment readiness;
- data validation;
- protocol confirmation;

until those milestones have actually been completed.

Clearly distinguish:

- verified;
- partially verified;
- blocked;
- unresolved;
- not started.


# External-source fidelity

When implementing from a paper, specification, API, protocol, or reference
project, prioritize primary sources.

When a detail is uncertain:

1. inspect the primary source;
2. inspect official documentation;
3. inspect official code or data when available;
4. document the remaining uncertainty.

Do not silently replace ambiguity with common practice.

If an interpretation cannot be validated, report it as unresolved.


# Data leakage

Avoid train/validation/test leakage.

Do not use test data for:

- training;
- validation;
- hyperparameter selection;
- normalization statistics;

unless the explicit protocol requires it.

When group metadata exists, preserve it.

Do not describe sample-level evaluation as subject-independent,
writer-independent, user-independent, patient-independent, or equivalent
unless the split actually enforces that property.


# Failure policy

When a task fails, collect evidence before changing anything.

Record when relevant:

- command;
- exit code;
- traceback;
- affected stage;
- current process state;
- generated artifacts;
- last completed unit;
- configuration;
- Git state.

Do not immediately retry with changed parameters.

Do not hide failed attempts.

Do not classify an infrastructure interruption as a model defect without
evidence.


# Blocker policy

A blocker must be concrete.

Valid blockers include:

- reproducible test failure;
- missing required input;
- permission failure;
- incompatible interface;
- contradictory validated data;
- unsafe output behavior;
- undefined required protocol decision.

Speculative concerns are not blockers.

When blocked:

1. identify the exact blocker;
2. provide evidence;
3. define the smallest possible fix or decision scope;
4. stop only when user authorization is genuinely required.


# Dependency changes

Do not install or upgrade dependencies casually.

Before changing dependencies:

- confirm the dependency is required;
- inspect the current environment;
- prefer minimal additions;
- avoid broad upgrades.

Do not modify system-wide environments unless explicitly required.

Record dependency changes that affect reproducibility.


# Security

Never expose:

- credentials;
- API keys;
- tokens;
- private keys;
- passwords;
- private user data.

Do not commit secrets.

Do not bypass access restrictions.

Do not download from untrusted mirrors when an official source exists.

Do not execute destructive or suspicious commands without clear need and
authorization.


# Final acceptance

Before declaring a task complete, verify the relevant acceptance criteria.

For code changes, this often includes:

- intended behavior implemented;
- tests pass;
- reviewer blockers resolved;
- no unrelated modifications;
- generated files properly isolated;
- raw data unchanged;
- Git diff clean;
- documentation consistent with behavior.

Completion means the requested goal has been satisfied, not merely that code
was written.


# Final report

The main agent should provide a concise final report covering relevant items
such as:

- what changed;
- which files changed;
- tests executed;
- reviewer result;
- unresolved issues;
- generated outputs;
- commit hash;
- final Git status.

Do not overload the final report with internal implementation chatter that does
not help the user continue the project.
