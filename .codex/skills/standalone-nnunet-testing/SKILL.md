---
name: standalone-nnunet-testing
description: Test and validate standalone_nnunet2d safely and efficiently. Use for pytest in standalone_nnunet2d, checkpoint metadata/compatibility tests, inference or model-loading tests, synthetic fixtures, validation-level selection, large temporary checkpoint handling, Windows pytest temp roots, or deciding between focused, affected, and full suites.
---

# standalone nnU-Net testing

## Validation principle

Choose the lightest validation that gives trustworthy evidence.

Prefer:
`focused test -> affected group -> broader/full suite when risk justifies it`

Do not rerun an expensive full suite merely for ceremony when unchanged high-risk
code is already covered by trustworthy evidence.

## Checkpoint fixture policy

The standalone model may produce checkpoints hundreds of MiB in size.

For metadata, provenance, alignment status, tamper detection, rejection, CLI
validation, and other pre-model-load tests:
- use the smallest synthetic checkpoint satisfying the reader/format contract;
- when rejection should occur before model loading, use a model-loading sentinel
  when practical to verify `_load_model` is not reached.

Use full `PlainConvUNet2D` checkpoints only when the test genuinely exercises:
- real model loading;
- full inference;
- state-dict compatibility;
- checkpoint conversion;
- architecture/runtime compatibility requiring real weights.

Do not weaken real integration coverage merely to reduce disk use.

## Temporary-file safety

Avoid unnecessary large temporary `.pt`, `.pth`, arrays, caches, and copied datasets.

Temporary test artifacts are disposable and are not experiment results.

Use a controlled disposable pytest root when practical.

On the known Windows workstation, when `D:` is available, prefer:
`D:\codex-pytest-temp\stroke-lesion-segmentation`

Reuse/clear a deterministic disposable root rather than creating many persistent
roots with names such as `standalone-full-final` or `inference-related-final`.

If a non-system root is unavailable, use the environment's ordinary temporary
mechanism or report a material constraint; do not silently redirect a known large
suite to a persistent `C:` directory.

Do not place large temporary artifacts under source, repository, user-profile, or
tool-state directories.

## Test quality

Tests should verify behavior, not merely execution.

Where applicable test:
- expected shapes/channels;
- failure behavior;
- boundary/malformed input;
- deterministic behavior;
- numerical finiteness;
- state isolation;
- leakage guards;
- split correctness;
- serialization;
- output-path safety;
- failure propagation;
- end-to-end smoke behavior.

Do not mock away the production guard being tested.

Prefer dependency substitution around environment-specific roots while allowing
the real validation logic to execute.

## Bug-fix TDD

For a defect:
1. reproduce it with a focused test;
2. observe RED;
3. apply the minimal production fix;
4. observe GREEN;
5. run the directly affected group;
6. broaden validation only when shared/high-risk code changed.

A RED caused by a genuinely missing module/API is acceptable evidence when it
proves the new test detects absent functionality.

## Test evidence

Prefer both:
- successful exit code;
- visible test summary such as `74 passed`.

Do not treat empty/suspicious wrapper output as strong evidence merely because an
exit code appears successful.

## Validation levels

### Level 1
Localized low-risk change:
`focused tests -> status/changed-file check -> targeted acceptance`

### Level 2
Moderate change:
`focused tests -> affected group -> targeted diff inspection -> integration validation`

### Level 3
High-risk correctness-sensitive change:
`focused tests -> affected group -> independent read-only review -> fixer if needed -> revalidation -> full validation when appropriate`

Use `level3-review` for the review itself.

## Real-data boundary

Synthetic tests are engineering validation only.

Do not claim real Dataset501/502/CI-1 validation, clinical performance, or server
compatibility unless those were actually exercised.

Do not write to real nnU-Net raw/preprocessed/results directories during
synthetic validation.
