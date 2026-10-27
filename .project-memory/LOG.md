# Log

## 2026-10-27 - Recognize attached-name timepoints
- One-line regex fix accepts nameDx-DWI as well as name-Dx-DWI, retaining the existing trailing boundary and DWI-containing-label rules. Filename-only local audit increased candidate patient/timepoint keys from 98 to 131; actual conversion acceptance remains unverified.
- Validation: AST and simple static diff review only. No tests, real conversion, or raw-data edits; implementation delegated to a leaf worker.

## 2026-10-27 - User-selected standard label suffix priority
- Minimal naming-only fix: retain pure-DWI-first selection, then prefer normal .nii/.nii.gz over extra-dot suffixes within each type. Extra-dot files remain fallback; same-rank equality/conflict checks unchanged.
- Validation: AST and simple static diff review only, per user request. No tests, conversion or raw-data edits; leaf worker implemented.

## 2026-10-27 - User-selected pure DWI label priority
- Minimal label-only fix: pure DWI names outrank combined DWIADC per patient/timepoint, independent of path traversal order. Same-priority byte-equality and conflict checks retained; b selection and geometry unchanged.
- Validation: syntax/AST and simple static diff review only, as requested. No tests, actual conversion, or raw-data changes; implementation delegated to a leaf worker.

## 2026-10-27 - Minimal identical-label duplicate fix
- Added a full-byte equality check before duplicate-label rejection; identical files retain the first sorted candidate, differing files still fail. Updated the runbook; b selection and other conversion logic unchanged.
- Validation: AST parse and simple static diff review only, per user request; no tests, real conversion or raw-data modification. Leaf worker implemented; parent integrated.

## 2026-10-27 - CI-1 explicit b selection and label discovery
- Integrated only task commit `0df8d43` content: high/low/numeric b selection before deduplication, DWIADC matching, regular 3D/native-grid guards, fresh-output protection, successful manifest plus separate rejection report, and runbook.
- Validation: AST parse and Git diff checks; independent Level 3 PASS STATIC after fixing full 3D slice-position validation. Final runtime tests were waived by user; no real conversion or experiment was run.
- Provenance correction: supplied 95-entry manifest has 25 distinct patient fields. Existing splits are unchanged and patient-group separation remains unverified.
- Implementation/fix delegated to a leaf worker; main agent performed static acceptance, task-only integration and memory synchronization. No unrelated branch history or remote publication.

## 2026-10-21 - Initialize canonical project-wide memory
- Added STATUS/GOALS/NEXT/LOG on master, distinguishing integrated history from current feature-branch inventory and historical server-returned development evidence.
- Evidence: local Git state and branch pointers; detailed experiment provenance remains in codex/genseg-reference project memory. No fresh server/runtime/full-suite verification.
- Main agent documentation only; production source, raw data, splits, checkpoints and other working trees unchanged. External publication remains subject to the reported automatic-review block.
