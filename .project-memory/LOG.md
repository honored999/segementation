# Log

## 2026-10-27 - CI-1 explicit b selection and label discovery
- Integrated only task commit `0df8d43` content: high/low/numeric b selection before deduplication, DWIADC matching, regular 3D/native-grid guards, fresh-output protection, successful manifest plus separate rejection report, and runbook.
- Validation: AST parse and Git diff checks; independent Level 3 PASS STATIC after fixing full 3D slice-position validation. Final runtime tests were waived by user; no real conversion or experiment was run.
- Provenance correction: supplied 95-entry manifest has 25 distinct patient fields. Existing splits are unchanged and patient-group separation remains unverified.
- Implementation/fix delegated to a leaf worker; main agent performed static acceptance, task-only integration and memory synchronization. No unrelated branch history or remote publication.

## 2026-10-21 - Initialize canonical project-wide memory
- Added STATUS/GOALS/NEXT/LOG on master, distinguishing integrated history from current feature-branch inventory and historical server-returned development evidence.
- Evidence: local Git state and branch pointers; detailed experiment provenance remains in codex/genseg-reference project memory. No fresh server/runtime/full-suite verification.
- Main agent documentation only; production source, raw data, splits, checkpoints and other working trees unchanged. External publication remains subject to the reported automatic-review block.
