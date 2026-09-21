# Next

## Current focus

- Decide whether to run a separately authorized, isolated Lite-UPerNet
  engineering preflight or experiment. Implementation itself is complete.

## Next actions

1. Keep both baseline models and their checkpoints unchanged.
2. If experimentation is requested, define isolated output names and preserve
   the fixed patient-level five-fold split and matched single-output contract.
3. Treat any synthetic smoke/preflight evidence separately from formal OOF
   results.

## Blockers

- None for the Lite-UPerNet implementation.
- Whole-repository `python -m pytest -q` still encounters six pre-existing
  `non_teacher_student_files` import-collection errors outside this task diff.
