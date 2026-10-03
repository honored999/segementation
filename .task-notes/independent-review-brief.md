# Independent final review brief

Read-only target: E:/study/研一/work14-图像分割/segementation/.worktrees/upernet-no-stage7
Baseline: 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89
Branch: codex/upernet-no-stage7
Final source/test/doc digests will be supplied after worker completion. Inspect untracked implementation files as well as git diff; do not assume untracked files appear in ordinary diff.

Review architecture invariants: no stage7 module,parameter or forward; original stage0..6 settings preserved; UPerNet inputs fixed(1,3,5,6),native shapes256,64,16,8 for512 input; unchanged PPM(1,2,4),FPN128 and training MRO. Verify original plans/configuration immutability, supported-plan rejection, official external discovery plus train/inference build signature, strict synthetic state reconstruction, geometry/accounting consistency and original baseline unaffected. Do not request changes unrelated to this approved ablation.

Review evidence against exact final snapshot: trustworthy small CPU tests, inherited TopK10/early-stop contracts, affected baseline regressions; no real training/data/checkpoints or performance claim. Documentation must clearly describe an external custom Trainer on the official framework; independent output identity and fresh initial training; --c only own Trainer; final checkpoint validation to match current comparison policy. Existing fixed patient splits and medical source data remain untouched. No push,commit or canonical master integration in scope.

Reviewer is an independent leaf context, cannot spawn agents or edit files. Every shell operation must use require_escalated (user explicitly forbids sandbox attempts). If any runtime test is necessary, fresh CPU/RAM/GPU/VRAM preflight and80percent resource guard apply; prefer existing exact-snapshot evidence over redundant test runs. End with PASS or BLOCKING with evidence and minimal fix, optional NON-BLOCKING; specify exact snapshot,digests,scope and scientific/server limitations.
