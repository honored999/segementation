# HANDOFF — Dataset509 Q

COMPLETED: code + local CPU engineering checks only. No server work or real run.

- Branch: `codex/dataset509-quality`; base/HEAD `5472832bcd077c7e98244a8f7016b7c6ad997428`.
- Worktree: `E:/study/研一/work14-图像分割/segementation/.worktrees/dataset509-quality`.
- All additions are uncommitted/untracked; no commit/push. Root user changes and canonical memory untouched. Read-only LesionGen dependency HEAD verified `c498118e6e3b3e4bb7a1c2380bb777c13f601e04`; no dependency diff.
- Files: `experiments/dataset509_quality/{quality.py,data.py,loss.py,trainer.py,trainer514.py,run.py,api_contract.py,api_contract.json,focused_checks.py,baseline.json,q.json,quality_filter.json,README.md}`; `SOURCE_HASHES.json` binds deliverables for deployment.

## Actual interfaces

The copied `trainer514.py` is byte-identical to the existing 514 adapter.
Baseline factory returns that class directly. Q/filter subclass it; QualityDataset
extends the pinned AugmentedDataset. SourceLoader overrides only get_bbox and
batch metadata attachment, keeping official crop/DA logic. SourceLoss delegates
validation to original Dice+TopK10. Only Q uses separately normalized source
TopK10 and quality-weighted per-plane Dice. Filter removes low-weight appended
planes and keeps the original mixed loss. Offline sidecar uses native pool
image/mask ring contrast, training100 real references, and content/split pins.
Initial lambda=1, weight floor=.1, filter threshold=.5; these are hypotheses.
C is not implemented.

## Verified locally (newconda; 2026-11-15)

nnU-Net 2.8.1, Torch 2.11.0+cu126, batchgenerators .25.3, v2 .3.4.
`focused_checks.py --lesiongen-repo E:/study/研一/work14-图像分割/CG-SAMR-reference/LesionGen` exited 0:
- native contrast boundaries, empty/full mask, image border and nonfinite input;
- independent source TopK budgets, actual quality gradient attenuation, finite
  connected gradients, zero weights, single source, ignore and deep supervision;
- original baseline class/method identity and exact constructor signature;
- actual official loader, random/foreground sampling, within-case plane weights,
  full augmentations and RNG equality, padding, ordinary filter/all-filtered cases;
- one real spawned augmentation worker through batch metadata and DS backward;
- actual Trainer CPU construction and Q/filter loss selection (no model initialize).

AST/JSON, exact baseline-adapter bytes, all authored files whitespace/conflict
scan and git diff --check passed. The tracked diff is empty because files are
new; the explicit authored-file scan covers them. An initially incomplete
constructor fixture lacked architecture; fixed fixture and rerun passed.
No dependency installation, full suite, model training, or agent dispatch.

## Unverified / main-agent next

No real pool sidecar/distribution was built, no signal validity demonstrated.
No server API inspection, 508 real weight load, GPU preflight, deployment,
training or original-space evaluation. Local API fingerprints fail closed on
server drift; rerun focused checks in existing nnunet5090 and inspect drift
before modifying that contract. Patient independence still requires the existing
patient-split evidence; string-disjoint cases alone are insufficient.

Fill copied mode config paths/hashes; example stopping/iteration values are
inherited defaults, not verified historical queue policy. Establish a common
update budget and matched baseline; old 81.518% is not a matched-budget control.
Run offline quality.py on existing native package + prepared pool, inspect its
anonymous training-only audit and limitations. Then verify server resources and
choose a currently available GPU UUID. Use NEW output roots for each invocation:

```
python run.py --config q.local.json --stage preflight --gpu-uuid GPU-<uuid> --output <new-preflight-root>
python run.py --config q.local.json --stage train --gpu-uuid GPU-<uuid> --output <new-training-root>
```

Analogous baseline/filter commands and sidecar/check commands are in README.
Every stage starts fresh with 508 weights only, never optimizer/early-stop resume.
Main agent owns real preflight acceptance, formal training and complete original
28-volume evaluation. No notifications or server task scheduling were added.
