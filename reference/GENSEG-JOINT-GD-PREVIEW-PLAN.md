# Completed G/D checkpoint preview

Date: 2026-10-18
Branch: codex/genseg-reference
Starting commit: 161985670b413bf475cdb0061ab8ce2afb014870

## User need and design

The user completed GD500 and joint100; final joint images remain nearly constant gray. They request an executable method to inspect the GD500 generator before joint optimization, using the same selected slices as the existing joint preview. Existing preview accepts joint.pt only.

Chosen minimal design: extend existing preview with explicit --phase {joint,gan_pretrain}, default joint. A separate renderer/standalone weight-loading script would duplicate audited cohort/display/provenance logic. Do not add either. No model architecture, BN, optimizer, loss, budget, sampling, seed, target or checkpoint-writing changes.

## Acceptance requirements

- Existing joint preview stays default and compatible; wrong-phase checkpoints are rejected explicitly.
- GAN preview requires top-level completed phase gan_pretrain receipt and exact explicit gan.pt SHA256; validate pinned source, fixed inner61/internal15/external19 identities, target provenance, width64/downs8 model, frozen-BN policy and saved inner61 normalization before patient arrays/G construction.
- Reuse pretrain.audit_pretrained/validate_protocol and existing preview staging wherever feasible. Do not call load_pretrained with constructed D, because preview must construct G only.
- Deserialize CPU with weights_only=True; streamed checkpoint hash before and after loading from the same handle; payload schema/receipt agreement checked. G state and both architecture tensors validated completely before live restoration, keeping all alpha aliases bound correctly. D tensors may be read as CPU payload but D/S models and optimizers must never be constructed.
- Reuse deterministic sorted first4 inner61 patients, first lesion plus up to2 valid empty slices, native full-plane padding/cropping, saved scale, common display range and ignore masking. No inner15/external19 image/label arrays or scale refit.
- Protect raw/source/target/training run and existing outputs using resolved paths. Use a fresh output directory; final provenance/input/checkpoint/PNG hashes and completion manifest only after success. Preserve zero-valid-plane failure behavior.
- Record input checkpoint phase/file and truthful CPU payload description in preview.json. Images remain qualitative engineering PNGs, not generated training data or efficacy evidence.

## Ownership and checks

Leaf implementer owns experiments/genseg_joint/cli.py, preview.py, and relevant preview/CLI test source only. No nested agents, no Git commits/push, no project-memory edits.
Main owns this plan, runbook/review receipt, branch memory, acceptance and Git.
Independent read-only reviewer inspects exact final source/test hashes for Level3 checkpoint/provenance/data safety. Worker/reviewer inherit platform model (explicit model override unavailable without human request); reasoning xhigh requested, no claim of LunaMax identity.

User constraint: LOCAL STATIC ONLY. No pytest, production imports, model/torch loading, CUDA, numerical tests or resource probes. Only text, git, hashes and newconda stdlib AST parsing. Test sources are added for future execution on a suitable machine, not claimed passing locally. All shell commands require_escalated directly.

## Delivery

1. Leaf minimal implementation and test source; HANDOFF with exact changed files, checks, hashes, deferred runtime limitations.
2. Parent AST/diff/targeted interface inspection, static independent Level3 review and blocker fixes.
3. Document server commands using gan.pt SHA 33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6 and existing pretrain path C:\lijialin\HD-BET_outputs\genseg_gd_pretrain_500steps_trial01, fresh preview directory.
4. Commit/push intended files on existing branch per standing user authorization; data computer git pull --ff-only and GPU1 preview. No local runtime validation or automatic server execution.
