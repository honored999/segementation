# Fixed training slice G/D diagnostic

Date: 2026-10-18
Branch: codex/genseg-reference
Base: b9df51a192d59d4d4aeafdddb55b6fbe87eb6de8; starting tree clean.

## Authorized question and minimal design

The user asks how to check whether the generator can learn one fixed training slice after GD500 and joint100 previews both remain gray. Add an explicit diagnose-single-slice command. Initialize G/D from the completed same-inner61 GD500 checkpoint (SHA 33dc65d16c4d03fafa1d2d9884d61e11f298a9621751c126b2b72c1bc08957a6). Repeat one inner61 lesion slice under the SAME frozen-BN, frozen-alpha, stateless-SGD G/D rates and L1 policy. No joint/architecture/S training and no BN/optimizer switch or new random data augmentation. Reuse existing PretrainEngine rather than adding another training update implementation.

This is a fixed-training-sample diagnostic, not evidence of generalization or a completed GAN phase. Do not save reusable checkpoint weights or use save_pretrain. Completion is diagnostic-only. A short run failing to reconstruct does not prove lack of model capacity or identify BN as the cause.

## Inputs and outputs

- Common source/raw-root/baseline-run/targets/targets-sha256 plus paired required --pretrained-run/--pretrained-sha256, --gpu and explicit positive --steps. Fixed seed20261018. No user-adjustable rates/BN/loss knobs in this diagnostic.
- Before model/patient arrays, reuse audit_gan and existing completed-pretrain validation; strict source, target, fixed61/15/19, model, saved scaling, phase, schema and hash. Reuse load_pretrained to restore BOTH G/D safely; no S/optimizers or RNG restoration.
- Choose sorted(inner61)[0] deterministically, then that patient's first valid lesion plane using existing select_planes. Fail clearly if none; do not silently switch cases/cohorts. Check role membership before array read. Record selection. Never read internal15/external19 image-label arrays or refit scale.
- Use exactly existing full native plane/pad-to256/minmax/ignore normalization. Hold the same tensors for all updates; no sampling or flips.
- Resource guards before patient images/models, immediately before models, before each update/probe; expensive identity audits precede the final resource sample so the sample remains adjacent to computation; same selected UUID, thresholds/headroom, allocator ceiling, synchronize/unused cache release and bounded between-completed-step util-only guard. No other GPU interference, threshold changes, unbounded retries or OOM swallowing.
- Before and after updates, record valid-pixel target/generated mean/std/min/max and normalized L1, generated spatial variation; report foreground/background separately when available. Render before.png/after.png with existing four-panel renderer, saved scale and native H/W. Same display range, no per-image normalization.
- Add diagnostic-only pre/post gradient observations using production FunctionalModel/gan_generator_loss, not a second update implementation. Per-parameter gradient norms and lightweight Conv2d/ConvTranspose2d/BatchNorm activation summaries may be captured by temporary forward hooks; hooks return None, keep detached scalar summaries only, remove in finally. No changes to model/BN buffers/optimizer/RNG due to observations; no backward hooks that replace gradients, no retained activation graphs. Check finiteness and propagate genuine errors.
- Always fresh protected output isolated from source/targets/pretrain/old outputs; started.json, steps.jsonl and guard_events.jsonl, before/after PNG, diagnostic.json completion only after final source/input/pretrain/implementation/PNG audit. failed.json on exceptions, no success marker on failure. Do not emit gan.pt/joint.pt or a valid pretrain/joint completion receipt.
- Record exact initialization hash, saved protocol/scaling, fixed slice selection, counts, resources, source/implementation identity and explicitly exploratory classification. Diagnostics must not claim automatic learned/not-learned PASS thresholds.

## Ownership and verification

One leaf implementer owns experiments/genseg_joint/diagnostic.py (new), cli.py and tests/test_single_slice.py (new). May minimally adapt related CLI test fixtures if required, otherwise do not touch existing code. Do not modify core.py/runtime.py/pretrain.py/models.py/data.py/preview.py or upstream. Parent owns plan/runbook/review receipt, branch memory and Git. No nested agents or worker commits.

User LOCAL STATIC ONLY overrides runtime testing workflows: no pytest, production imports, torch/models/CUDA, numerical tests or resource probes locally. Use only text/git/hash and newconda -I -S stdlib ast.parse. Write focused future tests for role gating, fixed sample, before/after stats/probe purity, safe initialization, resource refusal, protected outputs/final audits, completion semantics, no-checkpoint output. Clearly report tests NOT RUN. All shell commands directly require_escalated.

Separate-context read-only Level3 reviewer must inspect final frozen source/test hashes, especially scientific/provenance/checkpoint/observation semantics and failure paths. Inherited platform model, xhigh requested; no unverifiable LunaMax claim. Commit/push only after parent static checks and review; data computer git pull --ff-only. First server3-step transport/probe run, then fresh100-step diagnostic from original completedGD500, retaining failed/old outputs. No local/server training started automatically.
