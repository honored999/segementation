# Independent Level 3 review — BLOCKING

Date: 2026-10-03 (Asia/Hong_Kong). Role: manual independent read-only Level3 reviewer. Requested profile: LunaMax; observed actual model/reasoning profile: unknown. This reviewer did not implement the change, created no agents/tasks, and did not modify production source, tests, plans, memory, dependencies or protected checkouts.

## Exact reviewed snapshot

- Worktree: `E:\study\研一\work14-图像分割\segementation-upernet-configurable-stages`.
- Branch: `codex/upernet-configurable-stages`.
- Base / HEAD: `8a41ce3b0d4e2c980c9e9063a60b055b0a841d89`.
- Uncommitted / unmerged; initial dirty scope is exactly the supplied implementation and branch-local memory/task notes.
- Manifest: `.task-notes/upernet-selected-stages/snapshot_manifest.json`.
- Manifest SHA256: `8655a1725ce8ae8da79d3b7d581525a07d70034ff6262c57ca6971e779eb1878`. The user separately supplied this same hash during review; the external handoff anchor is matched.
- All 23 manifest files have matching SHA256 and byte sizes. All 24 ignored files match their manifest hashes and inventory; they are only generated pycache/pytest cache, with no additional ignored source/test files. Original 23-file hashes and manifest were rechecked at completion and have no drift.
- All 23 Python source hashes in each of focused-final, baseline-final and affected-final also match current files. Their reported 55 / 25 / 105 passed (exit 0) are implementer evidence, not independent test counts.
- Three reviewer validation JSONs and this report are new supplemental evidence, excluded from the original manifest. The original manifest and earlier evidence were not overwritten.
- Installed sources inspected directly: nnunetv2 2.8.1, PyTorch 2.11.0+cu126, dynamic-network-architectures 0.4.4, using `D:\Anaconda\envs\newconda\python.exe`. No installation/upgrade.

## BLOCKING — [P1] A legal selection excluding the final stage fails default DDP on the second iteration

Locations:

- `nnunet_ext_trainers/nnUNetTrainerUPerNetSelectedStagesTopK10EarlyStopping.py:149-155`: constructs the complete encoder with a selection that may end before its final stage.
- `nnunet_ext_trainers/nnUNetTrainerMixins.py:543-546`, `_PlainConvUNetUPerNet.forward`: executes all encoder stages but sends only selected outputs into the decoder/loss.
- New Trainer `initialize`, lines 157-162, delegates to the inherited official initialization without handling the now-unused trailing-stage parameters.
- Actual installed official `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:249-253`: DDP wrapper is constructed as `DDP(self.network, device_ids=[self.local_rank])`.
- Installed PyTorch `torch/nn/parallel/distributed.py:665`: `find_unused_parameters=False` by default.

Evidence: an independently executed, one-process CPU Gloo DDP synthetic probe compared four-stage models with identical geometry and seed. No optimizer step, training epoch, real data, CUDA allocation or multi-GPU execution was involved.

```text
SELECTION [0, 3] UNUSED []
SECOND_ITERATION_OK [0, 3]
SELECTION [0, 2] UNUSED ["encoder.stages.3.0.convs.0.conv.weight", "encoder.stages.3.0.convs.0.norm.weight", "encoder.stages.3.0.convs.0.norm.bias"]
SECOND_ITERATION_ERROR [0, 2] Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.
Parameter indices which did not receive grad for rank 0: 9 10 11
SYNTHETIC_CPU_DDP_REGRESSION_REPRODUCED
```

The probe exited 0 because it successfully reproduced and checked the expected failure; this does not mean the affected production behavior passed. Selecting a deepest stage earlier than the encoder end disconnects all later-stage parameters from the loss. This also applies structurally to the required eight-stage choices `[1,3,5,6]` and `[2,3,4,5]`. It is a deterministic default-DDP correctness issue, not a deferred memory/speed measurement.

Violated contract: legal explicit stage choices must work with the inherited official Trainer while retaining the complete encoder and original training semantics. Accepting these selections without addressing the official reducer contract creates a new failure absent from the original automatic selection, which includes the last stage.

Smallest fix scope: handle the unused trailing parameters only in the new Trainer's DDP construction/integration (for example, enable unused-parameter detection for this case through an appropriately scoped supported path). Keep the complete encoder and its forward execution, single-GPU gradients/optimizer semantics, loss and old Trainers unchanged. Do not delete stages or add dummy zero gradients just to satisfy the reducer. Add a tiny two-iteration CPU DDP regression with a last-stage control; check the new path's compile/DDP ordering and checkpoint identity guards. No broad training-loop copy or third-party modification is required. This reviewer did not apply a fix or dispatch a fixer.

Reproducer core (run after a fresh resource preflight, in the reviewed worktree using newconda; it intentionally catches and verifies the failure):

```python
import importlib.util, sys, tempfile
from datetime import timedelta
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist

torch.set_num_threads(1)
sys.path.insert(0, str(Path.cwd() / 'nnunet_ext_trainers'))
spec = importlib.util.spec_from_file_location(
    'review_fixtures', Path.cwd() / 'nnunet_ext_trainers/tests/test_upernet_selected_stages.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
with tempfile.TemporaryDirectory(prefix='upernet-review-cpu-ddp-') as td:
    store = dist.FileStore(str(Path(td) / 'rendezvous'), 1)
    dist.init_process_group('gloo', store=store, rank=0, world_size=1,
                            timeout=timedelta(seconds=20))
    try:
        for selection in ([0,3], [0,2]):
            torch.manual_seed(9)
            model = f.build(selection)
            wrapped = nn.parallel.DistributedDataParallel(model)
            wrapped(torch.randn(1,1,17,17)).square().mean().backward()
            print(selection, [name for name, p in model.named_parameters()
                              if p.requires_grad and p.grad is None])
            try:
                wrapped(torch.randn(1,1,17,17)).square().mean().backward()
                assert selection == [0,3]
            except RuntimeError as error:
                assert selection == [0,2]
                assert 'Expected to have finished reduction' in str(error)
                print(error)
            del wrapped, model
    finally:
        dist.destroy_process_group()
        del store
```

## Other reviewed contracts — no additional blocker found

1. Selection reads resolved `ConfigurationManager.configuration`; inheritance comes from actual `PlansManager.get_configuration` / `_internal_resolve_configuration_inheritance`. Only missing key defaults. Bool/float/string/null, length/order/duplicate/range and either-axis scale failures reject; architecture kwargs key rejects before official allocation. Existing Trainers still choose the original four levels.
2. N-1 lateral/refinement branches, deepest PPM(1,2,4), FPN128, N*128 fusion, raw output, exact-size bilinear/align_corners=False and actual BCHW/channel/batch/strict spatial checks are retained. Full encoder remains registered/executed. Frozen git-show baseline is valid: it loads the original module from 8a41ce3, resets the same RNG seed per build, compares state tensor keys/values and bitwise logits. Independent frozen regression passed.
3. Network identity is versioned `_extra_state`, checked by a root pre-hook with exact primitive types before parameter copying. Installed PyTorch module.py:2391-2400 precedes copy_ at 2496 and recursive child loading at 2598-2606. New Trainer checks before inherited loader changes my_init_kwargs/logger/optimizer. Official predictor loads first-fold state at line 131 and later fold states at 517/519, covering both normal and OptimizedModule unwrap paths. Independent mismatch, resume, predictor and eager wrapper tests passed. DDP prefix handling was read against inherited loader; actual GPU/DDP checkpoint execution remains unverified.
4. New-wrapper complete encoder accounting follows actual Conv2d geometry; shared decoder accounts actual selected stages. Original four-layer wrapper intentionally retains the library's floor-based odd accounting to preserve its contract. This localizes the new-model fix as required. Independent odd/anisotropic network hooks and tiny N=2/4/8 decoder hooks passed. Implementer's complete eight-stage 129x129 fixture is hash-bound existing evidence; it was inspected, not rerun by this reviewer.
5. Plans CLI is UTF-8 JSON only, refuses duplicate keys/invalid configuration/inheritance/misplaced indices, resolves paths, requires distinct/new correctly named output, serializes with allow_nan=False before exclusive x creation, and preserves source bytes/configuration/inheritance/data_identifier. Independent API/CLI round-trip and refusal checks passed. No image/checkpoint/preprocessing interfaces are invoked by the CLI.
6. Diff leaves loss/early stop/optimizer/scheduler/splits/augmentation/batch/patch/foreground/inference policies and existing Trainer source unchanged. Four-layer compatibility passed. New checkpoint protocol is deliberately exclusive to the new Trainer. README's 8x final concat ratio is correct structural arithmetic, explicitly not a whole-model memory or performance measurement. Branch-local memory truthfully says unmerged/PENDING; canonical master memory paths do not exist in Git and were not modified.

NON-BLOCKING: none separately raised. Server/CUDA/multi-rank DDP/whole-model feasibility/performance are not acceptance passes inferred from local evidence.

## Independent validation and resource evidence

All pytest commands invoked the existing recorder with a unique label, `-p no:cacheprovider`, PYTHONDONTWRITEBYTECODE=1, single-thread CPU settings and fresh resource sampling. Full argv/stdout/stderr/source hashes are stored in the corresponding new JSON; no existing evidence was overwritten.

| Independent command/label | Result | CPU % | RAM % | GPU0 % | VRAM0 % |
|---|---|---:|---:|---:|---:|
| run_validation.py review-level3-20261003-identity-a: test_identity_rejected_before_parameters_optimizer_and_logger | 4 passed, 4 warnings in 6.45s; exit 0 | 10.22 | 60.76 | 24 | 21.13 |
| run_validation.py review-level3-20261003-runtime-a: predictor, actual CPU initialize/resume, frozen baseline, eager wrapper/nested identity | 4 passed, 6 warnings in 9.15s; exit 0 | 13.69 | 60.43 | 23 | 20.74 |
| run_validation.py review-level3-20261003-geometry-plans-a: odd/anisotropic hooks, N-level tiny decoder, plans copy/CLI/refusal/ambiguity | 17 passed, 4 warnings in 21.18s; exit 0 | 12.62 | 61.74 | 26 | 22.27 |
| Inline python -B -: original Unicode FileStore URI attempt | infrastructure failure before model validation; exit 1 | 16.42 | 61.39 | 36 | 21.55 |
| Inline python -B -: HashStore attempt | infrastructure failure: HashStore absent; exit 1 | 18.17 | 62.90 | 14 | 25.59 |
| Inline python -B -: ASCII FileStore, CPU Gloo, rank=0/world_size=1 | regression reproduced; control passed; exit 0 | 10.50 | 63.87 | 31 | 22.93 |

Every preflight decision was NORMAL (<80%); no unrelated process was interrupted. Planned scope was tiny single-thread CPU models, not full-size training. No numeric projected/peak utilization is claimed.

Exact pytest node argv:

```text
D:\Anaconda\envs\newconda\python.exe -B .task-notes/upernet-selected-stages/run_validation.py review-level3-20261003-identity-a
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_identity_rejected_before_parameters_optimizer_and_logger -p no:cacheprovider

D:\Anaconda\envs\newconda\python.exe -B .task-notes/upernet-selected-stages/run_validation.py review-level3-20261003-runtime-a
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_real_predictor_rebuild_and_multifold_weights
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_actual_cpu_trainer_initialize_resume_and_validation_rebuild
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_four_level_frozen_baseline_keys_shapes_values_and_count
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_compiled_wrapper_and_nested_root_identity_guards -p no:cacheprovider

D:\Anaconda\envs\newconda\python.exe -B .task-notes/upernet-selected-stages/run_validation.py review-level3-20261003-geometry-plans-a
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_tiny_forward_backward_full_encoder_and_accounting
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_n_level_decoder_tiny_backward_accounting
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_plan_copy_roundtrip_and_cli
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_plan_copy_refuses_and_preserves_source
  nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_source_plans_ambiguity_rejected -p no:cacheprovider
```

Three pytest commands account for 25 independently passing parameterized cases. No root/standalone/full external suite was rerun. The default-DDP bug is not covered by those 25 cases or the implementer's 105-case suite.

## Protection / final state

- Protected root `segementation`: feat/h2former-stroke @ d717a7a82d30a1d23fbb2c3aab4425c1708806fa; status clean.
- Read-only baseline `segementation-upernet-es-topk10`: codex/nnunet-upernet-es-topk10 @ 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89; status clean.
- Read-only reference `.worktrees/upernet-no-stage7`: codex/upernet-no-stage7 @ f1b9ca87e28bda0c8941b639d87349b7d9012338; status clean.
- Reviewed source/tests/memory/earlier evidence unchanged. Reviewer additions: this one Markdown report and three uniquely named validation JSONs. Synthetic pytest fixtures remain in the existing sibling task-test-temp root. Successful DDP rendezvous used a unique E:\CodexScratch\temp directory and the context manager removed that probe-owned directory; no user data cleanup occurred.
- git diff --check: exit 0. Existing five tracked modifications and existing untracked implementation/task artifacts remain; supplemental review files are untracked. No commit/push/merge/reset/stash/clean, no agent creation, no direct source/test implementation, no canonical or branch-local memory edit by reviewer.
- Not run: server, CUDA forward/backward/compile, multi-rank or multi-GPU DDP, actual official CUDA/DDP Trainer training/checkpoint operations, real patient data/checkpoints/training/validation/prediction/preprocessing, full-size eight-level feasibility, speed/peak memory, formal five-fold/full-volume OOF evaluation.

HANDOFF
- status: COMPLETED (independent review completed; implementation acceptance remains BLOCKING)
- role: manual independent read-only Level3 reviewer
- requested model/profile: LunaMax
- observed model/profile/reasoning: unknown
- reviewed absolute worktree: E:\study\研一\work14-图像分割\segementation-upernet-configurable-stages
- branch/base/HEAD: codex/upernet-configurable-stages / 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89
- manifest SHA256: 8655a1725ce8ae8da79d3b7d581525a07d70034ff6262c57ca6971e779eb1878 (separate user anchor matched; original 23 files unchanged)
- PASS: selection/configuration/decoder/accounting/plans safety/identity and tested CPU predictor/rebuild/baseline contracts; 25 narrow pytest cases, exit 0
- BLOCKING: P1 unused trailing encoder parameters cause default DDP second-iteration failure; independent single-rank CPU Gloo reproducer exit 0 confirms this failure
- NON-BLOCKING: none separately raised
- resource preflight: six fresh samples, all NORMAL; failed Store setup attempts retained as infrastructure evidence, not model failures
- unverified: server/CUDA/multi-rank-DDP/performance/real-data/formal evaluation as enumerated above
- protection/file changes: protected root/baseline/NoStage7 clean; original source snapshot unchanged; only report + three review evidence JSONs added, isolated synthetic temp fixtures generated
- direct source/test implementation: no
- project memory: not modified; canonical master paths absent in Git; branch-local PENDING records preserved
- Git: no new commit; target remains uncommitted/unmerged, original dirty scope plus supplemental review artifacts; diff --check exit 0
- next step: user manually dispatches a scoped fixer for the new Trainer DDP unused-parameter handling and two-iteration regression, then obtains hash-bound focused revalidation and independent re-review; no merge/overall acceptance yet