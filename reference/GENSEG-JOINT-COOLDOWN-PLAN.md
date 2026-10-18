# Selected-GPU between-step sampling cooldown

Baseline2ebbef7266b58d30a242e129990967b0e5058535; branchcodex/genseg-reference initially clean. User data-PC joint3 run completed step1 then refused before step2 at GPU1 utilization81percent and1759/32607MiB memory5.39455percent. GD3 succeeded. This establishes utilization rejection; recent own-kernel sampling versus another workload is unresolved. Preserve failed joint_3steps_trial01; no successful joint3 or500+100 claim.

Latest user policy: local STATIC ONLY, no pytest, production imports/model/numerical/CUDA/resource probes. All shell require_escalated directly; newconda ast.parse allowed. Sole leaf implementation; no nested workers. Parent owns docs/Git/acceptance.

## Minimal change

- Introduce typed selected-utilization-only refusal after validating all relevant constraints. Between-step helper after synchronize/empty_cache retries ONLY that condition, max5 waits of1second (6 attempts total), then refuses. No random timing/seeding, model compute/sampling or updates between retries. KeyboardInterrupt/errors propagate. Initial/pre-model/CLI/preview guards no wait.
- Every attempt enforces selected physicalGPU index/UUID consistency, finite valid telemetry, VRAMpercent<80, freeVRAM>=12GiB, CPU/RAM<80 and freeRAM>=8GiB. Permanent issues refuse immediately even if GPUutil also high. OtherGPU workload unaffected. A transient-util retry is not a threshold exemption and must not attribute utilization to a process without evidence.
- Handle both telemetry samples if existing guard queries twice; secondary bind_gpu gpu_util may also be high. Reuse resource validation without global monkeypatching, no broad external module changes or special exceptions keyed on opaque string parsing. Preserve public guard defaults/shape, raw/split/oldweights, seed/rates/loss/BN/architecture/optimization/counters/checkpoint semantics.
- Log bounded waiting events promptly, success attempt count and waited seconds, structured diagnostics when persistentbusy fails. Preserve final fail metadata and incomplete output; no busy-spin/indefinitewait/auto-restart.
- Regression source cases (NOT RUN): busy->idle proceeds, permanentbusy exhausts exactbound, busy+CPU/RAM/memory/headroom/UUID/malformed refuses no wait, secondaryutil handled, initialguard immediate, no roles/steps/RNG changes. Update existing mocks if necessary without weakening actual checks.
- Writable leaf scope experiments/genseg_joint/runtime.py, tests/test_runtime.py, tests/test_multistep.py; narrowly necessary fixture compatibility in tests/test_pretrain.py only. Parent docs/source static inspection/hash+AST; separate staticreview exactsnapshot.

## Server retry

Retain complete GD3 and its independently hashed gan.pt. Pull fix, run joint3 in NEW genseg_joint_3steps_trial02 from the same GD3. No retraining GD is required when itscomplete receipt andhash match. Inspect step1/2/3 and complete=true before consideringGD500+joint100. Preservetrial01 failed evidence; no resume. Parent updatesSHORT-RUNBOOK to distinguish historical no-retry receipt from this boundedpolicy, records STATICPASS/runtimeNOTRUN, commits/pushes ordinary featurebranch.

Acceptance: parent/worker AST3files and diffcheck PASS; independent Level3 PASS STATIC on frozenhashes inCOOLDOWN-REVIEW.json. Runtime NOT RUN per user; servertrial02 must validate. Initial reviewer429 transportfailure replaced; no source/profileclaims from failedchannel.
