# GenSeg joint single-GPU resource guard correction

Baseline6bb7f1d. User observed GPU1 idle while GPU0 was busy: selecting1 passed bind_gpu but failed the later loop requiring ALL physical GPUs below80percent. Selecting0 correctly failed selected-GPU utilization. Both errors occurred in CLI's initial guard before neural-model construction or training/output creation.

This patch limits the TRAINING runtime guard to the explicitly selected physical GPU, bound by UUID. CPU/system RAM, selected GPU compute utilization (<80percent), selected VRAM utilization (<80percent), free VRAM/RAM minimums and allocator ceiling remain protected. Other GPU tasks are not queried as blockers or modified. An index/UUID mismatch or invalid selected telemetry fails closed; error messages identify the card and utilization/VRAM value. Existing command --gpu1 remains valid once this runtime.py reaches the data PC.

This is a resource-check correction only, not permission to lower thresholds, terminate jobs, change the model/data/seed or bypass protections. Local test execution still uses the existing all-GPU resource preflight runner. Tests must show selected1idle/other0busy succeeds, and selected-card busy/utilization/VRAM or UUID mismatch still refuses.

The original GENSEG-JOINT-REVIEW-SNAPSHOT/DELIVERY are historical receipts for the initial delivery. This patch does not rewrite them or imply their runtime.py hash is current. Runtime experiment records continue to hash the actually loaded source. New review/test evidence accompanies this fix. Canonical splits, patient arrays, old weights and target manifest are unchanged.

Server-first synchronization remains the user preference: copy the corrected runtime.py into the matching repository path, commit/push explicit fix files there, then pull the same branch locally. No commit/push is performed automatically here. Output C:/lijialin/HD-BET_outputs/genseg_joint_trial01 remains fresh for the two initial-guard refusals shown by the user. If any subsequent run creates outputs, preserve them and use trial02.
