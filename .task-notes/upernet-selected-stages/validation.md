# Validation evidence — implementation stage only

Base: 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89; branch codex/upernet-configurable-stages.
All commands used D:/Anaconda/envs/newconda/python.exe. Installed nnunetv2==2.8.1 source and fresh discovery checked. No environment installations/upgrades.

Resource sampling: Windows GetSystemTimes over one second; GlobalMemoryStatusEx RAM; nvidia-smi each GPU. Serial one-thread CPU synthetic fixtures; no GPU model allocation. Samples below 80%; no required local synthetic coverage deferred by guard. These are current preflight readings, not peak measurements.

## affected-final
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\affected-final"]`
- Exit: 0; visible summary: 105 passed, 8 warnings in 31.98s
- CPU 13.73%, RAM 48.62%, GPU(s): [{"index": 0, "gpu_percent": 13.0, "vram_percent": 15.266927083333334, "used_MiB": 938.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/affected-final.json`.

## baseline-final
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_official_upernet_trainers.py", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\baseline-final"]`
- Exit: 0; visible summary: 25 passed, 7 warnings in 9.09s
- CPU 11.07%, RAM 48.79%, GPU(s): [{"index": 0, "gpu_percent": 11.0, "vram_percent": 15.152994791666666, "used_MiB": 931.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/baseline-final.json`.

## focused-expanded
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\focused-expanded"]`
- Exit: 1; visible summary: 1 failed, 54 passed, 6 warnings in 29.12s
- CPU 7.61%, RAM 49.07%, GPU(s): [{"index": 0, "gpu_percent": 11.0, "vram_percent": 15.543619791666666, "used_MiB": 955.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/focused-expanded.json`.

## focused-final
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\focused-final"]`
- Exit: 0; visible summary: 55 passed, 6 warnings in 29.23s
- CPU 7.56%, RAM 48.24%, GPU(s): [{"index": 0, "gpu_percent": 11.0, "vram_percent": 15.152994791666666, "used_MiB": 931.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/focused-final.json`.

## green-first
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\green-first"]`
- Exit: 1; visible summary: 1 failed, 48 passed, 5 warnings in 27.25s
- CPU 8.66%, RAM 50.56%, GPU(s): [{"index": 0, "gpu_percent": 7.0, "vram_percent": 15.462239583333334, "used_MiB": 950.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/green-first.json`.

## predictor-fixed
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_real_predictor_rebuild_and_multifold_weights", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\predictor-fixed"]`
- Exit: 0; visible summary: 1 passed, 5 warnings in 6.74s
- CPU 8.48%, RAM 50.72%, GPU(s): [{"index": 0, "gpu_percent": 8.0, "vram_percent": 15.218098958333334, "used_MiB": 935.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/predictor-fixed.json`.

## red-confirmed
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\red-confirmed"]`
- Exit: 2; visible summary: 4 warnings, 1 error in 4.28s
- CPU 8.28%, RAM 50.63%, GPU(s): [{"index": 0, "gpu_percent": 8.0, "vram_percent": 15.218098958333334, "used_MiB": 935.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/red-confirmed.json`.

## red-plans-ambiguity
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_source_plans_ambiguity_rejected", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\red-plans-ambiguity"]`
- Exit: 1; visible summary: 2 failed, 1 passed, 4 warnings in 5.48s
- CPU 9.28%, RAM 48.99%, GPU(s): [{"index": 0, "gpu_percent": 17.0, "vram_percent": 15.804036458333334, "used_MiB": 971.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/red-plans-ambiguity.json`.

## trainer-fixture-fixed
- Command argv: `["D:\\Anaconda\\envs\\newconda\\python.exe", "-m", "pytest", "nnunet_ext_trainers/tests/test_upernet_selected_stages.py::test_actual_cpu_trainer_initialize_resume_and_validation_rebuild", "-q", "--basetemp", "E:\\study\\研一\\work14-图像分割\\segementation-upernet-configurable-stages-test-temp\\trainer-fixture-fixed"]`
- Exit: 0; visible summary: 1 passed, 5 warnings in 5.46s
- CPU 10.72%, RAM 48.86%, GPU(s): [{"index": 0, "gpu_percent": 10.0, "vram_percent": 15.266927083333334, "used_MiB": 938.0, "total_MiB": 6144.0}]; decision NORMAL.
- Exact stdout/stderr and command-specific Python source SHA256: `validation/trainer-fixture-fixed.json`.

## Failure provenance and resolution
- Initial evidence wrapper encountered GBK/UTF-8 printing failure before saving a result; discarded as reliable test evidence. Added UTF-8 console/subprocess settings. No production change for this tooling error.
- red-confirmed: genuinely missing new validate_upernet_feature_indices API (exit 2 collection RED).
- green-first: 48 passed/1 failed; predictor synthetic input omitted Z dimension. Corrected fixture to C×Z×H×W; predictor-fixed passed.
- red-plans-ambiguity: duplicate JSON keys and absent plans_name accepted (2 failed/1 passed). Minimal CLI guards added; included in later GREEN.
- focused-expanded: 54 passed/1 failed; official infer_dataset_class requires a format-bearing file in the synthetic directory. Added tiny synthetic npz sentinel; actual CPU constructor/initialize/load now passed. No raw data involved.

## Coverage
- Strict/default/inherited selections, 2/4/8 decoder levels, valid explicit four-level replacements, invalid scalar types/order/duplicates/range/both-axis scales.
- Frozen git-show baseline: four-level module/state tensor keys, shapes, initialization, bitwise logits and divisible-input accounting preserved. New extra_state is intentional, exclusive to new Trainer. Legacy encoder floor-based odd accounting left unchanged; new wrapper exact geometry tested.
- Tiny backward: stage0/deepest gradients, complete encoder including unselected last stage; Conv2d hook counting on odd and anisotropic inputs; full eight-stage 129×129 two-channel encoder, no full-size architecture/patch test.
- Strict checkpoint round-trip; same-tensor-shape different indices refused before model/optimizer/logger updates; missing/corrupt/type-spoof identity refused; DDP key prefix normalization, nested root pre-hook and CPU compiled wrapper loading.
- Actual official CPU Trainer constructor/initialize/optimizer setup and uninitialized checkpoint resume; single-output validation lifecycle. No training epoch or real-data --val executed.
- Fresh subprocess official resolver and builder signature; actual predictor.initialize_from_trained_model_folder with synthetic plans/dataset/fold checkpoints; real multi-fold sliding window with matched weights and refusal of mismatched first/later folds.
- CLI API/subprocess round-trip, inherited target, source bytes unchanged, data_identifier retained; same/existing/wrong-name/invalid source output rejection.
- Final focused 55 passed (exit 0); unchanged official-UPerNet baseline 25 passed (exit 0); entire inspected external-Trainer affected suite 105 passed (exit 0). Installed SciPy/PyTorch warnings retained in logs.

## Deferred / not run by task authorization
- Independent Level 3 review: PENDING; implementer inspection does not count as independent review.
- Server, CUDA forward/backward/compilation, actual distributed DDP execution, real checkpoints/patient data, real training/validation/prediction, preprocessing, full-size eight-level feasibility, memory/speed and five-fold formal performance: NOT RUN.
- Root repository/standalone suites: out of changed-component scope, NOT RUN.
- Git commit/push/merge: not authorized, NOT DONE. Protected baseline/root/NoStage7 contents unchanged as verified by final status.
- Temporary synthetic checkpoints/fixtures isolated in sibling task-test-temp root; not production output or review deliverables. No tests accessed raw medical data.
