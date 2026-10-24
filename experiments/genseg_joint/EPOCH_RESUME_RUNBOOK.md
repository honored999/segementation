# GenSeg nnU-Net S epoch recovery v1

Engineering implementation; independent Level 3 review and production GPU validation remain pending.

The nnU-Net S `train --epochs` entry now saves each completed epoch. Old author-S and `--steps` defaults remain final-only. Total epochs must match the saved plan and must not exceed 80. A saved completed plan cannot be extended. A process interrupted inside an epoch restarts that epoch from its last committed boundary. Already running old processes cannot acquire recovery points retroactively; missing weights cannot be reconstructed from logs.

Recovery format: `genseg_nnunet_epoch_resume_v1`, files `resume_epoch_NNNN.pt`, commit pointer `resume_latest.json`. The pointer lists the latest two committed points with SHA256. Only this attempt's older committed files are removed, after pointer publication. Temporary and orphan files from interrupted publication are not treated as committed points. Recovery files never create `complete.json`; final `joint.pt`/schema3 and the existing final80 inference gate remain separate. New final protocol metadata explicitly identifies epoch recovery and its lineage; historical schema2/3 without the marker retain their original policy checks.

State includes G/D/S, both alpha tensors, G/D/S Adam and staged-alpha states, engine/GAN steps, original BN baselines, module modes, validation generator, Python/NumPy global/torch CPU/selected CUDA RNG, positive inventory/RNG/steps, real inventory/permutation/cursor/RNG/steps. Restore audits existing parameter/alpha aliases and optimizer ownership. Cache rebuild and source/data/descriptor checks precede restoring RNG; saved scale is reused without refitting. Resume skips GD20 weight copying but still audits the bound GD20 source artifacts. New process resource admission is performed again on the same GPU UUID; allocator state is not claimed identical across processes.

Atomic publication: CPU copies directly from live tensors (no deepcopy of GPU moments), explicit CPU reserve admission, same-directory exclusive temporary checkpoint, flush/fsync, hash plus weights_only CPU readback, audit, atomic model rename; then temporary manifest flush/fsync and atomic pointer replacement last. No recursive deletion. Failure propagates and leaves the prior committed pointer readable. No claim of protection against hardware/filesystem failure beyond the OS guarantees of these operations.

## CMD commands

Run only after main-agent acceptance, manually dispatched focused independent review, and authorized synchronization into a separate server checkout. Do not edit or run from the source directory of an existing training process. The following setup prompts reuse the actual paths and byte budgets from the approved server command; no local path or budget is assumed to be valid on the server. Paste into interactive CMD (not PowerShell). No installation is required by these commands.

```bat
conda activate nnunet5090
set /p "CODE_ROOT=Approved separate server code checkout: "
cd /d "%CODE_ROOT%"
set /p "SOURCE=Original approved source 2d directory: "
set /p "RAW_ROOT=Original raw root (read only): "
set /p "TARGETS=Original targets manifest: "
set /p "BASELINE=Original baseline run: "
set /p "GD20=Original completed GD20 run: "
set /p "CACHE_BYTES=Original approved cache-budget-bytes: "
set /p "LOADER_BYTES=Original approved loader-workspace-bytes: "
set /p "MODEL_BYTES=Original approved model-reserve-bytes: "
set "GD20_SHA=1547b42e65a7f21e8452d3509ce862f059884dd7e2bb82047eea7b33b7a5b75b"
```

New run: select a new nonexistent output directory, different from any prior run. This starts a fresh 80-epoch experiment; it does not recover an old process.

```bat
set /p "ATTEMPT_OUT=New nonexistent attempt output directory: "
python -m experiments.genseg_joint.joint_cli train --strategy joint_adam_positive_nnunet2d_sgdproxy_v1 --source "%SOURCE%" --raw-root "%RAW_ROOT%" --targets "%TARGETS%" --baseline-run "%BASELINE%" --pretrained-run "%GD20%" --pretrained-sha256 %GD20_SHA% --output "%ATTEMPT_OUT%" --gpu 1 --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_BYTES% --model-reserve-bytes %MODEL_BYTES% --epochs 80
```

Resume: retain the parent run and checkpoint read only. Choose a committed point from its pointer and supply its separately recorded SHA256, not an automatically trusted hash of an arbitrary input. No valid committed point means recovery fails. Use the same data, bindings, GD20, source version and GPU; select another nonexistent output directory outside the parent run. `--epochs 80` is the original total limit, not 80 additional epochs. Logs start at completed_units + 1. A point from an originally shorter plan cannot be changed to 80.

```bat
set /p "RESUME_FILE=Committed resume_epoch_NNNN.pt absolute path: "
set /p "RESUME_SHA=Independently recorded checkpoint SHA256: "
set /p "ATTEMPT_OUT=Another new nonexistent attempt output directory: "
python -m experiments.genseg_joint.joint_cli train --strategy joint_adam_positive_nnunet2d_sgdproxy_v1 --source "%SOURCE%" --raw-root "%RAW_ROOT%" --targets "%TARGETS%" --baseline-run "%BASELINE%" --pretrained-run "%GD20%" --pretrained-sha256 %GD20_SHA% --output "%ATTEMPT_OUT%" --gpu 1 --cache-budget-bytes %CACHE_BYTES% --loader-workspace-bytes %LOADER_BYTES% --model-reserve-bytes %MODEL_BYTES% --epochs 80 --resume-from "%RESUME_FILE%" --resume-sha256 %RESUME_SHA%
```

NOT_RUN: real server interruption/recovery, GPU periodic-save CPU/global VRAM peaks, throughput, long-run storage behavior, and 80-epoch scientific outcomes. The reported historical three-unit GPU result did not validate this implementation. Local tests use tiny CPU models and hardware/data stubs where stated; they do not establish CUDA bitwise reproducibility or server readiness.
