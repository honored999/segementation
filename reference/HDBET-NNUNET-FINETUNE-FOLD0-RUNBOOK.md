# HD-BET3mm＋ISLES508源权重：普通nnU-Net全参数微调fold0

Approved steering 2026-10-19: replace the pending random-initialized screen with pretrained initialization, using the SAME fixed Dataset501 fold0 (76train/19development), ordinary deep supervision, TopK10 and early stopping. No source/test changes are needed. This addendum supersedes the fresh-training command in HDBET-NNUNET-FOLD0-RUNBOOK.md for this run only; its previously reviewed preparation tool is unchanged.

## Source identity and hypothesis

The two user-supplied roots are historical PRIVATE-DATA fine-tuning outputs:
- C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_pretrained_ES : ordinary full-parameter Dataset501 fine-tuning.
- C:\lijialin\models3d\nnUNet\nnUNet_results_ISLES508_CKPD_ES : CKPD Dataset501 fine-tuning.

Do not use an arbitrary checkpoint from either root as the ISLES source. The source below was explicitly supplied in the historical chat and appears in the old loading log; current existence/content/hash must still be checked on the server:

C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset508_ISLES2022DWI\nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d\fold_0\checkpoint_best.pth

The old six ordinary fine-tuning runs do not yet have a verified source-checkpoint hash. They are historical references until matched. Improvement with stripping is evidence consistent with a preprocessing/domain mismatch; it cannot alone establish that ISLES skull stripping caused the old poor result. HD-BET3mm is also not identical to the ISLES release's brain extraction. Keep preprocessing fixed; no validation-driven per-case repair.

## 1. CMD setup and metadata-only source existence check

```bat
conda activate nnunet5090
set "REPO=C:\lijialin\segementation\.worktrees\genseg-reference"
cd /d "%REPO%"
git status --short
git pull --ff-only origin codex/genseg-reference
set "PYTHONPATH=%REPO%;%PYTHONPATH%"
set "ORIG_RAW=C:\lijialin\models3d\nnUNet\nnUNet_raw\Dataset501_StrokeLesion"
set "ORIG_PP=C:\lijialin\models3d\nnUNet\nnUNet_preprocessed\Dataset501_StrokeLesion"
set "TARGETS=C:\lijialin\HD-BET_outputs\genseg3mm_trial01\targets"
set "TARGET_SHA=23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56"
set "TRIAL=C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_isles508_finetune_fold0_trial01"
set "PRETRAIN=C:\lijialin\models3d\nnUNet\nnUNet_results\Dataset508_ISLES2022DWI\nnUNetTrainerTopK10EarlyStopping__nnUNetPlansFrom501__2d\fold_0\checkpoint_best.pth"
set "SOURCE_RECEIPT=%TRIAL%_isles508_source.json"
python -c "import os; from pathlib import Path; p=Path(os.environ['PRETRAIN']); assert p.is_file(), 'Historical source checkpoint missing'; assert p.parent.name=='fold_0' and p.parent.parent.parent.name=='Dataset508_ISLES2022DWI'; print('SOURCE FILE EXISTS:',p,'bytes=',p.stat().st_size)"
```

Only read-only existence/stat metadata above, no weight/model/CUDA loads. If missing, stop and find the original Dataset508 run; do not substitute a private Dataset501/CKPD/UPerNet checkpoint or a wildcard choice. Preserve any server worktree edits. All new directories/receipt files must be unused; select trial02 if needed.

## 2. SERVER ONLY: source checkpoint metadata and hash audit

This reads the trusted, locally trained checkpoint on CPU (not a model), after the resource guard. It is NOT a local/static check. Default official nnU-Net checkpoints include objects requiring weights_only=False; apply this ONLY to this known user-owned training artifact. No dataset arrays are read. Non-head target tensor compatibility is still enforced by official nnU-Net at actual initialization.

```bat
python -m experiments.hdbet_nnunet.cli guard --gpu 1 && python -c "import os,json,hashlib; from pathlib import Path; import torch; p=Path(os.environ['PRETRAIN']); receipt=Path(os.environ['SOURCE_RECEIPT']); assert not receipt.exists(), 'Fresh source receipt required'; digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest(); c=torch.load(p,map_location='cpu',weights_only=False); a=c['init_args']; s=a['plans']; t=json.loads((Path(os.environ['ORIG_PP'])/'nnUNetPlans.json').read_text(encoding='utf-8-sig')); assert c['trainer_name']=='nnUNetTrainerTopK10EarlyStopping'; assert s['dataset_name']=='Dataset508_ISLES2022DWI' and s['plans_name']=='nnUNetPlansFrom501'; assert a['configuration']=='2d' and str(a['fold'])=='0'; assert a['dataset_json']['channel_names']=={'0':'DWI'} and a['dataset_json']['labels']=={'background':0,'lesion':1}; assert isinstance(c['network_weights'],dict) and c['network_weights']; assert all(s['configurations']['2d'][k]==t['configurations']['2d'][k] for k in ('architecture','patch_size','spacing','normalization_schemes','use_mask_for_norm')), 'Source/target plan mismatch'; out={'checkpoint':str(p.resolve()),'checkpoint_sha256':digest,'dataset':'Dataset508_ISLES2022DWI','trainer':c['trainer_name'],'configuration':'2d','fold':0,'epoch':int(c['current_epoch']),'source_plans_name':s['plans_name'],'target_plans_sha256':hashlib.sha256((Path(os.environ['ORIG_PP'])/'nnUNetPlans.json').read_bytes()).hexdigest(),'purpose':'pretrained initialization only; no private-data resume; source hash must match again before train'}; f=receipt.open('x',encoding='utf-8'); json.dump(out,f,indent=2); f.close(); print('SOURCE CHECK PASS',json.dumps(out))"
```

If any assertion fails, stop with the output; do not relax source identity/architecture requirements. Preserve this receipt (outside TRIAL so fresh data preparation remains possible). Source and private patients must be confirmed independent by dataset provenance; anonymized filenames alone cannot prove independence. The preserved checkpoint hash locks the new experiment's source but cannot retroactively prove the old six runs used identical weights.

## 3. Prepare real-DWI95 and preprocess2d, unchanged workflow

Use sections2 and3 of reference/HDBET-NNUNET-FOLD0-RUNBOOK.md, keeping the NEW TRIAL above:

```bat
python -m experiments.hdbet_nnunet.cli check --raw-dataset "%ORIG_RAW%" --preprocessed-dataset "%ORIG_PP%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%TRIAL%" --gpu 1
```

Only after CHECK METADATA PASS:

```bat
python -m experiments.hdbet_nnunet.cli prepare --raw-dataset "%ORIG_RAW%" --preprocessed-dataset "%ORIG_PP%" --targets "%TARGETS%" --targets-sha256 "%TARGET_SHA%" --output "%TRIAL%" --gpu 1
```

Review train76 QA (cases/<case>/qa/zXXXX.png) before long training. Frozen HD-BET2.0.1/weights/defaultTTA + native-plane3mm + empty-plane-retain, no GT repair, byte-exact original labels and full native geometry. Remaining19 use the same fixed image-only process and remain development validation. If data were already prepared for the prior cold-start trial, retain it; this conservative runbook prepares a separate fresh trial so existing outputs and audited roots remain unambiguous.

```bat
set "nnUNet_raw=%TRIAL%\nnUNet_raw"
set "nnUNet_preprocessed=%TRIAL%\nnUNet_preprocessed"
set "nnUNet_results=%TRIAL%\nnUNet_results"
set "nnUNet_extTrainer=%REPO%\nnunet_ext_trainers"
python -m experiments.hdbet_nnunet.cli preprocess-check --output "%TRIAL%" --gpu 1 && python -m experiments.hdbet_nnunet.cli guard --gpu 1 && nnUNetv2_preprocess -d 501 -c 2d -plans_name nnUNetPlans -np 2
```

Original plans/splits and inherited baseline-fingerprint compatibility file stay unchanged. No replanning. Do not rerun official preprocessing into an existing configuration directory; it deletes/rebuilds that directory. Preserve incomplete outputs rather than overwriting.

## 4. Fresh full-parameter fine-tuning from ISLES508

Keep historical initial_lr=0.01/SGD0.99Nesterov/weight_decay3e-5/PolyLR/TopK10/ordinaryDS/early stopping1000,max warm300,patience100,min_delta1e-4. Do not reduce LR or freeze layers in this first matched screen. This is ordinary full-parameter tuning, not CKPD. Official -pretrained_weights loads compatible non-segmentation-head tensors and starts fresh optimizer/scheduler/epoch/early-stop state; segmentation heads remain freshly initialized. It does not resume ISLES training.

```bat
nvidia-smi -i 1 --query-gpu=uuid,name,memory.used,memory.total --format=csv
set "CUDA_VISIBLE_DEVICES=GPU-6c48a375-dd11-ff27-ed3d-0508532a9c6d"
set "nnUNet_n_proc_DA=2"
```

Verify UUID against current physicalGPU1 and retain historical nnUNet_compile setting if known. Do not introduce compile changes as an experimental factor. The guard is a launch check, not a sustained utilization limit. In the SAME CMD, verify the frozen source hash immediately before guarded initialization:

```bat
python -m experiments.hdbet_nnunet.cli ready --output "%TRIAL%" --gpu 1 && python -m experiments.hdbet_nnunet.cli trainer-check --gpu 1 && python -c "import os,json,hashlib; from pathlib import Path; r=json.loads(Path(os.environ['SOURCE_RECEIPT']).read_text(encoding='utf-8')); p=Path(os.environ['PRETRAIN']).resolve(); assert str(p)==r['checkpoint'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['checkpoint_sha256'], 'Source checkpoint changed'; print('PINNED SOURCE PASS')" && python -m experiments.hdbet_nnunet.cli guard --gpu 1 && nnUNetv2_train Dataset501_StrokeLesion 2d 0 -tr nnUNetTrainerTopK10EarlyStopping -p nnUNetPlans -pretrained_weights "%PRETRAIN%" --val_best
```

Do not add --c, use private-data final/best weights as the source, or mix CKPD calibration/adapters into this run. ready refuses an existing fold_0. Confirm the startup log explicitly loads the expected Dataset508 path before the first epoch; official loader refuses missing/mismatched non-head tensors.

Expected result directory:

```text
C:\lijialin\HD-BET_outputs\nnunet_hdbet3mm_isles508_finetune_fold0_trial01\nnUNet_results\Dataset501_StrokeLesion\nnUNetTrainerTopK10EarlyStopping__nnUNetPlans__2d\fold_0
```

## 5. Evaluation and valid comparison

Follow section5 of the base RUNBOOK using this TRIAL: exact fixed19 prediction membership, checkpoint_best (--val_best), original full-volume GT and identical F2 convention. No cropping GT to the brain mask and no reporting patchEMA as full-volume Dice.

Compare this B (ISLES source + stripped privateDWI) with A (EXACT SAME checkpoint hash + original privateDWI), same ordinaryTrainer/plans/fold/initial LR/stopping/validation/augmentation and environment. Old ordinary results in nnUNet_results_ISLES508_pretrained_ES are descriptive references until source identity and protocol are matched. CKPD results belong to a different adaptation experiment. This screen changes private input preprocessing only within the pretrained A/B; it does not isolate every difference between ISLES and private acquisition or prove clinical effectiveness.

Local validation: documentation/embedded-code AST and independent scientific STATIC review only; no source changes, tests, checkpoint load, models, patient arrays, CUDA, server preprocessing or training executed locally.