# Configurable-depth PlainConvUNet Trainer

This external Trainer keeps the official *PlainConvUNet* encoder and matching
*UNetDecoder*, with TopK10 loss and early stopping. Set
*encoder_last_stage=s* to retain zero-based encoder stages *0..s*; valid values
are *1* through the original final stage. For the eight-stage Dataset501 plan,
stage 5 keeps six encoder stages and removes stages 6 and 7; stage 6 removes
stage 7; stage 7 keeps the full encoder. Each depth is part of the
plan/configuration identity. Deep supervision is disabled.

On the training server, activate the existing nnU-Net environment and ensure
the *nnUNet_preprocessed* environment variable is set. This command writes a
new derived plans file, leaves the source plans and their *data_identifier*
unchanged, and does not touch *splits_final.json* or rerun preprocessing. It
stops if the destination or generated configuration names already exist.

~~~cmd
python -c "import json,pathlib; d=pathlib.Path(r'%nnUNet_preprocessed%\Dataset501_StrokeLesion'); src=d/'nnUNetPlans.json'; dst=d/'nnUNetPlansPlainConvDepth.json'; assert not dst.exists(), f'{dst} already exists'; p=json.loads(src.read_text(encoding='utf-8')); n=p['configurations']['2d']['architecture']['arch_kwargs']['n_stages']; names=[f'2d_stage{s}' for s in range(1,n)]; assert not any(name in p['configurations'] for name in names), 'derived configuration name already exists'; p['plans_name']='nnUNetPlansPlainConvDepth'; p['configurations'].update({f'2d_stage{s}':{'inherits_from':'2d','encoder_last_stage':s} for s in range(1,n)}); f=dst.open('x',encoding='utf-8'); json.dump(p,f,indent=4); f.close()"
~~~

From the repository root, expose the external Trainer and start a new run:

~~~cmd
set "nnUNet_extTrainer=%CD%\nnunet_ext_trainers"
nnUNetv2_train Dataset501_StrokeLesion 2d_stage5 0 -tr nnUNetTrainerPlainConvDepthTopK10EarlyStopping -p nnUNetPlansPlainConvDepth
~~~

Use the same Trainer, plans identifier, and depth configuration for continuation
with *--c*. Validate a completed run with *--val*; nnU-Net uses that run's
*checkpoint_final.pth*. A different depth is a different configuration and
requires a fresh run.

No local tests or runtime validation were run, as requested.
