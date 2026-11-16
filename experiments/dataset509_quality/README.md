# Dataset509 Q / independent C (Dataset514 fixed-pool adapters)

Current addition (2026-11-16): independent C sampling, local CPU engineering
checks only; no C deployment, real preflight, training or evaluation. Existing
Q implementation/evidence is preserved. Latest Q server status is recorded by
the main agent in the root repository reports/dataset509_q_20261115/MAIN_HANDOFF.md.
The Q sections below describe the original leaf implementation and its evidence;
they do not supersede that main-agent run record.
Read-only external dependency: LesionGen `c498118e6e3b3e4bb7a1c2380bb777c13f601e04`.
`trainer514.py` is byte-for-byte the existing identity adapter from
`reports/dataset514_20261108/trainer514.py`; the rest is a local thin extension.
Requires existing nnU-Net 2.8.1, NumPy, SciPy, Torch and batchgenerators; install nothing.

## Modes and baseline preservation

- `baseline` (default in the example baseline config): factory returns the original
  514 adapter directly, with inherited original loader, augmentations, loss,
  train_step and validation. No Q dataset, sidecar read or source loss is used.
- `q`: original fixed augmented pool and sampling probabilities. Record the selected
  Z from the official loader's actual `get_bbox`, where depth is exactly one.
  Original crop/DA/worker pipeline operates on unchanged image and segmentation.
  Scalars `q_source`, `q_quality`, `q_lesion_pixels` travel alongside the batch,
  never as network channels or extra segmentation labels. Missing/mismatched metadata
  fails. There is no per-case source guess. Original RNG behavior is retained by
  source instrumentation (verified against official loader with complete DA).
- `quality_filter`: remove synthetic planes with weight < .5 from each appended
  volume, recompute foreground locations, retain all real planes. Use original
  mixed Dice+TopK10, not Q loss or Q weighting. Metadata is for exposure only.
  All-filtered cases remain real-only, and an all-filtered pool remains usable.

Only binary DWI, Dataset514, 2D, fold0, one visible GPU, no cascade, no DDP,
no regions and no torch.compile are supported by Q/filter. The constructor keeps
nnU-Net's original signature (its init captures signature parameters from locals).
Input config is supplied through a run-local environment path, not a new parameter.
Baseline uses the same explicit launcher compile setting for a matched control.
Other trainer paths remain unmodified. CPU checks are synthetic and small.

## Offline candidate quality signal

`quality.py` reads **native exported image/label pairs** before model augmentation,
not student loss, teacher output, validation metrics or an inference-time mask.
For every pair, binary lesion M defines a ring
`R = dilate(M,5) minus dilate(M,1)` using SciPy's default 4-connected cross,
clipped to image bounds, excluding zero-valued image pixels. In native pixel units:

```
c = (median(I[M]) - median(I[R])) /
    max(1.4826 * MAD(I[R]), .01 * (P95(I_nonzero)-P5(I_nonzero)), 1e-8)
```

Images must be finite and binary labels aligned; invalid arrays fail. Fewer than
3 lesion pixels or 16 ring pixels (including empty/full masks) give no score and
synthetic weight 0. Image-border lesions use the available clipped ring; if it is
insufficient, weight is 0. All-zero images also have insufficient ring support.
A score arithmetic overflow fails. This mask use is training-only quality scoring,
never inference conditioning or ROI construction.

Reference quantiles L/M/H = 5th/50th/95th percentiles of valid REAL native slices
in the outer training100 pool only (1593 real pairs, 100-case coverage). No validation
pair participates. At least 100 valid reference slices and L < M < H are required;
otherwise sidecar construction refuses the degenerate signal. Define
`d=abs(c-M)/(M-L if c<M else H-M)` and `w=max(.1,exp(-d*d/2))` for a valid score.
Invalid-score synthetic planes have w=0. The symmetric rule penalizes unusually
weak **or** strong contrast. The .1 floor avoids deleting valid but unusual slices
in Q. Filter uses the same w with .5 threshold, an initial conservative engineering
choice (roughly 1.177 reference widths), not tuned with held-out outcomes.

This is **a candidate plausibility proxy**, not demonstrated perceptual quality
or generator-error detection. Ring contamination, acquisition contrast, native
pixel spacing, lesion size and true low-contrast lesions can confound it. 2D
lesion pixels are not physical lesion volume. No actual pool distribution has
been measured in this implementation task. Do not claim signal reliability or
segmentation benefit. Sidecar returns a minimal anonymous audit of real/generated
score and lesion-size distributions, invalid/positive fractions, weight quantiles,
filter retained fraction and retained lesion sizes. Main agent must inspect these
before formal use; if the signal is unconvincing, report the evidence gap and keep
baseline/interface instead of claiming Q validity.

Sidecar binds native manifest/pair hashes, private provenance hash, source split,
training reference hash, scoring config/hash, and prepared pool manifest (which
binds every prepared synthetic file). It is a **private derived artifact**; do not
commit it. Per-source synthetic ordering follows the pinned `segmentation512.prepare`
manifest iteration contract. Sidecar and runtime both verify pool counts/hashes;
no split or raw/prepared input is overwritten. Scoring does not regenerate data.

## Q loss and boundaries

At each supervised resolution, group planes by source s and count valid source
pixels Ns, excluding ignore_label if present. Rank **unweighted** CE independently
inside each source and take `Ks=max(1,floor(.1*Ns))` (skip Ns=0). Compute
`CEs=sum(selected CE_pixel * w_plane)/Ks`. Each source has its own TopK budget;
weights never enter the denominator. Q Dice is foreground **per-plane** soft
`1-Dice`, smooth=1e-5, then `Ds=sum(w_plane*(1-Dice_plane))/number_of_valid_planes_s`.
This deliberately uses per-plane Dice rather than baseline batch Dice; it prevents
pixel-mass normalization from canceling a whole plane's quality discount. Empty
foreground planes still penalize foreground prediction. Ignore pixels enter
neither numerator nor denominator; all-ignore planes are excluded.

`L = (CE_real+D_real) + lambda*(CE_synthetic+D_synthetic)`, initially lambda=1.
Real weights are exactly 1; generated weights are the offline w. Fixed denominators
make all-generated weight .2 shrink that source's gradients by .2. No division by
sum of weights or by number of active sources. A missing source contributes zero;
all-zero synthetic weights yield connected zero gradient. Tiny valid sets use at
least one TopK pixel. This differs deliberately from baseline floor-zero behavior.

Deep-supervision scales are the original normalized 1/2^i with final scale zero
(single GPU, compile off); apply source/quality scalars at every active resolution.
Q doesn't change target downsampling. In this binary protocol the original
augmentation converts crop/spatial padding -1 to background 0, and Q treats that
background identically. The loss separately supports explicit ignore_label=2 in
focused checks, but the production Dataset514 label contract rejects ignore,
regions or new classes. No side metadata is spatially interpolated.

Validation always calls the unmodified original loss and original real-only
validation dataset; original EMA pseudo-Dice selects checkpoints. No Q-specific
validation loss affects stopping/selection. Full original-space 28-volume
inference/evaluation remains the main agent's existing pipeline.

## Run commands (main agent; not executed here)

Fill local copies of `baseline.json`, `q.json`, `quality_filter.json` with existing
pinned preparation/plans/split/508 paths and SHA256. No patient list, credentials,
notification address or hardcoded GPU ordinal is stored in configs. The main
agent supplies the currently available GPU UUID after a resource check.

```powershell
# Build a NEW private sidecar from existing native pool; shared input is read-only.
python quality.py --package <existing-native-package> --pool <prepared-base>/synthetic_pool --split <prepared-base>/splits_final.json --output <new-dir>/quality.private.json
# Run in existing server nnunet5090; no dependency installation.
python focused_checks.py --lesiongen-repo <pinned-LesionGen-repo>
# Substitute each mode's completed config; every invocation requires a NEW root.
python run.py --config baseline.local.json --stage preflight --gpu-uuid GPU-<uuid> --output <new-baseline-preflight>
python run.py --config q.local.json --stage preflight --gpu-uuid GPU-<uuid> --output <new-q-preflight>
python run.py --config quality_filter.local.json --stage preflight --gpu-uuid GPU-<uuid> --output <new-filter-preflight>
# Only after main-agent acceptance and real preflight, with common budget fixed:
python run.py --config q.local.json --stage train --gpu-uuid GPU-<uuid> --output <new-q-training>
# Analogous --stage train for baseline.local.json / quality_filter.local.json.
```

`api_contract.json` fingerprints inspected local nnU-Net 2.8.1 official methods.
Runtime refuses API drift. The server API was **not accessed** in this leaf task;
main agent must run the focused script there; review differing owning source and
adapt only if needed, never silently replace the expected fingerprint. Q/filter
checkpoints contain a quality-identity pin; external resume/optimizer restoration
is rejected by the inherited 514 contract. 508 is loaded only through official
`load_pretrained_weights`; optimizer/early stopping start fresh. For separate
inference, put this directory and external trainers/src on PYTHONPATH and retain
the run's DATASET509_QUALITY_CONFIG/pool variables, or reuse the constructed
trainer within the main agent's evaluation chain.

The launcher checks existing NumPy cache barrier (all 128 image/seg/property
files), manifest, pool content, fixed split/plans and 508 hash, dependency commit
and clean owning source. It cannot establish patient independence from case
strings alone: main agent must reuse the established patient-split evidence.
Output is fresh and disjoint from protected inputs. Stages are independent;
preflight performs one real train + one validation step, writes non-formal status,
and never resumes into formal training. Failed compute propagates with FAILED.
No automatic evaluation, scheduler registration, deployment or notifications.

Initial early stopping/iterations in example configs are inherited trainer
**defaults**, not a verified copy of the previous server queue configuration.
Main agent must fill the agreed protocol; equal early-stopping settings do not
ensure equal update counts. A strict common update-budget experiment can use
`min_epochs=max_epochs` with identical iterations and LR horizon across modes,
keeping original validation/best-checkpoint selection; record that deliberate
protocol choice. Historical 81.518% is fold0 development evidence, not a matched
budget control. Do not mix it into a causal Q ablation as if matched.

After successful training steps, Q/filter write cumulative anonymous exposure
counts, positive-weight synthetic fraction (count/total), weight sum/histogram,
and lesion-pixel-size bins to `quality_exposure.jsonl`; bins are
[0,1), [1,16), [16,64), [64,256), [256,1024), [1024,infinity).
Real exposure sizes use complete preprocessed planes; synthetic sizes use complete
native scored planes, so compare within source rather than equating their pixel
units. Exposure excludes prefetched/discarded warmup batches and counts only
optimizer-accepted batches. Logging uses no RNG or model state; logging I/O errors
are reported without changing optimization. Fresh-run counts are not resumed.


## Independent C (2026-11-16)

`mode=c` selects `nnUNetTrainerDataset509Curriculum`, a separate subclass of the
original 514 adapter. `_build_loss` and `validation_step` are inherited unchanged:
**original mixed Dice+TopK10, ordinary deep supervision, no SourceLoss, no quality
loss weights**. C does not support Q+C in this iteration. The only Q factory
change is a lazy C branch; baseline/q/quality_filter retain their old paths.
The old Q test's unsupported-mode assertion now uses `q+c` instead of `c`.

C uses the same native 1593 real + 478 synthetic pool, without regeneration.
`curriculum.py` builds a NEW private sidecar and prints an anonymous training-only
audit. It verifies the existing Q sidecar/native pair order and hashes, split,
plans, prepared synthetic pool, and real prepared seg/property hashes. No held-out
image is opened. Native geometry comes from the pinned provenance's audited
`spacing_xyz_mm` and `affine_xyz_to_lps_mm`; the owning LesionGen `audit.read_volume`
requires explicit NIfTI mm units. The builder verifies finite positive spacing,
affine-axis lengths and orthogonality; absent/mismatched geometry fails rather
than assigning pixel units. The provenance hash and per-case geometry/raw hashes
are included in the private sidecar. It also hashes the complete training
reference and fixed curriculum policy. This is trusted existing local provenance,
not permission to unpickle arbitrary downloaded nnU-Net property files.

C physical size is **2D native XY lesion area**:
`area_mm2 = binary lesion pixels * spacing_x_mm * spacing_y_mm`. It is neither
full 3D lesion volume nor number of pixels renamed as mm². Native real layers
are mapped to preprocessed Z via the verified crop bbox with no Z transpose or
Z resampling (the existing pinned 2D preparation contract); all real native Z
must be present, including negative layers. Generated order is the existing
per-case manifest append order. Prepared segmentation foreground counts are
verified again at runtime against each selected case. Missing spacing, altered
pool, mismatched plane metadata or unsupported geometry is an actual error.

Use four positive groups: small/large area at the real positive-layer median,
crossed with low/non-low contrast at the valid real positive-layer 25th contrast
percentile. Contrast is the existing native ring signal, not student/teacher or
validation behavior. Ties go to small area / non-low contrast. Unknown-contrast
positive real layers and negative real layers are separate preserved categories;
they do not determine contrast thresholds. No validation result selects thresholds.
Degenerate/tied empirical distributions may leave groups absent, which the audit
reports. Thresholds still have a well-defined deterministic interpretation.

Let `n_g` be the real reference frequency for positive group g. The synthetic
group gain is `a_g=min(2,sqrt(max_h(n_h)/max(n_g,1)))`; absent groups receive the
bounded cap and are explicitly reported, not filled with new images. A synthetic
plane is a **candidate trusted** plane only if its existing Q proxy weight >= .5,
its contrast group is known and, for a foreground slot, it has actual prepared
foreground. The proxy remains unvalidated and may confound true low contrast
with generation errors; do not claim that this gate identifies image quality.
A plane is candidate difficult if low contrast OR its group gain exceeds 1.

For zero-based epoch e define `alpha=clip((e-20)/80,0,1)`. Epochs 0..19 are the
20-epoch start stage; e=20 has alpha=0, e=60 alpha=.5, e>=100 alpha=1. Nominal
synthetic source probability is `.10 + .15*alpha`. For a uniformly drawn case,
synthetic plane weight is `trusted * a_group * (alpha if difficult else 1)`.
Normalize these weights **only among available synthetic planes**, then multiply
by the source probability. Thus group-relative gain survives normalization and
the source mass itself changes from .10 to .25. If synthetic weight sum is zero,
source probability becomes zero and that draw safely falls back to real; distinguish
no trusted candidate from waiting for the difficult phase. Cases missing a group
simply sample their available groups. No case is removed from case sampling.

Real random slots use uniform real Z (negative slices remain eligible). Real
foreground slots choose real Z proportional to actual foreground pixel count;
if no real foreground exists, retain uniform real Z and record the fallback.
Real source mass is always at least .75. Unlike the baseline's stored foreground
coordinate subsample, C rebases actual foreground pixels on its chosen plane
before calling the original bbox implementation; this is an explicit sampling
change. Ordinary XY foreground crop, padding and augmentations remain upstream.
No source-dependent loss scaling occurs. The overall synthetic exposure need not
reach .10/.25 because availability fallback and foreground slots are conditional.

`CurriculumLoader.get_bbox` selects the actual Z, rebases that exact plane's
foreground to depth one for the original XY bbox function, and returns the original
Z to the original crop/augmentation code. `c_source/c_group/c_z` are separate batch
metadata, never image channels or labels. The original single-thread augmenter
has no prefetch queue. Only workers=0 is accepted in config, trainer and loader;
every epoch hook updates the actual loader using current_epoch. A stale batch
raises. No optimizer, LR, epoch or budget reset at phase transitions. Validation
uses the original real-only dataset, loader, loss and checkpoint criterion.

`curriculum_exposure.jsonl` records per-epoch/cumulative actual source×group counts,
mean conditional source×group probabilities for optimizer-accepted samples,
phase alpha, gains/cap and fallback counts. The sidecar audit also provides
probability tables averaged over training cases for random/foreground slots at
start/transition/end, along with missing groups, real frequencies and thresholds.
For batch12 with ordinary .33 foreground oversampling, the default deterministic
slot split is 8 random + 4 foreground. Logging is anonymous, draws no RNG, and
I/O logging failure does not modify training. Counts exclude discarded warmup
loader batches. Fresh-run counts are not restored by an evaluation checkpoint.

### C commands for main-agent handoff (not executed in this C task)

Fill a local copy of `c.json` using existing preparation/plans/split/508 hashes.
Build sidecar with the same `curriculum` field (the placeholder sidecar pin is not
read during construction), inspect its audit, then fill its path/SHA256. Keep C
outputs separate from all raw, prepared, native package and old Q outputs.

```powershell
python curriculum.py --package <existing-native-package> --base <existing-prepared-514-base> --plans <existing-plans.json> --quality-sidecar <existing-quality.private.json> --config c.local.json --output <new-C-metadata-dir>/curriculum.private.json
python focused_checks_c.py --lesiongen-repo <pinned-LesionGen-repo>
python run.py --config c.local.json --stage preflight --gpu-uuid GPU-<available-uuid> --output <new-C-preflight-root>
python run.py --config c.local.json --stage train --gpu-uuid GPU-<available-uuid> --output <new-C-training-root>
```

The main agent owns current resource checks, one real GPU preflight, server
acceptance, deployment and run. No such commands were executed by this C leaf.
The confirmed historical C protocol in `c.json` is seed12345, 2D/fold0, original
508 weights only, batch12, 250 training/50 validation steps per epoch, at most
1000 epochs, min_epochs=0, patience100, min_delta=.0001. C stages count inside
that budget. Early stopping can yield different actual updates across groups;
this is an exploratory historical-budget comparison, not equal-step causality.

`run_layout.py` minimally parameterizes the existing controller/evaluator. An old
job_spec with no mode still uses q.formal.json / q_training / q_metrics exactly.
For a NEW C run root set job_spec `mode` to `c` and put the filled config at
c.formal.json; defaults become c_training / c_metrics. Optional config_name,
training_name and metrics_name must be distinct leaf names under that root.
Controller commands are still sequential train then evaluate. Main agent must
populate source_files with the NEW deployed snapshot hashes (including new C
modules and run_layout), and must deploy into a NEW code directory, never replace
the old Q server snapshot. Commands after setup:

```powershell
python server_job.py --run-root <new-C-controller-root>
# Or evaluation alone after completed C training with that same job_spec/layout:
python evaluate_run.py --run-root <new-C-controller-root>
```

Evaluation still loads the best checkpoint and predicts the same full original
28 volumes. Historical seven-metric formula (`--f2-mode paper`) and supplemental
component metrics are unchanged. No scheduler registration or notifications are
performed by the C implementation task.

Accepted Q hashes are preserved verbatim in SOURCE_HASHES_Q_ACCEPTED_20261116.json.
Current SOURCE_HASHES.json includes C additions and updated shared entry points;
HANDOFF_C.md names changed/unchanged files and actual C-only verification. Existing
Q HANDOFF and results remain historical evidence. Real C sidecar distribution,
GPU preflight, formal training and evaluation are all **NOT RUN** here.
