# GenSeg joint author-U-Net pilot: source and protocol decisions

Status: implementation in progress. Engineering pilot only; no patient training or segmentation benefit verified.

## Primary and secondary sources
Primary: Zhang et al., Generative AI enables medical image segmentation in ultra low-data regimes, Nature Communications16:6486(2025), DOI10.1038/s41467-025-61754-6, Methods Eqs4-11, architecture search space and optimization algorithm. Secondary: pinned author repository reference/GenSeg at6480e178a5d44cb838f53f7951aae48138b8f5eb.

| Decision | Evidence class | Implementation contract |
|---|---|---|
| Three-level weights -> synthetic segmentation -> validation-driven architecture | Paper-defined | Virtual G update and virtual S update both retain higher-order graphs; alpha uses real inner-validation loss. Ordinary weights updated separately. |
| One-step gradient descent approximations | Paper Eqs5/7/8 | Explicit functional virtual SGD, no Betty dependency or first-order shortcut. |
| Direct alpha path through searchable G forward | Unresolved paper notation | Full autodiff includes direct alpha dependence and indirect dependence through G'; Eq9 only displays indirect chain. Document this interpretation. Independently test indirect chain. |
| GAN plus L1; BCE plus soft Dice | Author-code choice | Follow source losses with valid-pixel masks for nnUNet signed -1; real and synthetic coefficients both1. |
| Author UNet | User-selected / author-code choice | Source unet/unet_model.py and unet_parts.py reused, 1 input/1 output for binary DWI. No nnUNet Trainer integration. |
| Train76 / external19 | Existing project invariant | Preserve original fixed five-fold memberships. New inner61/15 derived only from76, saved under experiment output, never overwrite canonical split. |
| 3mm targets | Prior user-approved project adaptation | Reuse exact completed targets manifest23d25e2b2a6d7c7ca2a270690a9d4d35116ff4f834458b48a9049ff889894a56. Source images/labels remain read-only. |
| Existing preprocessing and target fill | Explicit inherited preprocessing | Keep original nnUNet plans/crop/grid/normalization; targets were built using old76-case generator global-minimum background fill. This pilot does not claim to refit source preprocessing within inner61. Joint input scaling is fit on inner61 only. |
| Fresh models | Isolation requirement | Do not load old76-trained weights, including generator, even though old run metadata/checkpoint hashes remain target provenance. No outside19 arrays. |
| Upconv architecture only in pinned runtime | Deliberate source limitation | Author generator's downconv is a plain convolution; conv_arch is unused. Learn only active upconv architecture and retain unused conv tensor explicitly. This is not the full searchable conv+upconv paper space. |
| Differentiable floating-point generated images | Necessary repaired adaptation | Do not copy author example's detached, quantized, equalized synthetic path or repeated-key dependency dictionary. |
| BatchNorm buffer handling and virtual optimizer choices | Engineering decisions | Functional virtual states isolated from live parameters/buffers; record frozen-stat/eval behavior and optimizer differences explicitly in run protocol. |
| Native plane resolution | Minimal-change decision | No spatial resampling by default; bottom/right padding only, preserving ignore labels. |
| First run | Engineering decision | Explicit short budget, no automatic 5000-update run, long-run quality or holdout metrics. Save evidence of finite gradients, alpha changes, resource peaks, complete/incomplete state. |

## Evaluation boundary
The new61/15 fit changes training availability relative to the old76-case offline generator. It is therefore NOT an isolated end-to-end-vs-separate efficacy comparison. A fair future comparison needs a separate generator/segmenter fitted on exactly the same inner61, same targets/pretraining budgets/initialization and untouched external19 evaluation. No such performance result exists yet. Training-slice losses or previews are not full-volume metrics.

All outputs on the data PC: C:/lijialin/HD-BET_outputs/genseg_joint_trial01 (newtrial directory for each attempt). Canonical project memory is not changed for unintegrated implementation. Source worktree is separate from codex/genseg-reference; no implicit push.

## Real-image contract (resolved before implementation)
G/D targets, S real-training images and inner15 real architecture-validation images all use the same3mm masked/fill target representation from TargetDataset. Scaling is fit on inner61 only and applied unchanged to inner15. Enforce patientrole beforeload. An indirect-only sentinel fixes explicitforwardalpha while retaining alpha throughvirtualGweights; removing that dependency kills the isolated indirect term, not necessarily the full total hypergradient. Original unmasked data are not a second segmentationtrainingbranch. Future outside19 fullvolumeevaluation is absent fromthispilot and would require the same inference-timemaskpipeline.


## Acceptance evidence and checkpoint semantics
Passing tests may establish the optimization graph and isolation contract, not image quality or DWI efficacy. The actual pinned generator and exact author UNet must exercise the same production jointstep at tiny CPU shapes; an analytical smooth fixture separately verifies hypergradient finite differences. The indirect-only test holds synthesis forward architecture fixed while differentiating virtual G weights; finite nonzero total gradients alone are insufficient. Runtime evidence reports both gradient norms and architecture value deltas. Ordinary parameters, all architecture globals and optimizer ownership/state, configuration, innerpatient split, scaling, immutable target/source identities, RNG state and completion metadata must be represented in run artifacts. Interrupted outputs remain incomplete and must never be promoted by manually creating a completion marker.

The copied reference software originally used a dependency dictionary with a duplicate generator key and detached/quantized synthetic inputs. New code must not execute that entrypoint or imply these constructs implement the displayed paper derivative chain. No upstream reference files are patched; repaired adaptation is separate under experiments/genseg_joint.

## Historical validation qualification
The external19 are the project's existing fold0 DEVELOPMENT validation cohort, not a newly locked pristine test set. Earlier qualitative HD-BET/3mm mask checks included at least one member of that cohort. Thus existing preprocessing decisions are not proven independent of all19, and a later result on them is still development evidence. This pilot's guarantee is narrower and verifiable: no19 image/label arrays enter THIS joint fitting, scaling, architecture optimization or checkpoint selection. No final-test or fivefold-OOF claim is supported.

## Outer-architecture optimizer (numerical fix before acceptance)
The first actual-network tiny CPU fixture exposed a finite hypergradient whose default-rate float32 SGD update rounded to zero. That run correctly refused success. The accepted revision must use a staged Adam update ONLY for outer upconv alpha (lr1e-4, betas0.5/0.999, eps1e-8, weight_decay0). This follows the author-code optimizer family and deliberately deviates from Eq8's displayed plain SGD update. Virtual G/S and actual G/D/S remain stateless SGD. Adam candidate and moments are staged on private tensors; optimizer state commits only after all ordinary candidates are finite. Failure must leave live model/architecture/moment state unchanged. Default-rate actual-network proof, not an increased-learning-rate test alone, is required before packaging. No rates are auto-tuned, and no retry based on external19 is allowed.
