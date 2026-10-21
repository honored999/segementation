# Goals

## Project goal
- Develop reproducible stroke-lesion segmentation with protected source data, fixed patient splits, traceable initialization and original-volume evaluation.

## Current milestone
- Consolidate the completed HD-BET3mm fold-0 initialization-versus-scratch development comparison and determine the next controlled experiment from paired evidence.
- Keep GenSeg quality repair gated by evidence; preserve earlier decoder experiments as available engineering options rather than the current primary milestone.

## Success criteria
- Compare identical cohorts, preprocessing, Trainer, supervision, stopping, checkpoint selection and metric definitions; vary only the declared factor.
- Archive source checkpoint hashes, protocol/configuration and paired case metrics before causal interpretation.
- Report all relevant metrics and separate selection effects, development observations, synthetic/static checks and formal five-fold OOF results.
- Require diagnostic image-quality gates before generated samples enter segmentation training; do not infer efficacy from loss, gradients or completion flags alone.
