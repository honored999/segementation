# Next

## Current focus
- Run the read-only geometry-vs-loss diagnostic on the selected server-side DWIs to determine whether a simple content-pose correction improves centroid/tilt while the unchanged ADN loss penalizes it.

## Next actions
1. Run `python -m standalone_nnunet2d.tools.adn_geometry_loss_comparison --dataset-dir <Dataset501> --cases case021 case034 case051 --checkpoint <checkpoint> --device cuda --output-dir <new-output-dir>`.
2. Compare identity, ADN prediction, and geometry alignment using the reported loss changes plus centroid/tilt before and after; treat the heuristic geometry estimate as diagnostic evidence only.
3. Use the existing loss-landscape results together with the new comparison to decide whether the current loss conflicts with visibly improved pose; do not alter losses/ranges or continue training implicitly.
4. Separately re-run the Fold 0 diagnostic preflight when training-path validation is desired.

## Blockers
- The new model-input padding path is synthetic-validated locally but pending server rerun on the 76 Fold 0 training images.
