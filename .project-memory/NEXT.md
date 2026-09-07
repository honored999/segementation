# Next

## Current focus
- Run the read-only fixed-grid loss landscape on selected server-side DWIs to determine whether the unchanged alignment losses prefer near-identity candidates before considering more training.

## Next actions
1. Run `python -m standalone_nnunet2d.tools.adn_loss_landscape --dataset-dir <Dataset501> --cases case021 case034 case051 --device cuda --output-dir <new-output-dir>`.
2. Compare identity against each case's global minimum, ratio, improvement percentage, and boundary flag without assigning physical direction to the tx sign.
3. Decide from the diagnostic evidence whether a broader grid is warranted; do not alter losses/ranges or continue training implicitly.
4. Separately re-run the Fold 0 diagnostic preflight when training-path validation is desired.

## Blockers
- The new model-input padding path is synthetic-validated locally but pending server rerun on the 76 Fold 0 training images.
