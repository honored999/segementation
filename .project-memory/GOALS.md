# Goals

## Project goal
- Evaluate stroke-lesion segmentation and supporting image-alignment methods without modifying raw medical data or weakening fixed-split evidence boundaries.

## Current milestone
- Run the isolated ADN diagnostic workflow on server-side Dataset501 Fold 0 training DWIs and inspect case-level QC before considering any segmentation integration.

## Success criteria
- Preserve the existing ADN architecture/ranges and Dataset501 split.
- Keep alignment supervision image-only and outputs separate from raw data.
- Treat diagnostic QC as engineering evidence, not formal clinical or segmentation evidence.
