# Selected encoder stages — approved task implementation

Base: 8a41ce3b0d4e2c980c9e9063a60b055b0a841d89. User assigns direct implementation; no agents or Git integration.

1. Read `upernet_feature_indices` from resolved ConfigurationManager.configuration, outside architecture kwargs. Validate indices and cumulative 2D scales before official network allocation. Missing key alone uses existing automatic selector.
2. Generalize existing decoder lengths/loops/counts, preserve four-level names, order, initialization, interpolation and tensor paths. Full encoder executes even when deepest selected stage is earlier.
3. New Trainer alone constructs a subclass of the existing wrapper with network `_extra_state` containing versioned indices, cumulative scales, selected channels, encoder stage count, output classes, FPN and PPM identity. A root load pre-hook rejects missing/corrupted/different identity before PyTorch visits any parameter. Official predictor first-fold and later fold load_state_dict paths therefore enforce the same contract. Trainer load additionally validates before delegating to inherited early-stopping loader, avoiding optimizer/logger mutation on identity mismatch. No save/load/train loop copies.
4. Existing encoder library counts use floor division, inaccurate on odd strided convolution inputs. New wrapper computes complete encoder output counts from actual Conv2d geometry; shared decoder count already follows actual geometry. Preserve legacy wrapper accounting and checkpoint protocol; new-model count equals hooks on odd/anisotropic inputs.
5. Plans CLI reads only UTF-8 JSON; resolves inheritance with official PlansManager; validates geometry, output name and existing/same resolved path before exclusive creation. Source bytes and original data_identifier remain unchanged.
6. CPU synthetic focused RED/GREEN, frozen baseline via git show, real official discovery/predictor directory reconstruction, same-shape mismatch and affected regressions. Every invocation uses resource preflight and single CPU thread. No patient data, server/CUDA tests, training, formal evaluation or full-size eight-stage fixture.

Independent Level 3 review is pending and will be dispatched manually by the user. Canonical master memory files are absent at startup; branch memory only.
