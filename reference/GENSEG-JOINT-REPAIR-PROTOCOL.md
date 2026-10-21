# Joint repair protocol v1 (implementation contract)

Strategy: `joint_adam_trainbn_positive_sgdproxy_v1`. Controlled adaptation, not exact differentiable Adam or paper reproduction.

| State | Actual | Virtual/architecture validation |
|---|---|---|
| G/D/S | fresh Adam, 2e-5/2e-5/1e-5, betas .5/.999, eps 1e-8, zero decay | private entry weights; one SGD G/S proxy; D fixed |
| BN | restored G/D statistics; fresh S; G1/D3/S2 per unit | entry snapshot, eval, no running-stat propagation |
| alpha | fresh staged Adam 1e-4; upconv only | direct generated-image and indirect virtual-G routes |
| sampling | real: unchanged patient/plane sampler, includes negatives | independent uniform inner61 positive slices + aligned flips; inner15 architecture only |

Weights-only transition restores audited G/D and both alpha tensors, never optimizer/RNG/sampler/S. Legacy remains unchanged. Any failure after live mutation poisons the engine; no rollback/retry/completion claim.

Snapshot: owned contiguous read-only image/label arrays; pre/post full audit; unit stat guards and source hashes; final full audit. Stat guards do not detect content changed and restored between checks. No external19 arrays. Cache capacity is explicit and failure stops, without fallback.

Tests: single-thread tiny CPU only. Numerical comparison tolerance fixed before tests: rtol=1e-6, atol=1e-8 for identical new-engine adapters; double finite difference rtol=1e-5, atol=1e-8. Real CUDA peak/throughput and 20-epoch production weights NOT_RUN. Independent Level3 review remains required.


B1/B2 focused correction (2026-10-21): no optimizer/BN/loss/sampler or actual/virtual protocol changes. joint_repair.py and core.py unchanged. Snapshot/check now separate permanent owned float32 image + original signed-label dtype cache from temporary loading allocations. Header-only Blosc2 image/label and ZIP-member NPY mask metadata are checked first. Only float image / signed label with itemsize <=8, matching [1,Z,H,W], bool [Z,H,W] keep are accepted. NPZ/NPY source loaders are explicitly outside the audited joint snapshot backend and refused (legacy Source unchanged).

For N voxels, source image itemsize X and label itemsize Y: final=N*(4+Y). Additional transient envelope=N*(2*X+Y+2+32+32)+explicit loader workspace reserve. This intentionally sums overlapping phases conservatively: source image+seg materialization; TargetDataset image copy; keep and inverse keep; Source finite/isin validation and snapshot finite/isin validation with 32 bytes/voxel each for boolean masks, comparisons, integer indexing/table operations (binary signed labels -1/0/1). Target copy replaces source image after return; source/target locals are freed on return; owned copies remain read-only. Unknown external Blosc codec workspaces, memory-mapping effects, pickle properties and Python allocator overhead are NOT bounded by the array expression: separate positive loader workspace reserve required. Reserve is a user assumption, not proof of a precise peak. Model reserve is independent; production peaks still require real preflight.

Per-case pre-load guard: final_estimate-actual_cached_bytes+case_extra_transient, with model reserve added by runtime. Owned dtype/shape/nbytes checked against headers, live post-load resources checked, source pre/post hashes and read-only ownership retained. check exposes both estimates and combined startup requirement without image/label decompression; inherited TargetDataset mask audits remain disclosed.

Production hyperloss tests freeze virtual G numerical weights at alpha0 for direct F(alpha,V_G(alpha0)); indirect uses F(alpha0,V_G(alpha)). Test-side production virtual_chain/bind_alpha wrappers isolate routes without replacing loss/gradient operators. Independently perturbed re-detach removes autograd but retains finite-difference numerical dependence; negative control explicitly expects None for severed indirect, zero for zero S proxy rate. Both production paths also have FD relative-error <1% in addition to unchanged rtol1e-5/atol1e-8, preventing tiny indirect derivatives from passing solely via atol. No CLI gradient-path switches.
