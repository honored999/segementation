# PSF grayscale adaptation (450 nm)

This is a localized standalone adaptation of the read-only share package
`E:\study\研一\work14-图像分割\psf_conversion_share`, inspected on 2026-11-03.
It fits kernels to phase-only metasurface PSFs; it does not integrate a network,
train a segmentation model, consume medical data, or establish hardware accuracy.

## Provenance and preserved comparison

The following files are copied byte-for-byte. `fit_psf.py` remains the original
RGB comparison entry, including its original optional L1 comparison. No original
phase coefficients or propagation implementation were edited.

| Local relative path (also source relative path) | Source SHA256 |
| --- | --- |
| `fit_psf.py` | `E251CDA52251F3EB415E116AC2FA6ED5DDC476B3F7A9DF24E088EE55AD29C1E8` |
| `optics/propagation.py` | `B9D7CF9C302814EB232D13D74C26220C5D371CCD628B4DB353225E87193FCE1E` |
| `optics/fitted_phase_parameters_p400.py` | `A71E0B7DCCB4F0E9FCCD094A059EDC0B0ED7195EA88549EFD9C324A1F3D3BE20` |

The source guide `README_先看这里.md` was read, not copied; its SHA256 was
`A40A93BF51299440BF8F41089CE741365D4AC5E8664946224032FCB9DFACF28D`.
That guide traces the share code to
`white_light_psf_handover_slim/car_segmentation/src/psf/`, especially
`train_segmentation_psf.py`, `propagation.py`, and
`fitted_phase_parameters_p400.py`. This is share-package provenance, not an
independent audit of that upstream project. Source validation reports describe
that package; they are not runtime evidence for this adaptation.

## Grayscale contract and deliberate changes

`fit_psf_gray450.py` is minimally adapted from the preserved `fit_psf.py`:

- Input is exactly `[32,1,3,3]`, finite, via `.npy`, a tensor `.pt`, or a keyed
  state dict (including `model_state_dict` / `state_dict` wrappers). The complete
  tensor is validated before `--limit` selects its first output channels.
- `--demo` generates the same full `[32,1,3,3]` shape and validates it before
  limiting. Random demo kernels are synthetic, never trained task weights.
- Only `compute_phase_fitted(width.double(), 450)` is evaluated. The propagation
  factory receives scalar `450e-9`, selecting `AngularSpectrumPropagation`.
  The RGB dictionary wrapper requires three channels and is not instantiated.
  No red/green field, propagation, phase evaluation, or optimization is done by
  the grayscale entry. Other wavelength coefficients remain in the intact copy.
- Phase shape is `[1,n,n]`, complex field shape is `[1,1,n,n]`, and intensity
  output `psf_raw` and `target` are `[1,n,n]`. Phase evaluation remains float64;
  the optical field is converted to complex64 as in the source.
- Grayscale accepts only `--loss l2_shape` (also checked inside `shape_loss`).
  The copied RGB entry remains unchanged for comparison.
- Outputs require an entirely new directory, including refusal of an existing
  empty directory. The resolved output must be outside this tool source tree.
  The copied RGB entry retains its original, less strict result-file guard.

All fitting mathematics remain from the share entry: width is
`80 + 240 * sigmoid(raw_width)` nm, initialization is uniform `[-1,1]`, optimizer
is Adam, targets use nearest interpolation, kernels are flipped 180 degrees,
and positive/negative branches share one per-output maximum-absolute scale.
A wholly zero branch is skipped (empty width map, zero PSF, no loss history).
The best raw parameters are snapshotted before `optimizer.step()` and restored
before final export. Early stopping, finite loss/gradient checks, and the
least-squares gain calculation are preserved.

The exact source L2 shape formula is retained for active channels:

```python
p = psf[active].flatten(1)
t = target[active].flatten(1)
p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
s = (p - t).square().sum(1)
loss = s.square().mean().clamp_min(1e-24).sqrt()
```

Defaults remain `n=800`, `dx_nm=400`, `distance_um=2400`,
`camera_pixel_um=3.45`, `patch_camera=6`, `steps=500`, `patience=30`,
`lr=5e-3`, and `seed=42`. The P400 library still rejects other `dx_nm` values.

## Usage and exported files

Dependencies are the source package's NumPy >=1.24 and PyTorch >=2.1; no
installation or dependency change was performed. Prefer an existing `newconda`
environment. From the target worktree root, example commands for a separately
authorized future run are:

```powershell
conda run -n newconda python tools/psf_conversion/fit_psf_gray450.py --kernels E:/derived/gray_kernels.npy --limit 1 --out E:/derived/psf_gray450_first
conda run -n newconda python tools/psf_conversion/fit_psf_gray450.py --demo --limit 1 --out E:/derived/psf_gray450_demo
```

These examples have not been run. Do not place outputs in the source share
package or raw-data directories. Default `--out results` is relative to the
working directory; explicitly select a fresh derived-output directory.
`--limit` limits fitting, never relaxes the full input contract.

`electrical_kernels.npy` stores the selected original electrical kernels;
`shared_scales.npy` stores their positive/negative shared scales. Each
`kernel_XXX_pos.npz` or `kernel_XXX_neg.npz` contains:

| Key | Meaning / shape |
| --- | --- |
| `width_nm` | Best restored physical widths `[n,n]`, or empty `[0,0]` when skipped |
| `psf_raw` | Raw intensity `[1,n,n]` |
| `target` | Rotated, scaled, nonnegative target `[1,n,n]` |
| `active_gray` | Boolean active-channel mask `[1]` |
| `gain_gray` | Least-squares numerical gain `[1]` |
| `wavelength_nm` | Scalar `450` |
| `history` | Per-step shape losses (empty when skipped) |

`config.json` records arguments, `patch_meta`, `grayscale=true`,
`wavelength_nm=450`, target rotation, zero-channel caveat and phase-library
identity. No grayscale key claims RGB. `summary.json` records branch index,
sign, skipped state, best loss and step count. One output normally fits two
signed branches; a full 32-output input has up to 64 separately fitted devices.

## Sampling, gain and interpretation limits inherited from the source

The target patch is still
`int(patch_camera * camera_pixel_um * 1000 / dx_nm)`: default 51 simulation
samples, spanning 20.4 um rather than the ideal 20.7 um. The 800-point simulation
width is 320 um (about 92.75 camera pixel spacings). Per-side propagation padding
is still `n`, so the default FFT field is 2400 by 2400; a full fit can be costly.

Camera pixel pitch only sets the target support here. There is no camera pixel
area integration, pixel-response model, sensor-size crop, or tiled-device
crosstalk model. Nearest interpolation does not establish equivalence to a
3-by-3 digital convolution. Retained optical assumptions are a P400/H600,
80-320 nm width, phase-only fitted response and homogeneous free-space angular
spectrum propagation. The copied phase module has historical docstring mentions
of 360 nm; the actual retained optimization width map is 80-320 nm. Material,
geometry, transmission amplitude, spectral bandwidth and glass/air interfaces
would need separate modeling and calibration.

The exact numerical gain is
`sum(psf * target) / max(sum(psf**2), 1e-12)` per channel. Shared scales and gains
support numerical comparisons; they do not prove physical brightness, exposure,
detector calibration, optical efficiency or task accuracy. Ignoring a zero
branch in the loss does not physically extinguish a device: its path must be
blocked or its contribution calibrated. Signed-input encoding, signed-output
subtraction and calibrated readout are not implemented here. Kernel rotation
aligns optical convolution with PyTorch correlation, but sampling, stride,
padding, dilation, crop and electronic bias still require explicit alignment.
The exported width map is not a digital convolution weight, and an 800-by-800
PSF cannot directly replace a 3-by-3 network kernel.

## Original fitting task validation boundary

The original fitting task used source inspection and static file/diff/hash checks only. No
Python runtime, runtime imports, fitting, numerical tests, real data, network
replacement, installation, training, or hardware validation was performed.
Independent read-only review is owned by the parent agent on the terminal
snapshot. Static consistency does not establish executable or numerical success.


## Independent tilted-plane-wave isoplanatism validation

`validate_isoplanatism.py` reads fixed physical width maps and never calls the
fitter, optimizer or backward. Run from this directory in newconda:

```powershell
python validate_isoplanatism.py --artifact <kernel_000_pos.npz> --config <config.json> --out <fresh_external_directory> --device cpu
python validate_isoplanatism.py --self-check --angles -0.5 0 0.5 --out <fresh_external_directory>
```

An output destination must be explicitly supplied, outside the source checkout,
fresh, and disjoint from inputs. Real input and persistent output destinations
are not supplied for this task: real validation remains
`BLOCKED_MISSING_OPTIMIZED_ARTIFACT`. No server access is implied.
`--axes x y`, `--angles`, `--paddings`, `--roi-size` are configurable.
Real runs include source baseline padding (normally n) and default to n and 2n.
A single padding is permitted but recorded NOT_CHECKED. Source config must include
n, dx_nm, distance_um, wavelength_nm, camera_pixel_um, patch_camera and patch_meta;
patch must match the original int(camera pitch * patch / dx) calculation.
Missing padding is derived from the locked source entry, never an ASM default.
Empty-width zero branches are skipped with a reason and input hash.

Every angle, including zero, uses independent original ASM propagation on one
fixed field. Extended planes embed only zero amplitude outside the physical
aperture; central crops are compared to the original padding call. No source
physics files or fitting settings are changed. Arrays are [y,x], physical +y
is down with origin upper. Measured d is off-axis displacement; translation
for alignment is -d, implemented by sampling at coordinates +d with bilinear
zero extension and an explicit mask. Postprocessing transfers tensors to CPU;
ASM alone runs on the selected device. Integer ROI following rounds to nearest
(ties to even), clips the requested intersection, reports coverage and never
recenters a partially lost region.

Outputs: per-angle NPZ (raw/extended complex fields, reference and off-axis
intensities, measured/theory aligned arrays and masks), common-scale PNGs,
metrics CSV/JSON, and manifest with input/source hashes, physical parameters,
versions, resources, engineering tolerances and limitations. Energy is raw
sum(I)*dx^2, not per-angle normalized. Shape is L2-normalized on the common
reference support and valid mask; squared error is separate. Undefined Pearson
or insufficient overlap is null with a reason. Shape changes are continuous
physical findings, never numerical-invalid merely for exceeding a threshold.
No application acceptance threshold was specified, so there is no real-device
isoplanatism PASS/FAIL. Diagnostic tolerances are prospective engineering flags,
not claims derived from earlier prototypes. Wrapped transfer-function phase
steps cannot rule out advances above pi; no warning does not exclude aliasing.
Two paddings and edge diagnostics are evidence, not a proof of no aliasing.

The self-check is a separate 64-sample, 400 nm, 80 um ideal focusing fixture,
with smooth finite amplitude, not an optimized 800-sample device. It exercises
real ASM, zero and signed X/Y angles, central crop consistency and two paddings.
Tests also include analytical shifts, undefined metrics, clipped ROI and an
ASM truncation stress fixture. Pytest products belong in system temporary
storage; no persistent experiment is implied by them. The original normal-
incidence phase library does not model angular unit-cell electromagnetic
response. z*tan(theta) is an approximate geometric reference. CMOS integration,
OLED source geometry/coherence and hardware field coverage remain unvalidated.
Independent Level 3 review must be arranged by the parent coordinator.


Continuation engineering evidence (2026-11-04, newconda, disposable system temp):
`python -m pytest -q -s tools/psf_conversion/tests/test_isoplanatism.py -p no:cacheprovider --confcutdir=tools/psf_conversion/tests --basetemp <system-temp>/cases`
returned **18 passed in 8.24s** with OMP/MKL threads limited to one and bytecode
writing disabled. Initial defect reproduction: 3 failed / 6 passed, then focused
completion: 16 passed, final affected suite: 18 passed. Final preflight CPU 10.4%,
RAM 59.2%, visible GPU compute 5%, VRAM 1272/6144 MiB. The ideal fixture produced
12 rows (X/Y, -0.5/0/+0.5 degrees, paddings 64/128), maximum shift error
0.0072278028546357564 pixels, direct/extended crop error 0, and consistent
padding diagnostics. The +12 degree truncation fixture gave output/extended
energy 0.0015057595387051557 and was flagged truncated. CUDA propagation,
carrier and CPU postprocessing paths passed on a 16-sample fixture.
These are engineering results, not formal optimized-device or hardware evidence.
Temporary NPZ/PNG/CSV/JSON/manifest products were removed automatically after
pytest; no persistent simulation destination was selected. Missing inputs are
explicitly tested to fail without creating an output directory. JSON uses null
for undefined metrics; CSV uses an empty cell for a scalar missing value and
JSON null inside structured cells. Independent Level 3 acceptance is pending.


### FIX-01: cross-padding comparisons and unique artifacts

The previous 18-pass engineering record above describes the pre-fix snapshot.
FIX-01 addresses only the two independently reported blocking issues; original
fitting, phase and ASM files remain unchanged.

The baseline is explicitly `config.padding` for optimized artifacts and `n=64`
for the ideal fixture, regardless of the order of `--paddings`; it must be
included. Every other requested padding is compared to that baseline for each
artifact/axis/exact angle. Direct shape distance is
`||P_candidate/||P_candidate||2 - P_baseline/||P_baseline||2||2` over the entire
fixed n-by-n center crop at identical coordinates and dx, without registration
or a positive-baseline mask. The existing 0.01 engineering budget now applies
to this direct distance, not the difference between two relative-to-reference
errors. Raw output energy and measured displacement differences remain extra
checks with unchanged limits. Zero/nonfinite PSFs or missing diagnostics cannot
be consistent. This is a finite-window stability check, not proof of no aliasing.

`padding_stability` is an explicitly labeled case-level summary aggregating
all baseline-vs-candidate pairs. Any inconsistent pair makes the case
inconsistent; undefined/missing comparisons are indeterminate, and a single
padding is NOT_CHECKED. Each pair records paddings, NPZ filenames, direct metric
definition/value, energy/shift diagnostics, status/reasons and comparison ID.
Each row identifies its own padding and `padding_comparison_ids`; a baseline
self-comparison never counts as convergence. Truncation consumes the aggregate
status, so inconsistent/unchecked padding cannot yield not_detected on that
basis. Only two saved center-crop PSFs are loaded per pair; extra extended
complex fields are not retained for this fix.

Every actual propagation receives a deterministic unique `sample_id` across
the run. Metrics store original-precision `angle_deg`, `sample_id`, `npz_file`
and `png_file`; NPZ also stores angle, padding and ID. Use these recorded paths,
not rounded angle strings, to locate evidence. Close angles such as 0.1 and
0.10000001 remain distinct. All file writes use exclusive creation and reject
unexpected collisions. Cross-padding references use the same saved NPZ mapping.


FIX-01 validation (2026-11-04, newconda, single-thread CPU): three new regressions
failed on the old snapshot (equal-energy shape incorrectly consistent, middle
padding omitted, close-angle file overwrite). After the fix the same focused
selection returned 3 passed / 18 deselected in 4.89s. The affected command
`python -m pytest -q -s tools/psf_conversion/tests/test_isoplanatism.py -k "not cuda" -p no:cacheprovider --confcutdir=tools/psf_conversion/tests --basetemp <fresh-system-temp>/cases`
returned 24 passed / 1 deselected in 11.87s, including independent recomputation
from recorded NPZ paths, three-padding order invariance, undefined PSF rejection,
collision preservation, fixture-baseline enforcement and original small ASM.
Final run preflight: CPU 10.9%, RAM 66.4%, GPU compute 2%, VRAM 1252/6144 MiB;
CUDA was deliberately deselected. Temporary output was isolated outside source
and cleaned by TemporaryDirectory. No persistent/800-grid/real-device simulation
was performed. Real validation remains BLOCKED_MISSING_OPTIMIZED_ARTIFACT;
FIX-01 is pending independent focused re-review, not final acceptance.
