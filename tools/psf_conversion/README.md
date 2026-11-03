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

## Validation boundary

This task used source inspection and static file/diff/hash checks only. No
Python runtime, runtime imports, fitting, numerical tests, real data, network
replacement, installation, training, or hardware validation was performed.
Independent read-only review is owned by the parent agent on the terminal
snapshot. Static consistency does not establish executable or numerical success.
