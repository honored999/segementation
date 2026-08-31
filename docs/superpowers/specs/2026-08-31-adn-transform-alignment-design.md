# ADN Transformation Alignment Design

This isolated module implements only the ADN/SrSNet transformation network `T` (tau).  It accepts `[B, C, D, H, W]` tensors and treats `W` / normalized `x` as left-right **model space**.  This is not a claim about an incoming DICOM or NIfTI array: a future external adapter must first use image geometry/orientation to produce this canonical convention.  Such an adapter, I/O, resampling, and real CI-1 validation are outside this change.

`ADNTransformAligner` predicts six `tanh`-bounded values using the official four-stage 3-D encoder (32, 64, 128, 256).  Raw values are scaled into `rx, ry, rz, tx, ty, tz`; default nonzero limits match the official release (`rz=40 degrees`, `tx=0.5`).  Its fixed-pool layer is deliberately replaced by `AdaptiveAvgPool3d(1)` so MRI volumes can have variable eligible shapes.  This is an engineering adaptation, not an ADN-paper configuration.

The sampling matrix passed to `affine_grid` maps normalized output coordinates `(x, y, z)` to normalized input sampling coordinates.  `grid_sample` uses `align_corners=False`, zero padding, and trilinear (`bilinear`) image interpolation by default.  The named left-right helper always flips `W`.  Loss calculation remains independent of inference: it returns flip L1, inverse-warp reconstruction L1, and their sum.
