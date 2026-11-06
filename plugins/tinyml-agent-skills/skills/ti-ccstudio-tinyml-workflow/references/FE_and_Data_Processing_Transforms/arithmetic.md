# Feature-extraction arithmetic: will this preset fit my data?

Companion to `selection-rules.md`. Everything here is derived from
`tinyml-tinyverse/tinyml_tinyverse/common/datasets/timeseries_dataset.py`
(`BaseGenericTSDataset`, used by classification; regression/AD/forecasting differ, see section 7).
`fe_rules.fit()` (section 8) is tested against the real dataset class.

## 1. Notation

| Symbol | Config key | Default (params.py) |
|---|---|---|
| R | rows in one file (after dropping header row and "time" columns) | - |
| V | `variables` | 1 |
| F | `frame_size` | 1 |
| B | `feature_size_per_frame` | None |
| C | `num_frame_concat` | 1 |
| S | `frame_skip` | 1 |
| A | `analysis_bandwidth` | 1 |
| m | `min_bin` | 1 |
| s | `stride_size` (SimpleWindow only) | 0.01 |
| k | `int(sampling_rate // new_sr)` (Downsample only) | 1 |
| fs | sampling rate of the raw signal in Hz (`sampling_rate` is only metadata unless Downsample is used) | 1 |

## 2. One file, step by step

1. **Load.** Columns whose first-row text contains "time" are dropped. If cell [0,0] is not a float, row 0 is a header and R drops by 1. First `V` columns (or the named/indexed ones) are kept.
2. **Downsample** (if in `data_proc_transforms`): `R' = ceil(R / k)`. `new_sr > sampling_rate` gives k=0 and the file is skipped.
3. **SimpleWindow** (if present): window length F, stride `int(F*s)` (falls back to F when that is 0), `keep_short_tails` is always False. Frames afterwards = `ceil((R' - F + 1) / int(F*s))`; needs `R' >= F`. With stride < F the rows are duplicated, then re-framed non-overlapping.
   Without SimpleWindow: frames = `floor(R' / F)`; the `R' mod F` leftover rows are discarded.
4. **Per frame** (each variable independently): apply the transform chain (section 3). Order is fixed by the code, not by your list.
5. **CONCAT.** Window i = frames `i-C+1 .. i` flattened frame-major. Allowed i: `C-1 .. frames-1`, kept only if `i % S == 0` (absolute index). Windows slide by one frame, so neighbours share (C-1)/C of their data.
   - `frames < C` raises ValueError and the file is skipped (a warning only).
   - `C = 1`: every frame is a sample and `frame_skip` is ignored.
   - `C > 1` without `'CONCAT'` in `feat_ext_transform` gives an empty result (error).
6. **Stack.** `2D1`: shape `(N, V, wl, 1)`. `1D`: `(N, 1, wl*V, 1)`. `wl = per_frame_len * C`.

**Windows per file:** `N = frames - C + 1` for S=1, C>=2; `N = frames` for C=1; otherwise count `i in [C-1, frames-1]` with `i % S == 0`.
**Minimum file length:** `R >= F*C` (exactly one window). Prefer `R >= 10*F*C` for useful sample counts. With SimpleWindow the bound is `R' >= F + (C-1)*int(F*s)`.

## 3. Per-frame length through the chain (classification)

Transforms run in this fixed order regardless of how you list them:
`WINDOWING, RAW_FE, TO_Q15, FFT_Q15, Q15_SCALE, Q15_MAG, FFT_FE, NORMALIZE, ROUND_OFF, FFT_POS_HALF, ABS, KURT_FE, ENT_FE, ZCR_FE, DOM_FE, SLOPE_FE, PIR_FE, PIR_FE_Q15, DC_REMOVE, BIN_Q15, BINNING, LOG_DB` then `CONCAT`, then `FFT_COL`. `L2_NORM_ROW` runs once before framing on the whole `(V, R)` matrix.

| Step | Length out | Notes |
|---|---|---|
| input frame | F | |
| `FFT_FE` | F (complex) | numpy FFT |
| `FFT_POS_HALF` | `F/2 + (m if m is not None else 1)` | m=1 or None: F/2+1 (DC..Nyquist); m=0: F/2 |
| `ABS` | same | drops phase |
| `DC_REMOVE` | previous - 1 | plain `x[1:]` |
| `BINNING` | B | `bin_size = F//2//B//A`; band k = mean (sum if `normalize_bin` false) of `x[off + k*bin_size : off + (k+1)*bin_size]`, `off = m or 0` |
| `LOG_DB` | same | `log_mul * log_base(log_threshold + x)` |
| `FFT_Q15` | 2F (int16) | CMSIS rfft |
| `Q15_SCALE` | 2F | shift by `q15_scale_factor` |
| `Q15_MAG` | F/2 + 1 | |
| `BIN_Q15` | B | same bin_size formula as BINNING; int16 |
| `RAW_FE` | F | subtracts the frame mean if `dc_remove` (default True) |
| `WINDOWING`, `NORMALIZE`, `ROUND_OFF` | F | Hann window / divide by F / round |
| `PIR_FE`, `PIR_FE_Q15` | `(window_count, 25)` 2-D | input must be 1 variable; see `selection-rules.md` |
| `FFT_COL` | `(V//2, N, 1, time_bins)` | needs V >= 2; after CONCAT; ch becomes `V//2` |

Final per-frame length `p`: B if BINNING/BIN_Q15 is present, else the length after the last length-changing step above.
`wl = p * C` (times V for `1D`). **The model input is `(ch, wl, 1)`**: the chosen model must accept exactly this.

Common chains (p for F=32, m=1): FFTBIN `[FFT_FE, FFT_POS_HALF, DC_REMOVE, ABS, BINNING, LOG_DB, CONCAT]` gives p=B.
FFT `[FFT_FE, FFT_POS_HALF, ABS, DC_REMOVE, LOG_DB, CONCAT]` gives p=F/2 (=16). RAW `[RAW_FE, CONCAT]` gives p=F.

## 4. Constraints (hard gates)

Violating one makes files skip, the run abort, or train/deploy disagree.

1. `F` is a power of two (device uses `FE_FFT_STAGES = int(log2 F)`; Q15 CMSIS needs valid sizes). Device GUI limits FFT frames to 32..1024.
2. `R >= F*C` for **every** file you want to use; count the files that fail.
3. `B >= 1` and `B*A <= F/2` (else `bin_size = 0`, sums are empty, NaN after normalise).
4. Ideally `(F/2) mod (B*A) == 0`; otherwise top bins are silently ignored.
5. `C >= 2` requires `'CONCAT'` in `feat_ext_transform`.
6. `LOG_DB` needs real, non-negative input (put `ABS` before it). Its `log_mul`, `log_base`, `log_threshold` must be positive; if unset, released tinyverse 1.5.0 defaults them to 20, 10 and 1e-100 (a source checkout without that guard crashes with a TypeError), so set them explicitly.
7. `Downsample`: `new_sr <= sampling_rate`, both set (`new_sr=None` raises TypeError and crashes the run).
8. Window input width and channels must match the model's expected input.
9. Float chain and Q15 chain are not mixed.
10. Files per class: the default split is `amongst_files` with `(0.6, 0.3, 0.1)` per class, so each class needs at least about 4 files to give every split at least one (rounding not verified).

## 5. The `min_bin` trap

Binning offset `off = m` is applied **after** `DC_REMOVE` has already dropped bin 0. For F=32, B=4 (verified by running it):

| m | length entering BINNING | elements per band |
|---|---|---|
| None | 16 | 4, 4, 4, 4 |
| 0 | 15 | 4, 4, 4, 3 |
| 1 (modelmaker default) | 16 | 4, 4, 4, 3 |

With m=0 or 1 the last band sums one element fewer but is still divided by `bin_size`. It is small, but the device must do the same.
Also `FFT_POS_HALF` with m=0 is one shorter, so the FFT chain without binning has per-frame length `F/2 - 1` (not F/2). Always compute from `m` (and see what `run.yaml` resolved), never from the preset name. The anomaly-detection class treats `m=0` as 1 in its emitted sizes (see section 7).

## 6. Physical quantities

| Quantity | Formula |
|---|---|
| FFT bin width | `fs / F` Hz |
| Lowest frequency seen (after DC_REMOVE) | `fs / F` |
| Highest | `fs / 2` (Nyquist), reduced by `1/A` when A > 1 |
| Band k edges (BINNING, m offset ignored) | `[k*bin_size, (k+1)*bin_size) * fs/F` Hz |
| Frame duration | `F / fs` |
| Window duration | `F * C / fs` |
| Inference rate on device | `fs / (F * S)` (one result per kept frame) |
| First result after | `F * C` samples |
| Raw buffer | `F * V` samples; feature ring buffer `B * C * V` values |

## 7. Differences by task type

| Aspect | Classification | Regression | Anomaly detection | Forecasting |
|---|---|---|---|---|
| Class | `GenericTSDataset` | `GenericTSDatasetReg` | `GenericTSDatasetAD` | `GenericTSDatasetForecasting` |
| Label | parent folder | last column of file | parent folder | target columns, shape `(N, H, T)` |
| FE transforms | all | all except `PIR_FE`, and `1D` stacking | only `WINDOWING, RAW_FE, FFT_FE, NORMALIZE, FFT_POS_HALF, ABS, DC_REMOVE, BINNING, LOG_DB, CONCAT`; others **silently ignored** | **none** (use `SimpleWindow` only) |
| Short file | skipped | skipped | warning only, result may be empty | - |
| Extras | `gain_variations` doubles matching classes | X and Y counts can silently misalign (see below) | `min_bin=0` treated as 1 in emitted sizes | `N = ceil((R' - (F+H) + 1) / int(F*s))`, `H = forecast_horizon` |

- Regression: with `SimpleWindow`, y is averaged over each window. With FE and `C > 1` or `S > 1`, X gets shorter than Y with no check; keep `C = 1`, `S = 1` or verify counts.
- Forecasting: `.npy` input is unsupported (no header row); `scale` divides x only; if `SimpleWindow` is missing or FE is on, windowing does not happen and the run crashes with a generic exception.
- `offset > 0` is only safe with `C = 1`.

## 8. Fit calculator

`scripts/fe_rules.py` implements all of this arithmetic. `tests/test_fe_rules.py` runs it against the real
`GenericTSDataset` (11 framing cases, 6 SimpleWindow cases, the Q15 chain), so it matches the training code.
`get_data_proc_feat_ext_recommendations` and `validate_feat_ext_data_shape` already call it (the ranker's rows gate,
`rows_check`, `computed_output_shape`); call it directly to explore:

```bash
cd $SCRIPTS_DIR && $VENV_PYTHON - <<'PY'
import fe_rules as fr
FFTBIN = ["FFT_FE","FFT_POS_HALF","DC_REMOVE","ABS","BINNING","LOG_DB","CONCAT"]
# one file with 1000 rows, 3 columns: frames, windows, model input shape
print(fr.fit(1000, 32, num_frame_concat=8, variables=3, transforms=FFTBIN, feature_size_per_frame=4))
# fewest rows a file needs (Downsample factor 2, or SimpleWindow with a 16-row step)
print(fr.min_rows_required(32, 8, downsample=2), fr.min_rows_required(32, 8, window_stride=16))
# problems in a pipeline: (errors, warnings)
print(fr.pipeline_issues("generic_timeseries_classification", FFTBIN, frame_size=32, feature_size_per_frame=4, num_frame_concat=8))
PY
```

`fit()` returns `{"frames", "windows", "input_shape", "rows_per_window"}` or `{"skipped": reason}`.
Arguments: `window_stride` (rows) when SimpleWindow is used, `downsample` as the integer factor, and the
`transforms` list decides the per-frame length (section 3). Run it for every file length you care about
(or at least the shortest and a typical one) and sum windows per class.

## 9. Worked example: R=1000, V=3

| | Preset A: F=32, B=4, C=8 | Preset B: F=64, B=8, C=4 |
|---|---|---|
| frames | 31 (992 used, 8 dropped) | 15 (960 used, 40 dropped) |
| per-frame chain | 32 -> 17 -> 16 -> 4 bands (bin_size 4) | 64 -> 33 -> 32 -> 8 bands (bin_size 4) |
| windows (S=1) | 24 | 12 |
| input `2D1` | (3, 32, 1) | (3, 32, 1) |
| input `1D` | (1, 96, 1) | (1, 96, 1) |
| S=2 | 12 windows | 6 windows |
| rows per window | 256 | 256 |

Same model input, same window span, but B has twice the frequency resolution (`fs/64`), half the time steps, and half the samples per file.
