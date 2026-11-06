# Selecting a feature-extraction preset and data-processing transforms

For agents (and humans) choosing `feature_extraction_name`, `feat_ext_transform` and
`data_proc_transforms` for a dataset. Read `arithmetic.md` for the formulas this file refers to.

**Do not copy preset values into notes. Read them from the code:**
`FEATURE_EXTRACTION_PRESET_DESCRIPTIONS` in
`tinyml-modelmaker/tinyml_modelmaker/ai_modules/<timeseries|audio|vision|radar>/constants.py`.
Transform behaviour is in
`tinyml-tinyverse/tinyml_tinyverse/common/datasets/{timeseries,image,audio}_dataset.py`.
These files are the source of truth; if they disagree with this document, the code wins and this document needs fixing.

```python
from tinyml_modelmaker.ai_modules.timeseries.constants import FEATURE_EXTRACTION_PRESET_DESCRIPTIONS as P
for name, p in P.items():
    d = p["data_processing_feature_extraction"]; print(name, p["common"]["task_type"], d)
```

## 1. What to know before choosing

| Input | How to get it | Why it matters |
|---|---|---|
| task type | user / dataset layout | decides the dataset class, and which transforms even run (arithmetic.md s7) |
| rows per file R | dataset analysis (min, median, count of files) | hard minimum `F*C` per file |
| columns V | dataset analysis (after dropping time columns and header) | must equal the preset's `variables` or be overridden |
| sampling rate fs | user / sensor datasheet | converts bins to Hz and frames to seconds |
| signal character | user | stationary vs transient decides FFT vs RAW, and frame length |
| event duration | user | window `F*C/fs` must cover it |
| classes, files per class | dataset analysis | split and balance |
| target device | user | MSPM0 means the Q15 chain; others use the float chain |
| model | modelzoo | input must equal `(ch, wl, 1)` |

## 2. Procedure

1. **List candidates.** Take presets of the task's module whose `common.task_type` matches (it is documentation only, never enforced, so also accept `Generic_*` or `Custom_*` presets when the task is generic). Drop presets for other applications (ArcFault, PIR, WiFiCSI, ECG, HandGesture, FallDetection, MotorFault) unless the dataset is that application.
2. **Device filter** (pass `target_device` to the recommendation tool, which then excludes the others). MSPM0 devices: only Q15 presets (`FFT_Q15` in the transform list; names with `FE_RFFT` or `_MSPM0`, plus `PIR..._FixedPoint`, `ECG2500Input_Roundoff_1Frame`). Other devices: float presets. The device libraries lack several flags (section 7).
3. **Apply the hard gates** (section 3) to every candidate using each file's R. A candidate that skips most files is out. Report how many files each survivor skips.
4. **Rank survivors** (section 4).
5. **If nothing fits**, either override one or two keys on the nearest preset (smaller `frame_size`, smaller `num_frame_concat`) or build a `Custom_Default` pipeline from the standard chains (section 6). Config keys always override preset values.
6. **Verify.** Run `validate_feat_ext_data_shape` (it applies these gates, reports `rows_check` and the model input shape). Then run the dataset prepare step (or set `dont_train_just_feat_ext`) on the chosen config and compare the resulting `X.shape` with `fit()`'s prediction, and compare it with the model's expected input. Do this before training.
7. **Explain the choice**: for the winner and runner-up, show frames/windows per class, input shape, window duration, band edges in Hz.

## 3. Hard gates (all must pass)

- **G1.** `F` is a power of two (FFT chains). Device GUI limits are 32..1024.
- **G2.** Every file used satisfies `R >= F*C` (or the SimpleWindow bound). Count the failures.
- **G3.** If binning: `1 <= B` and `B*A <= F/2`; prefer `(F/2) mod (B*A) == 0`.
- **G4.** `C > 1` implies `'CONCAT'` is in the list.
- **G5.** `LOG_DB` is preceded by `ABS` (or a magnitude step) and `log_mul`, `log_base`, `log_threshold` are positive (unset, released tinyverse 1.5.0 defaults them to 20, 10, 1e-100; set them explicitly anyway).
- **G6.** `V` equals the number of data columns and the stacking choice produces the model's input `(ch, wl, 1)`.
- **G7.** Transform names are exactly right and in the set the task's dataset class implements. Unknown or wrong-case names are **silently ignored** (e.g. `DownSample` does nothing; `FE_DECIMATE` and `FE_Q15` in `MotorFault_128Input_RAW_...` do nothing).
- **G8.** The chain stays in one family (float or Q15).
- **G9.** Each class has enough files for the file-level split (about 4 or more; rounding not verified).

## 4. Ranking (soft)

1. **Physics fits the signal.** Features must contain the frequencies that distinguish the classes: FFT bin width `fs/F`, band edges `k*bin_size*fs/F`. Pick FFT/FFTBIN when classes differ by spectrum, RAW when shape/time-domain matters or the device cannot spare FFT time.
2. **Window covers the event.** `F*C/fs` at least as long as the shortest event of interest; frame `F/fs` short enough that the signal is roughly stationary inside it.
3. **Sample budget.** Total windows per class are `sum(frames - C + 1)`, but neighbouring windows are 7/8 (for C=8) the same data; the independent sample count is about `R / (F*C)` per file. Prefer settings that leave at least a few dozen independent windows per class and similar totals across classes.
4. **Small model input.** Smaller `wl` means smaller NN and faster inference. Compare `wl*ch` against the device memory budget.
5. **Resolution trade-off.** For a fixed window length `F*C`, larger `F` gives finer frequency resolution and fewer time steps; larger `C` gives more time context at the same `F`.
6. **Preset provenance.** If an example in `tinyml-modelzoo/examples/*/config*.yaml` already uses the preset for a similar dataset, prefer it.

## 5. Golden rules

1. Read the preset values, not the name. Names can be stale (`# ch=3` comments, mismatched sizes). Resolve defaults from `params.py` and check the final `run.yaml`.
2. List order is irrelevant: the code runs transforms in a fixed order (arithmetic.md s3). `DC_REMOVE` before or after `Q15_MAG` in your list changes nothing.
3. `DC_REMOVE` is `x[1:]` on the spectrum; it assumes a spectrum already exists. It is unrelated to the `dc_remove` config key, which only affects `RAW_FE` (and the device always removes the mean for `FE_RAW`, so keep `dc_remove: True`).
4. `normalize_bin` is a mean vs sum in Python, but the MSPM0 device always normalises BIN_Q15. Set `normalize_bin: True` for Q15 chains.
5. `min_bin` changes lengths and the last bin; set it deliberately and report the resolved value (arithmetic.md s5).
6. Keep Python and device identical: anything the device library lacks (section 7) is training-only and cannot be deployed.
7. `frame_skip` only thins windows for `C >= 2`, and thins `X` but not `X_raw`. Use `1` unless you control the downstream pipeline.
8. `offset > 0` is unsafe with `C > 1`. Leave `offset: 0`.
9. `scale` divides all data before everything else. Use it to bring raw ADC counts into range (e.g. 4096.0 for 12-bit), not as a feature.
10. `SimpleWindow` is a data-proc transform that cuts rows into windows before framing. If the feature chain is non-empty it is re-framed anyway; use it when files are one long recording and you want overlap via `stride_size < 1`. With `stride_size` so small that `int(F*s) == 0` it silently becomes non-overlapping.
11. The data-proc names are exactly `Downsample` and `SimpleWindow`; `DownSample` or `Simplewindow` are silently ignored. `Downsample` first divides the rate by `k = int(sampling_rate // new_sr)`; after it, `fs`, `F` and bin edges are in the new rate.
12. Use at most one family of spectral features per pipeline; do not stack `FFT_FE` with `FFT_Q15`.
13. Never use `KURT_FE`, `ENT_FE`, `ZCR_FE`, `DOM_FE`, `SLOPE_FE` on their own (shape check fails). They exist only as parts of `PIR_FE`.
14. `PIR_FE`: one variable, F=125-style frames, `window_count`, `chunk_size`, `fft_size` (multiple of 32) must be set explicitly (modelmaker's defaults are 1). Only use the PIR presets for PIR data. `PIR_FE_Q15` hard-codes 25 windows x 4 chunks and 8-bit unsigned input.
15. `FFT_COL` / `L2_NORM_ROW`: only for multi-column CSI-like data (V >= 2, `FFT_COL` yields `V//2` channels). No device library support.
16. `HAAR`/`HADAMARD` and `RANGE_NORMALIZE` do nothing in Python (the last only emits a flag). `HandGesture256Input_RangeNormalize_1Frame` really is just `scale=4096` plus `SimpleWindow`.
17. Anomaly detection: only the float FFT/RAW chain is executed; Q15, ROUND_OFF, PIR, FFT_COL are ignored. Forecasting: no FE at all, only `SimpleWindow`.
18. Regression: keep `C=1`, `S=1` when using FE, or verify `len(X) == len(Y)`.
19. When the data is already a fixed-size window per file, you may use no FE and only `SimpleWindow`/`scale` (the model sees raw `(V, W, 1)`); files must still hold `>= W` rows.
20. Do not "fix" code quirks from inside a selection; choose settings that avoid them and say so.

## 6. Standard chains

| Goal | `feat_ext_transform` | Per-frame length | Use when |
|---|---|---|---|
| Binned log-spectrum, float | `FFT_FE, FFT_POS_HALF, DC_REMOVE, ABS, BINNING, LOG_DB, CONCAT` | B | default for vibration/sound/current signals on non-MSPM0 (`Generic_*_FFTBIN_*`, `Input256_FFTBIN_*`) |
| Full log-spectrum, float | `FFT_FE, FFT_POS_HALF, ABS, DC_REMOVE, LOG_DB, CONCAT` | F/2 (m=1) | small F where binning is unnecessary (`Generic_*_FFT_*`) |
| Raw samples | `RAW_FE, CONCAT` | F | time-domain shape matters, or no FFT budget (`Generic_*_RAW_*`) |
| Fixed-point spectrum (MSPM0) | `FFT_Q15, Q15_SCALE, Q15_MAG, DC_REMOVE, BIN_Q15, CONCAT` | B | MSPM0 devices (`*_FE_RFFT_*`, `*_MSPM0`) |
| Arc-fault style | `FFT_FE, FFT_POS_HALF, WINDOWING, BINNING, NORMALIZE, ABS, LOG_DB, CONCAT` with `log_base='e', log_mul=10, log_threshold=1e-12` | B | high-rate current signals (`FFT1024Input_*`, `ArcFault_*`) |
| Log magnitude only | `ABS, LOG_DB, CONCAT` | F | tiny F, many variables (`Generic_8Input_ABS_8Feature_1Frame`) |
| Round-off | `ROUND_OFF` | F | ECG-style integer inputs on MSPM0 |
| No FE | `[]` plus `SimpleWindow`, `scale` | raw | regression/forecasting/time-domain NN |

Float log settings used by presets: `log_mul=20, log_base=10, log_threshold=1e-100`. A threshold that small gives -2000 dB on exact zeros; consider `1e-6` if the signal can be digitally silent (changes train/deploy consistency, so apply on both sides).

## 7. Device notes (from reading the feature-extraction libraries; confirm against the current library)

- c28 / c29 / am13 libraries support `FE_WIN, FE_FFT, FE_NORMALIZE, FE_LOG, FE_BIN, FE_RAW, FE_DC_REM, FE_CONCAT`. Hann tables exist for F = 32..1024 (2048 on c28); other sizes fall back to an empty table.
- MSPM0 supports `FE_RFFT, FE_COMPLEX_MAG_SCALE, FE_MAG, FE_BIN, FE_DC_REM, FE_PIR_Q15`.
- No device support: `FE_L2_NORM_ROW, FE_FFT_COL, FE_PIR` (float), `FE_KURT, FE_ENT, FE_ZCR, FE_DOM, FE_SLOPE, FE_RANGE_NORMALIZE`.
- Compile step may force `SKIP_NORMALIZE`/`OUTPUT_INT` by target; with `skip_normalize` and no CONCAT it forces `FE_CONCAT` with 1 frame.
- Device and model compatibility is by convention in the Model Composer filters (`tinyml-mlbackend/model_composer_extensions/scripts/*.py`), not in modelmaker.

## 8. Non-timeseries modules (short)

- **Audio** (`audio/constants.py`): presets set `audio_feature` (MFCC or LPC), `sampling_rate`, clip length, `n_mfcc`/`n_mels` or `lpc_order`/`nlpc`, frame length/step in ms. Check clip duration >= preset duration and the sampling rate matches the device (the `MSPM0_LPC_*` presets mirror the MSPM0G5187 firmware).
- **Vision** (`vision/constants.py`): presets set resize target, channels (GRAYSCALE vs RGB), mean/scale. Match channels and image size to the dataset and model; `CoffeeBean_Augmentation_Default` only adds augmentation.
- **Radar**: one preset (`Mnist_Default`), no named transforms.
- Transform names for image: `RGB, GRAYSCALE, RESIZE, RESIZE_PAD, AUTOCONTRAST, EQUALIZE, INVERT, CLAHE, SOBEL, LAPLACIAN, BINARIZE` (applied in the order listed, unlike timeseries); augmentations `RANDOM_HORIZONTAL_FLIP, RANDOM_VERTICAL_FLIP, RANDOM_ROTATION, COLOR_JITTER`. Timeseries augmenters (`training.augment_config`): `AddNoise, Convolve, Crop, Drift, Dropout, Pool, Quantize, Resize, Reverse, TimeWarp`; an unknown name raises `KeyError`.

## 9. Answer template for the user

State: chosen preset (or Custom pipeline) and any overrides; per-class files, skipped files, windows; model input shape; window duration and band edges in Hz; why the runner-up lost; open risks (min_bin, class imbalance, files per class).
