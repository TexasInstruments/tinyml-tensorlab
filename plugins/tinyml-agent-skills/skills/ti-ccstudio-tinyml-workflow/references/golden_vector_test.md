# Golden-vector test (Step 14C, C5, Gate G4b: mandatory for every BYOD run)

For a BYOD run, the project must provably run a golden-vector test of the trained model before it is built. This file covers adding the test when the template has none, and (section 0) verifying the wiring in every case. Bundled examples used as-is do not use this file.

Some SDK starters (e.g. MSPM0 `hello_world_ai`) have no on-device check, but every training run still produces golden vectors. This adds a model-level test to the project.

**One route only: the test function goes directly into `main.c`, like `application_main.c` in the C2000 templates.** The test vectors stay in `test_vector.c`, copied next to `main.c`. The projects this step applies to never have application logic in `main.c`: an exact match runs as-is (no C5), and the only hosts are generic templates and blank or demo starters (`empty_*`, `hello_world_ai*`). If the file under test has real application code, stop and ask the user; do not add a separate file. Needs the user's OK from Step 3B.

## 0. Verify the wiring (Gate G4b, always, after adding or finding the test)
Save as `gv_verify.py` in a temp dir and run `python3 gv_verify.py "$PROJECT_PATH"`. It must print `PASS` before Step 14D. It checks the project, outside `run/`, `Debug/` and `Release/`:

```python
import os, re, sys
root = sys.argv[1]
src = {}
for d, dirs, files in os.walk(root):
    dirs[:] = [x for x in dirs if x not in ("run", "Debug", "Release", ".git")]
    for f in files:
        if f.endswith((".c", ".h")):
            src[os.path.join(d, f)] = open(os.path.join(d, f), errors="ignore").read()
errs = []
callers = [p for p, t in src.items() if p.endswith(".c") and ("tvmgen_default_run" in t or "golden_vector_test" in t)]
main = [p for p in callers if re.search(r"\btest_result\b", src[p])]
if not main: errs.append("no .c file that runs the model defines/uses test_result")
for p in main:
    t = src[p]
    if not re.search(r"\b(int|void|uint8_t|bool)\s+\w*(golden|gv)\w*\s*\(", t) and not re.search(r"\btest_result\s*=", t):
        errs.append(f"{p}: no golden-vector test function or test_result assignment")
    body = re.sub(r"/\*.*?\*/|//[^\n]*", "", t, flags=re.S)
    fn = re.findall(r"\b(?:int|void|bool)\s+(\w*(?:golden|gv)\w*)\s*\(", body)
    if fn and not any(len(re.findall(rf"\b{f}\s*\(", body)) >= 2 for f in fn):
        errs.append(f"{p}: golden-vector test function {fn} is defined but never called")
    for inc in ("tvmgen_default.h", "user_input_config.h"):
        if inc not in t:
            errs.append(f"{p}: does not include {inc}")
tv = [p for p in src if p.endswith("test_vector.c")]
if not tv: errs.append("test_vector.c is not in the project")
else:
    t = src[tv[0]]
    for sym in ("raw_input_test", "model_test_input", "golden_output"):
        if sym not in t: errs.append(f"test_vector.c lacks {sym}")
if not any(p.endswith("user_input_config.h") for p in src): errs.append("user_input_config.h is not in the project")
if not any(p.endswith("tvmgen_default.h") for p in src): errs.append("tvmgen_default.h is not in the project")
print("FAIL: " + "; ".join(errs) if errs else "PASS: golden-vector test is present and wired"); sys.exit(1 if errs else 0)
```
Also read the call site yourself: the call sits after device (and NPU) init and before the idle loop. The script cannot judge that. Then tell the user what you verified.

## When to apply (adding the test)
After C4, apply it only if **both** hold:
- `$GV/test_vector.c` exists (the training run produced golden vectors), and
- no project source defines `test_result`: `grep -rn "test_result" "$PROJECT_PATH" --include=*.c --include=*.h` (excluding `run/`) finds nothing.

If the project already has a `test_result` check, C3 replaced its vectors and you are done. Find `MAIN_SRC`, the file that calls the model, with `grep -rln "tvmgen_default_run" "$PROJECT_PATH" --include=*.c` (not `compilation/` or `training/`). Its name varies (`application_main.c`, `hello_world_ai_npu.c`, ...); do not assume it.

## 1. Check the preconditions (each must print PASS; on FAIL stop and tell the user, do not guess)

**1a. The SDK's feature-extraction library must provide what the config's flags need.** Check every flag in `$GV/user_input_config.h` (e.g. `FE_FFT`, `FE_BIN`, `FE_LOG`, `FE_CONCAT`, `OUTPUT_INT`). `SKIP_NORMALIZE` is not checked here: whether the input is quantized is decided from `tvmgen_default.h` in 1c. Save as `fe_check.sh` and run `bash fe_check.sh "$SDK_ROOT" "$GV/user_input_config.h" "<HDR_PATH>"`. It also prints `FE_QUANT_FN=<name or none>`: the SDK's quantization function that matches the header's class (`FE_batchNormalization_npu` or `_cpu`), or `none`. This is informational. Section 2 does not call that function; the library's own feature extraction applies it.

```bash
# usage: fe_check.sh <SDK_ROOT> <user_input_config.h> <tvmgen_default.h>
SDK_ROOT="$1"; CFG="$2"; HDR="$3"
FE=$(grep -rlE "FE_runFeatureExtract|FE_Skip_Normalize" "$SDK_ROOT" --include='feature_extract*.c' 2>/dev/null | head -1)
[ -n "$FE" ] || { echo "FAIL: no feature-extraction library found in the SDK"; exit 1; }
LIB="$(dirname "$FE")"; missing=""
for flag in $(grep -E '^#define +[A-Za-z0-9_]+ *$' "$CFG" | awk '{print $2}' | grep -v '_H_$' | grep -v '^SKIP_NORMALIZE$'); do
  grep -qw "$flag" "$LIB"/feature_extract.* || missing="$missing $flag"
done
[ -z "$missing" ] || { echo "FAIL: $FE is missing:$missing"; exit 1; }
if grep -q "TVMGEN_DEFAULT_BIAS_LEN" "$HDR"; then Q=FE_batchNormalization_npu
elif grep -q "input_reciprocal_scale_data" "$HDR"; then Q=FE_batchNormalization_cpu
else Q=none; fi
SIG=""; [ "$Q" = none ] || SIG=$(grep -hE "^void[[:space:]]+$Q\(float[[:space:]]*\*[[:space:]]*input_buffer" "$FE" | head -1)
if [ -n "$SIG" ]; then echo "FE_QUANT_FN=$Q"; echo "FE_QUANT_SIG=$SIG"; else echo "FE_QUANT_FN=none"; fi
echo "PASS: $FE has every flag the config needs"
```

On FAIL (no library, or a flag the library lacks) do not generate the test. Tell the user which flags are missing. On MSPM0, the library has `FE_Skip_Normalize` (q15, not float input) and no `FE_batchNormalization_*`, so the function check prints `FE_QUANT_FN=none`. Section 2 does not quantize by itself either way. The flag check still fails there, because `FE_CONCAT`, `OUTPUT_INT`, and `FE_NORMALIZE` are not in that library. The golden-vector test is therefore unavailable for MSPM0 examples that use those flags.

**1b. The vectors must match the config.** The test needs the vector sizes to match the config. Save this as `gv_check.py` in a temp dir and run `python3 gv_check.py "$GV" "$GV/user_input_config.h"`:

```python
import re, sys
gv, uic = sys.argv[1:3]                 # golden_vectors dir, user_input_config.h
tv = open(f"{gv}/test_vector.c").read(); cfg = open(uic).read()
def macro(n):
    m = re.search(rf"#define\s+{n}\s+(\d+)", cfg); return int(m.group(1)) if m else None
def arr(n):
    m = re.search(rf"\b{n}\s*\[\s*(\d+)\s*\]", tv); return int(m.group(1)) if m else None
errs = []
need_in = (macro("FE_STACKING_CHANNELS") or 0) * (macro("FE_STACKING_FRAME_WIDTH") or 0)
if arr("model_test_input") != need_in: errs.append(f"model_test_input has {arr('model_test_input')} values, config implies {need_in}")
if arr("golden_output") != macro("FE_NN_OUT_SIZE"): errs.append(f"golden_output has {arr('golden_output')} values, FE_NN_OUT_SIZE={macro('FE_NN_OUT_SIZE')}")
print("FAIL: " + "; ".join(errs) if errs else "PASS: vector sizes match the config"); sys.exit(1 if errs else 0)
```

**1c. The header decides the input branch.** Quantization is decided by `tvmgen_default.h`, not by `SKIP_NORMALIZE` (a feature-extraction setting, not a model property). Save this as `gv_quant.py` in the same temp dir and run `python3 gv_quant.py "<HDR_PATH>"`. It prints `GV_QUANT=npu|cpu|none` and `GV_IN_T=int8_t|uint8_t|float`, and fails if the header's `Inputs:` dtype disagrees with the class (int8 for npu, uint8 for cpu, float32 for none):

```python
import re, sys
# usage: gv_quant.py <tvmgen_default.h>
hdr = open(sys.argv[1]).read()
if "TVMGEN_DEFAULT_BIAS_LEN" in hdr:
    quant, in_t, want = "npu", "int8_t", "int8"
elif "input_reciprocal_scale_data" in hdr:
    quant, in_t, want = "cpu", "uint8_t", "uint8"
else:
    quant, in_t, want = "none", "float", "float32"
m = re.search(r"Inputs:\s*\*?\s*Tensor\[\([^)]*\),\s*(\w+)\]", hdr)
dt = m.group(1) if m else "unknown"
print(f"GV_QUANT={quant}")
print(f"GV_IN_T={in_t}")
if dt != want:
    print(f"FAIL: Inputs dtype is {dt}, {quant} header needs {want}"); sys.exit(1)
print(f"PASS: input dtype {dt} matches {quant}")
```

Section 2 does not write its own quantizer. The SDK's feature-extraction library does the normalization, and it does so only when `SKIP_NORMALIZE` is defined in `user_input_config.h`. So the check is:

| 1c `GV_QUANT` | `SKIP_NORMALIZE` in `user_input_config.h` | Meaning |
|---|---|---|
| `npu` or `cpu` | must be defined | the library quantizes; the model input is int8 or uint8 |
| `none` | must not be defined | the library passes floats; the model input is float |

If the two disagree, stop. Measured on the C2000 SDK: with `SKIP_NORMALIZE` set and no quant params in the header, `feature_extract.c` does not compile (`tvmgen_default_input_reciprocal_scale_data` undeclared). The three ECG headers (`ecg_f28p55`, `ecg_anomaly_f28p55`, `ecg_cls_f28p55_model`) are in this state, so their configs need the flag removed before the test can build.

## 2. Add the files
Copy `$GV/test_vector.c` and `$GV/user_input_config.h` into `MAIN_SRC`'s directory. Do not create a separate test file: the test function below goes into `MAIN_SRC` itself (section 3). Change the `#include "tvmgen_default.h"` line in the function to the path `MAIN_SRC` uses (e.g. `"model/tvmgen_default.h"`).

The test runs the same three stages as the SDK's generic examples (`libraries/ai/examples/generic_timeseries_*/application_main.c`):

1. **Feature extraction, test mode.** `raw_input_test` goes through the SDK's feature-extraction library with `test_feature_extraction = true`. The result is compared with `model_test_input` (±1). This checks the feature pipeline, before normalization.
2. **Feature extraction, normal mode.** The same raw input goes through the library with `test_feature_extraction = false`. The library applies normalization itself, and only when `SKIP_NORMALIZE` is defined (1c). The result is the model input.
3. **Model.** The model input runs through `tvmgen_default_run`, and the output is compared with `golden_output`. The tolerance is set per task in `GV_TOL` (±1 for classification, regression, and forecasting; ±4 for anomaly detection, as the SDK example does).

The NPU wait is `extern volatile int32_t tvmgen_default_finished;` followed by `while (!tvmgen_default_finished);`, as in the SDK examples. The test does not write that flag.

```c
/* Golden-vector test, same three stages as the SDK generic_timeseries examples:
 * raw input -> feature extraction (checked against model_test_input)
 * -> model input -> model (checked against golden_output). */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "tvmgen_default.h"
#include "user_input_config.h"
#include "feature_extract.h"

#define GV_TOL  1            /* output tolerance; anomaly detection uses 4 */
#define PASS    1
#define FAIL    0

extern float raw_input_test[];       /* test_vector.c (active SET) */
extern float model_test_input[];
extern model_output_t golden_output[];

int test_result = FAIL;              /* 1 = pass, 0 = fail (watch in the debugger) */
int gv_feature_mismatches = -1;      /* -1 = test has not run */
int gv_output_mismatches = -1;

#define GV_FE_OUT_LEN  (1 * FE_STACKING_CHANNELS * FE_STACKING_FRAME_WIDTH * 1)
static model_input_t gv_fe_test[GV_FE_OUT_LEN];   /* features, test mode */
static model_input_t gv_model_input[GV_FE_OUT_LEN];
static model_output_t gv_model_output[FE_NN_OUT_SIZE];
static float gv_scratch[FE_FRAME_SIZE * 4];

/* Runs the SDK feature extraction once; test = true copies the pre-normalization features. */
static void gv_run_fe(float *raw, model_input_t *out, bool test)
{
    feature_extraction fe;
    feature_extraction_handle h = &fe;
    h->scratch_buffer = &gv_scratch[0];
    FE_allocFeatureExtract(h);
    h->input_buffer = raw;
    h->history_buffer = &model_test_input[0];   /* FE_init reads this as float history */
    h->output_buffer = out;
    h->test_feature_extraction = test;
    FE_initFeatureExtract(h);
    FE_runFeatureExtract(h);
}

int golden_vector_test(void)
{
    int i;

    /* 1. features vs model_test_input */
    gv_run_fe(&raw_input_test[0], gv_fe_test, true);
    gv_feature_mismatches = FE_compareModelInput(gv_fe_test, (model_input_t *)model_test_input);
    printf("Feature extraction mismatches %d\n", gv_feature_mismatches);

    /* 2. model input (normalized when SKIP_NORMALIZE is set) */
    gv_run_fe(&raw_input_test[0], gv_model_input, false);

    /* 3. model vs golden_output */
    struct tvmgen_default_inputs inputs = { (void *)gv_model_input };
    struct tvmgen_default_outputs outputs = { (void *)gv_model_output };
    tvmgen_default_run(&inputs, &outputs);
#ifdef TVMGEN_DEFAULT_TI_NPU
    extern volatile int32_t tvmgen_default_finished;
    while (!tvmgen_default_finished);
#endif

    gv_output_mismatches = 0;
    for (i = 0; i < FE_NN_OUT_SIZE; i++)
        if (fabsf((float)golden_output[i] - (float)gv_model_output[i]) > GV_TOL)
            gv_output_mismatches++;

    if (gv_feature_mismatches == 0 && gv_output_mismatches == 0)
        test_result = PASS;
    printf("Golden vectors matched: %d not matched: %d\n", FE_NN_OUT_SIZE - gv_output_mismatches, gv_output_mismatches);
    printf(test_result == PASS ? "All tests have passed.\r\n" : "Golden-vector test FAILED.\r\n");
    return test_result;
}
```

Notes on the code above:
- `FE_compareModelInput` is the SDK helper (`feature_extract.c:205`). It uses ±1 and compares the `model_input_t` type the library builds with.
- `FE_allocFeatureExtract`, `FE_initFeatureExtract`, and `FE_runFeatureExtract` are called the same way as in `application_main.c`. The scratch buffer must be at least `fe.size_required_by_library`. The example puts it in its own section (`#pragma DATA_SECTION`); copy that pragma if the device needs it.
- Anomaly detection (`anomalydetection`) does not use `FE_compareModelOutput`. Its own loop compares with `fabsf(diff) > 4` and then computes the reconstruction error against `RECONSTRUCTION_ERROR_THRESHOLD`. The template above has no reconstruction check. Add it only when `RECONSTRUCTION_ERROR_THRESHOLD` is defined in `user_input_config.h`.
- The three stages assume the features are float (`model_test_input` is float in every example). That holds for the SDK's test vectors.

## 3. Wire it in
1. Back up `MAIN_SRC` to `$RUN_DIR/compilation/<name>.orig` (that folder is inside `run/`, which is excluded from the build).
2. In `MAIN_SRC`, add the test code from section 2 (the includes, types, arrays, `test_result`, and `golden_vector_test()`) above `main`, and call `golden_vector_test();` **after** device init and, on NPU parts, after the NPU is powered and enabled, and **before** the first `tvmgen_default_run`. If the starter has no `tvmgen_default_run` call, the test's own call is the only model call; resolve any `<YOUR_..._HERE>` placeholders as described in the "Bare starter" section.

## 4. Review before saving
Show the user the diff of `MAIN_SRC` and the new files. If a code-review skill is available (e.g. `superpowers:requesting-code-review`), run it on the diff first. Check: the include path resolves; the call sits after NPU init; no duplicate `main`/`test_result`; types follow `user_input_config.h`. Wait for the user's OK, then have them refresh the project in the IDE (new files are not built until the project sees them) and continue to Step 14D.

## 5. Verify and debug
After flashing, set a breakpoint after the call and watch `test_result` (1 = pass), `gv_feature_mismatches` and `gv_output_mismatches` (both 0 on a pass; -1 means the test did not run), and `gv_model_output` against `golden_output`. Do one negative check: change one `golden_output` value in the project's `test_vector.c`, rebuild and flash, confirm `test_result` is 0, then restore it. A test that cannot fail proves nothing.

| Symptom | Likely cause |
|---|---|
| `test_result` 0, outputs far from golden | model input quantization differs from the model (wrong branch for the header's class, or stale `tvmgen_default.h`); re-run 1c and re-check C2/C4 |
| hangs in `while (!tvmgen_default_finished)` | NPU not powered or interrupt not enabled at the call site; move the call after the NPU setup |
| duplicate symbol `test_result` / `main` | the project already has a check; use it instead |
| header not found / undefined `golden_output` | wrong include path, or `test_vector.c` not in the build (refresh the project) |

For anything not in this table, use `superpowers:systematic-debugging` if available.

## Bare starter (blank `empty_*` / `hello_world_ai*` project)

Used for a demo or blank `main.c` (see the route rule above), including a bare starter approved in Step 3B (rule 3). The test is written **directly into `main.c`** instead of a separate file. The user's approval is required before any edit, because it changes the file.

**Steps (in order):**
1. Run the preconditions in sections 1a, 1b and 1c above. They are the same for every host.
2. Find `MAIN_SRC` (the starter's `main.c`, e.g. `empty_npu_ai.c` or `hello_world_ai.c`). Read it and list what it does: init calls, NPU setup, and whether it calls `tvmgen_default_run`.
3. Back up `MAIN_SRC` to `$RUN_DIR/compilation/<name>.orig`.
4. Write the change (below) into `MAIN_SRC` and copy `$GV/test_vector.c` and `$GV/user_input_config.h` next to it. Do not change the starter's other code.
5. **Show the user the full diff of `MAIN_SRC` and the list of copied files, and ask:** "This writes a golden-vector test into the blank starter's `main.c` (diff above). Apply it?" Wait for an explicit yes. On no, restore the backup and stop without building.
6. After the yes, the normal build and flash apply (Steps 14D, 14E). Run the negative check in section 5.

**What the code in `main.c` must do:**
- Set the test up with the same `golden_vector_test()` function from section 2, pasted into `main.c` (or into a header the user agrees to). Do not add a second `main`.
- **The starter must link the SDK feature-extraction library.** Section 2 calls `FE_allocFeatureExtract`, `FE_initFeatureExtract`, `FE_runFeatureExtract`, and `FE_compareModelInput`. Check that `feature_extract.c` and `feature_extract.h` (from 1a) are in the project, and that `user_input_config.h` is copied next to `main.c`. If the library is not in the project, bring in the SDK's own copy (the one 1a found) in the way this project's layout calls for: copy or link the files, and add them and their include path to the build if the project requires it. SDK starters are laid out differently, so work out the right way from the project and tell the user what you added; if the layout gives no clear answer, ask the user. Never write a second feature extractor.
- Set `test_result` (1 = pass, 0 = fail) as in section 2.
- Call `golden_vector_test()` once, after device init and NPU setup, before the starter's idle loop.
- **If the starter has no `tvmgen_default_run` call** (for example `empty_npu_ai`, which has `#define MODEL_AVAILABLE (0)` and a `<YOUR_INPUT_MAP_HERE>` placeholder), `golden_vector_test()` is the only model call. Include the trained `tvmgen_default.h` from wherever Step 14C placed it (C1 decides the location for a starter that has no placeholder files) and do not leave the placeholder in the file. Set `MODEL_AVAILABLE` to 1 only if the file still uses it to guard the model call.
- **If the starter already calls `tvmgen_default_run`** (`hello_world_ai`), keep that call as-is. The GV test runs first and does not replace it.

**Do not** leave the starter's placeholders (`<YOUR_..._HERE>`) in any compiled line. If a placeholder would remain, stop and tell the user before the build.

## Limits
- It only supports SDKs whose feature-extraction library passes 1a (the config's flags are present). Whether the function is called or the test quantizes itself is decided by 1a and 1c, not by the SDK name.
- The CPU-quantized branch (`input_reciprocal_scale_data` in the header) follows the SDK's CPU conversion, which is unsigned. It is compile-checked only; confirm it on hardware.
- The vector arrays add flash. Remove the test after verification if space is tight.
