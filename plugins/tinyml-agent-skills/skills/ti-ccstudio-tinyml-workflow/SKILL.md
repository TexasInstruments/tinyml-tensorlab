---
name: ti-ccstudio-tinyml-workflow
description: Guides users through end-to-end Tiny ML model creation, training, compilation, and device deployment (build and flash from the IDE). Use whenever the user mentions creating AI solutions or AI models, training for embedded devices, deploying to MCUs, flashing a device, memory footprint, or any tiny ML workflow, even if they only name one step. Triggers on any workflow step — dataset analysis, model selection, feature extraction, config creation, training, compilation, deployment. Can start from Step 1 or jump to any intermediate step. If the user has a dataset, config, trained model, compiled artifacts, or is at any point in the workflow, this skill handles it. This skill is the requirements and design process for these requests: use it directly instead of brainstorming or planning skills. Always confirm the tinyml-modelzoo installation before proceeding.
---

# Tiny ML Tensorlab Skill

Build, train, compile and deploy ML models to embedded MCUs with Tiny ML tensorlab.

`runner.py` lives in this skill's `scripts/` directory (`SCRIPTS_DIR`), not in tinyml-modelzoo. Session Setup sets it from the skill's base directory.

## Working with other skills

This skill already runs its own requirements flow (Steps 2–4) and its own plan (the workflow map below). **Do not invoke `superpowers:brainstorming`, `superpowers:writing-plans` or similar planning skills before or alongside it**: they ask design questions this skill answers itself, and they delay the import and example match. Skills that help inside the workflow:
- `superpowers:systematic-debugging`: when training or compilation fails in Step 13, or a build or on-device test fails in Step 14.
- `superpowers:verification-before-completion`: before saying a step is done at gates G2–G5, which already require evidence.
- A code-review skill (e.g. `superpowers:requesting-code-review`): on any source this skill generates for the project (Step 14C, C5).

## IDE and tooling

This skill works in both CCStudio and VS Code. It says **what** must happen (import a project, build it, flash and run it); **how** is up to you, using the tools the IDE gives you (CCStudio: MCP tools; VS Code: CLI project, build and debug tools; TI command-line utilities). Inspect the tools available in the session and use them; do not look for a fixed tool name. If a project operation cannot be automated, tell the user the exact reason and ask them to do that one step in their IDE and report the result; then continue. The one IDE-specific procedure is the build exclusion in Step 14B (CCStudio only).

## How to work through this skill

Later steps depend on the exact outputs of earlier ones. Skipping a check or a copy does not fail loudly; it surfaces later as a linker error or a corrupted binary. So:

1. **Keep a visible checklist.** At the end of Step 1, list the steps you will run in this session (use TaskCreate if available; otherwise print a markdown checklist and re-print it after each step). Mark a step done only after you have shown its evidence (command output or user reply).
2. **Gates are hard stops.** Each gate below has a pass condition you must demonstrate. If it fails, stop and fix; never continue "to see what happens".
3. **After every `runner.py` call**, read `success`. If `false`, read `errors`, fix, and re-run the same step. Never move on while `success` is false.
4. **After each step that writes a config section**, show the user the config so far and continue only on approval.

## Gate table

| Gate | Where | Pass condition (show the evidence) | If it fails |
|---|---|---|---|
| **G0** Template chosen | Step 3B, before the import | `$TEMPLATE_DIR` is an exact `$TASK_TYPE` match, a generic same-task template, a bare starter the *user* approved (its GV test goes into `main.c`), or a template the *user* picked from the golden-vector-host list; on an NPU device with both an NPU and a regular variant, the user explicitly chose between them | No match, an unanswered starter/NPU question: ask the user and wait. Never pick one yourself, never offer another application's example, and never write "proceeding with the NPU template". Do not import until they answer |
| **G0b** Requirements confirmed | Step 3C, before the import (3D) | The README's hardware/software prerequisites (or the generic list if there is no README) were shown and the user replied that they have them or resolved what was missing | No reply yet, or an item missing: ask and wait. Never import, and never skip this because the README lists nothing, the user seems experienced, or hardware was mentioned earlier. Only exception: verified absence of any README in the template |
| **G1** Compiler path | before the training run, Step 13 | CGT root found (via the IDE's tools, or given by the user) and set inline in the *same bash call* that runs `$TRAIN_CMD` | Ask the user for the CGT root. Do not train |
| **G2** Artifacts + footprint | Step 13A, before any Step 14 work | `$RUN_DIR/compilation/artifacts/mod.a` exists and is non-empty; FLASH/SRAM numbers shown; user confirms they fit | Surface the error, fix training/compilation, re-verify. Never deploy |
| **G3** Build exclusions | Step 14B, before copying | **CCStudio:** `.cproject` check prints `PASS` (covering exactly `run`, `config.yaml` and `.tinyml_template.json`), the user confirmed the exclusion list **before** the edit (asked every time), and the user confirms the folders are greyed out. **VS Code:** the user was told these are excluded by default; nothing to edit | CCStudio: fix the XML or exclude in the IDE, re-check |
| **G4** Artifacts replaced | Step 14C, before build | The template's placeholder library and header (and golden vectors, if it has them) were overwritten under their own file names, and `cmp` shows each identical to the trained output | Re-copy, re-compare |
| **G4b** Golden-vector test wired (**BYOD only, hard stop**) | Step 14C, C5, before build | `references/golden_vector_test.md` was read this turn; the project's `main.c` (or the file that calls the model) defines `test_result` and an actually-called golden-vector test using the copied `test_vector.c` / `user_input_config.h` / `tvmgen_default.h`; the `gv_verify.py` check in that reference prints `PASS`; the user approved the diff. Not applicable to a bundled example used as-is | Add or fix the test per the reference and re-verify. **Never go to Step 14D (build) or 14E without this PASS**, and never skip it because the template "looks fine" or the user is in a hurry |
| **G5** Build succeeded | Step 14D, before flash | The build finished with no errors, observed by you or confirmed by the user | Fix the reported error. Do not flash |
| **G6** Result reviewed (BYOD only) | Step 13B, before any Step 14 work | The user answered "Are you happy with this result?" for the latest run, with the metrics and footprint shown. Not applicable to a bundled example used as-is, or to a deploy-only entry with artifacts the user already had | Ask it now and wait. Never go to Step 14, and never skip it because the numbers look fine |

Why G1 is strict: compiler environment variables do not persist across shell tool calls, so an export in an earlier call is gone. If it is empty, Python silently falls back to a hardcoded default path in `constants.py`.

## Workflow map

| Step | What | Sub-steps / notes |
|---|---|---|
| 1 | Session setup | load `.env`, venv, updates, IDE and tooling check |
| 2 | Requirements | task type and device only |
| 3 | Project import | 3A–3D: find SDK, pick template, confirm requirements, import. Sets `PROJECT_PATH` |
| 3.5 | Bundled example match | use as-is / template + own dataset / from scratch |
| 4 | Dataset path and task name | |
| 5 | Common section | |
| 6 | Validate dataset format | |
| 7 | Dataset section | |
| 8A / 8B / 8C | Analyze dataset / feature extraction + data processing / select model | |
| 9 | Training section | includes quantization choice |
| 10 | Testing section | |
| 11 | Compilation section | |
| 12 | Assemble `config.yaml` | |
| 13 | Run training and compilation | G1 |
| 13A | Verify artifacts and memory footprint | G2 |
| 13B | Review the result with the user (**mandatory in the BYOD flow**, skipped otherwise) | G6 |
| 14 | Deploy: 14A verify, 14B exclude (CCStudio only; VS Code: inform), 14C copy, 14D build, 14E flash | G3, G4, G5 |

**Mid-workflow entry.** Do not restart from Step 2 if the user already has something. Session Setup (Step 1) always runs first, then jump:
- "Use an existing example" → Step 3.5 (after the Step 3 import)
- "I have a config" or "just run training" → Step 13
- "I have a dataset, build the config" → Step 4 (after Step 3)
- "I have trained/compiled artifacts, just deploy" → Step 14. Needs `PROJECT_PATH` (do Step 3 if missing), with artifacts under `$RUN_DIR/compilation/artifacts/` and golden vectors under `$RUN_DIR/training/` (`$RUN_DIR` = `$PROJECT_PATH/run/<date-time>/<model_name>`, see Step 13); put the user's files in such a folder if they are elsewhere. Run Step 13A first.

**Reference guides** (read when a step points to them): `references/session_setup.md`, `supported_devices.md` (Step 2), `project_import.md`, `example_matching.md`, `deployment_procedure.md`, `golden_vector_test.md`, `device_deployment_guide.md` (lookups and troubleshooting), `config_creation_guide.md`, `example_running_guide.md`.

---

## Step 1: Session setup (never skip, even when jumping to a later step)

Details and commands for each item: `references/session_setup.md`. Do not re-run `/tinyml-agent-skills:setup` yourself; if `.env` is missing or incomplete, send the user to it.

- [ ] Set `SCRIPTS_DIR` = `<this skill's base directory>/scripts` (the base directory is given when the skill loads; no search)
- [ ] Load `~/.tinyml-agent-skills/.env`; stop if missing/incomplete or `IS_REPO_SETUP` is not `true`/`1`
- [ ] Verify `VENV_PYTHON` exists; set `TRAIN_DIR` / `TRAIN_CMD` from `MODELZOO_INSTALL_MODE`. Use `$VENV_PYTHON` for every python call
- [ ] Detect the IDE (CCStudio or VS Code) and the project, build and debug tools available. A missing tool does not stop training; for import, build and flash the fallback is the user doing that step by hand

Then create the checklist described above.

## Step 2: Understand requirements

Ask only these two questions, and only for what the user has not already told you.

**Do not ask about a dataset, data availability, channel count, or whether to use an example in this step, and do not recap the requirements and wait for a go-ahead.** Those questions belong after the import and the example match: the dataset is asked only after Step 3.5 has presented the ModelZoo examples: in Step 4, or inside Step 3.5 once the user picks an option that needs their own data. Asking here would pull the user into data details before they know whether a ready-made example already covers their case.

1. **Task type.** Read the valid strings from source and let the user pick; do not infer them from domain knowledge:
   ```bash
   grep -A 50 "TASK_TYPE_TO_MODULE" "$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py"
   ```
   For vision also check `$TINYML_MODELMAKER_PATH/ai_modules/vision/constants.py`.
2. **Target device (MUST be a supported device).** Read `references/supported_devices.md` and check the user's device against its Device ID column (ignore case and spaces). Never accept a device from memory or from the user alone.
   - **Exact match:** set `TARGET_DEVICE` to the Device ID exactly as listed (e.g. `F2837xS`), and use the NPU column for the NPU checks later.
   - **Not an exact match but looks like a typo or partial name** (e.g. `F28P5` -> `F28P55`/`F28P551x`; `MSPM0G357` -> `MSPM0G3507`; `AM236` -> `AM263`; a family such as `MSPM0` or `F28P`): do not guess. Ask: "Do you mean `<X>`?" naming the single closest device, or, if you cannot narrow it to one, list the candidates ("Do you mean `<X>`, `<Y>` or `<Z>`?"), and wait for the answer.
   - **No plausible match:** tell the user plainly that the device is not supported, show the supported devices (by family), and ask which one they want. Do not continue with an unsupported device.
   Then confirm the Device ID also exists in the installed modelzoo, which is the source of truth for what can actually run:
   ```bash
   grep -E "TARGET_DEVICE|ALL_TARGET|SUPPORTED_DEVICE" "$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py" | head -40
   ```
   (radar devices are in `ai_modules/radar/constants.py`). If the list and the installed modelzoo disagree, tell the user and use what the installed modelzoo accepts.

Disambiguate when needed (e.g. "motor fault detection" → classify fault vs healthy, or detect anomalies?). Set `TASK_TYPE` and `TARGET_DEVICE`, then start Step 3 in the same turn. No confirmation prompt between Step 2 and Step 3A; the first point where you wait for the user is Step 3C (template prerequisites).

## Step 3: Project import (details: `references/project_import.md`)

Imports an SDK AI-example template into a new project in the IDE; that project holds all config, training and compilation outputs.

- [ ] **3A** Find `$SDK_ROOT` (using the IDE's project tools: CCStudio MCP tools, or the VS Code CLI project tools; latest installed version; ask the user only if the tools cannot find it) and `$AI_EXAMPLES_PATH`
- [ ] **3B** Pick `$TEMPLATE_DIR` (Gate G0, details in `references/project_import.md`): (1) exact `$TASK_TYPE` match; (2) generic same-task template `generic_timeseries_<family>`; (3) a bare starter (`empty_*` or `hello_world_ai*`, matching device family and NPU status), **only with the user's OK**, and then its golden-vector test is written into its `main.c` (Step 14C, C5). If none of these exists, **STOP** and list only the templates that have a golden-vector check (filter in `project_import.md`). Never list or suggest other applications' example projects, and never choose silently
- [ ] **3B-NPU (Gate G0, hard stop)** If `$TARGET_DEVICE` has a hardware NPU (`F28P55`, `MSPM0G5187`, `AM13E2`) and the SDK has both a regular and an NPU variant of the chosen template (names end in `_npu`, e.g. `hello_world_ai` / `hello_world_ai_npu`), **STOP and ask**: "`$TARGET_DEVICE` has an NPU, so the NPU project (`<name>_npu`) is the better fit. Which do you want: NPU project (recommended) or regular project?" End your turn there and wait for the answer. A recommendation is not a decision: do not import, and do not say you are proceeding with either one, until the user has replied. Devices without an NPU have no NPU variants, so skip this
- [ ] **3C (Gate G0b, MANDATORY, NEVER SKIPPED)** Read the template README, present hardware/software prerequisites (accelerators, BoosterPacks, EVMs, sensors, software), ask whether the user has them all, then **end your turn and wait for the reply**. Asked on every import, with no exceptions: not when the user already mentioned hardware, not when the README is short or lists nothing (then ask generically), not on repeat runs. The only exception: you have verified there is no README at all in the template tree, and then you continue silently. Do not start 3D without an answer
- [ ] **3D** Ask for workspace and project name, locate the `.projectspec` with `find "$TEMPLATE_DIR" -name "*.projectspec"` (never build the path from a pattern; the folder and file name vary by SDK/board, see `references/project_import.md`), then import that exact path as a project using the IDE's tools, rename if a generic fallback was used, and confirm the project is registered in the IDE. Save `PROJECT_PATH` and `PROJECT_NAME`. Then, on an NPU device, write `$PROJECT_PATH/.tinyml_template.json` (command in `references/project_import.md`): on an NPU device it holds `template_dir` and `npu_template: true|false`, so the choice survives later steps and sessions

If the import fails because the IDE or its tooling is unavailable, stop and tell the user the reason; ask them to start it or to import the `.projectspec` themselves.

**Non-NPU project on an NPU device.** If `.tinyml_template.json` says `npu_template: false`, the project runs the model on the CPU, so the config must use `training.quantization: 1` (mode 2 targets the hardware NPU) and `compilation.compile_preset_name: forced_soft_npu_preset`. Apply this wherever a config is written (Step 3.5, Step 9C, Step 11) and re-check it in Step 13. If the file is missing or says `npu_template: true`, change nothing.

## Step 3.5: Check for matching bundled examples (details: `references/example_matching.md`)

Run automatically right after the import, with no question to the user first. Never skip silently; always tell the user what was found.

**These are TinyML ModelZoo examples, NOT SDK templates.** "Bundled example" means the ready-made modelzoo example configs (each with its own `config.yaml` and example dataset URL) shipped with this skill in `references/examples/`. It has nothing to do with the SDK template or the project imported in Step 3; that project is only where the outputs go. The agent MUST go through `references/examples/` first (the `find_matching_examples` command does this), take the config and dataset URL/path from the matching example folder, and never describe the match as an SDK template, an imported example, or part of the project. When talking to the user, always say "TinyML ModelZoo example" and state that the dataset comes from that example's config, so they do not confuse it with the imported SDK template. Dataset questions (path, URL, channels) must NEVER be asked before this step has run and the user has answered the options below.

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py find_matching_examples \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\"}"
```

If an `exact` match exists, present it and ask which they want: (1) use the TinyML ModelZoo example as-is, with its own `config.yaml` and example dataset, (2) use its config as a template with the user's own dataset, (3) build from scratch. The dataset is asked only if they pick (2) or (3). With an NPU device and `npu_template: true`, prefer matches whose `model_name` is an NPU model (usually ends in `_NPU`; see the rule of thumb in Step 5). This only ranks which example to show; it never decides whether the template is imported. The template choice is made in 3B-NPU and does not depend on the `model_name` in any config. Options 1 and 2 copy the config to `$PROJECT_PATH/config.yaml`, apply the non-NPU override above if it applies (output paths are set in Step 13), then skip to Step 13; option 3 continues at Step 4. The exact commands are in the reference.

If there is **no `exact` match**, do not offer any example, template or as-is option, even if `same_task` or `same_device` entries exist: **tell the user that no ModelZoo example matches, and go to Step 4 (build from scratch).**

## Step 4: Dataset and task name

Reached only after Step 3.5 found no match or the user chose from scratch; never ask these (or any other dataset question) before the ModelZoo example match has been presented and answered. Ask: dataset location (path or URL), number of sensor channels, and an optional task name. Set `DATA_PATH`, `VARIABLES`, `TASK_NAME`. All files go in `$PROJECT_PATH`; bundled examples are never modified.

Reaching this step means the user is bringing their own dataset, so record it (Step 13B reads it):
```bash
$VENV_PYTHON - "$PROJECT_PATH" <<'PYEOF'
import json, sys
p = f"{sys.argv[1]}/.tinyml_template.json"
try:
    d = json.load(open(p))
except FileNotFoundError:
    d = {}
d["byod"] = True
json.dump(d, open(p, "w"), indent=2)
PYEOF
```

## Step 5: Common section

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_common_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $PROJECT_PATH/common.yaml
```

On success, note `inferred_module` as `TARGET_MODULE`. On failure, show `errors`, check valid values in `$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py`, retry. Optional custom run name: add `"run_name": "{date-time}/{model_name}"`.

## Step 6: Validate dataset format

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py validate_dataset_section \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$DATA_PATH\", \"task_type\": \"$TASK_TYPE\"}"
```

- `success: true` → set `EFFECTIVE_DATA_PATH` to `effective_input_data_path` and use it from now on instead of `DATA_PATH`.
- `success: false` → copy the user's dataset, reorganize the copy per `$SCRIPTS_DIR/constants.py` → `EXPECTED_STRUCTURES[{task_family}]`, set `EFFECTIVE_DATA_PATH` to the copy, re-run.

## Step 7: Dataset section

Ask the user for the split type (`amongst_files` or `within_files`) and split factor (e.g. `[0.6, 0.3, 0.1]`, an example only). If every class has only one data file, `within_files` is required. If the user does not know, omit the `split_type` and `split_factor` keys from the command below and tell them defaults come from `params.py`.

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_dataset_section_yaml \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$EFFECTIVE_DATA_PATH\", \"split_type\": \"<type>\", \"split_factor\": <factor>}" \
  --save-yaml $PROJECT_PATH/dataset.yaml
```

## Step 8A: Analyze the dataset

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py analyse_dataset \
  "{\"formatted_dataset_path\": \"$EFFECTIVE_DATA_PATH\", \"task_family\": \"{classification|anomalydetection|regression|forecasting}\"}"
```

The tool reports the row count of the shortest file as `shortest_sample_len` (also `min_sample_length` / `min_seq_length`), counted the way training counts rows (header row and time columns excluded). It is not a target: every sample needs `frame_size * num_frame_concat` rows (Step 8B), so it limits which presets and which `frame_size` / `num_frame_concat` you may choose.

The tool also reports `num_variables`, the number of data columns in the files (time columns excluded). Compare it with `$VARIABLES` from Step 4: if `$VARIABLES` is larger, stop and correct it; if smaller, confirm the user means to ignore the other columns. Presets that need more channels than `$VARIABLES` are excluded from the recommendations and rejected by the validator.

Persist the result to `$PROJECT_PATH/.tmp_dataset_stats.json` for Steps 8B, 8C and 9, keeping only the `data_distribution` shape for the task family:

```json
{"result": {"dataset_bucket": "<tiny|small|medium|large>", "dataset_size": 0, "task_type": "<TASK_TYPE>",
  "shortest_sample_len": 0, "num_variables": 0, "data_distribution": {"<key>": 0}, "formatted_dataset_path": "<path>"}}
```

`data_distribution` keys: classification `class_<name>` → count; anomaly detection `normal` / `anomalous`; regression `<filename>` → rows; forecasting `<sequence_name>` → entries.

## Step 8B: Feature extraction and data processing

**Part A, recommendations.** Choose the flags with this tree:
- Pattern is in frequency content? **Yes** → `prefer_fft: true`; need the full spectrum → `need_full_spectrum: true`, reduce features with binning → `false`. **No** → `prefer_fft: false`; need temporal context → `need_temporal_ctx: true` (multi-frame), single snapshot → `false`.

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_data_proc_feat_ext_recommendations \
  "{\"task_type\": \"$TASK_TYPE\", \"prefer_fft\": <bool>, \"need_full_spectrum\": <bool>, \"need_temporal_ctx\": <bool>, \"shortest_sample_len\": <shortest_sample_len from .tmp_dataset_stats.json>, \"variables\": $VARIABLES, \"target_device\": \"$TARGET_DEVICE\"}"
```
Add `\"sampling_rate\": <Hz>` when the user knows it: the output then includes window durations in seconds.

**Rows per file, not frame size alone.** A file gives a sample only if it has at least **`frame_size x num_frame_concat` rows** (with `Downsample` factor k: `(frame_size x num_frame_concat - 1) x k + 1`; with `SimpleWindow`: `frame_size + (num_frame_concat - 1) x step`, where step = `int(frame_size x stride_size)`). A shorter file is skipped silently, so the metrics come from fewer files than you think. `frame_size <= shortest` is necessary but not sufficient when `num_frame_concat > 1`.

Before you choose, write one line: `<frame_size> x <num_frame_concat> = <rows needed> <= shortest <S>? yes/no`. Continue only on "yes". Example with shortest = 300:

| preset | rows needed | fits? |
|---|---|---|
| `Generic_256Input_FFTBIN_16Feature_8Frame` (256 x 8) | 2048 | no (a 256-row frame fits, but 8 of them do not) |
| `Generic_32Input_FFTBIN_4Feature_8Frame` (32 x 8) | 256 | yes |
| `Generic_256Input_FFT_128Feature_1Frame` (256 x 1) | 256 | yes |

If your first pick fails, do not move to another large preset; lower `frame_size` or `num_frame_concat`. The tool already does this arithmetic: every preset that fails is in `excluded_presets` with the reason, and `ranked_presets_detailed` gives each survivor's `min_rows_required`, model `input_shape` and `preset_notes`.

**Read `references/FE_and_Data_Processing_Transforms/selection-rules.md` (gates, golden rules, standard chains) and `arithmetic.md` (frames, windows, shapes) before ranking.** They also explain the traps below. Device rule: MSPM0 devices run only the Q15 fixed-point chain (`*_FE_RFFT_*`, `*_MSPM0` presets); other devices only the float chain. Passing `target_device` makes the tool enforce it.

The tool takes its presets straight from that constants file, for the task's module and filtered to `$TASK_TYPE`, so every listed preset exists in the installed version (`preset_catalog_source` in the output names the source). If the tool errors that the catalog cannot be loaded, fix the task type or re-run `/tinyml-agent-skills:setup`; do not fall back to preset names from memory.

Choose only from `ranked_presets_detailed`, or a `Custom_` pipeline whose `frame_size x num_frame_concat` rows fit the shortest file. Never choose anything in `excluded_presets` (the tool lists each one with the reason), and read `frame_size_rule` in the output. Application-specific presets (HandGesture, FallDetection, WiFiCSI, ECG, PIR) are ranked down for generic tasks: take one only if the data is that application.

**No preset fits?** Build a custom pipeline and **leave `feature_extraction_name` unset**: a name you invent is not a registered preset, so modelmaker ignores it. Choose the steps from `references/FE_and_Data_Processing_Transforms/FE_transforms.md`; the tools reject step names training would silently ignore. Example for 3 channels with a shortest sample of 71 rows (frame_size 64 x num_frame_concat 1 = 64 rows, which fits; FFT frames must be a power of two; without `BINNING` the model input has `frame_size/2` = 32 features per frame, so no `feature_size_per_frame` is needed). For `LOG_DB`, set `log_mul`, `log_base` and `log_threshold` explicitly (unset, released tinyverse falls back to 20, 10 and 1e-100, but older source checkouts crash). Set `stride_size` explicitly (its default 0.01 means a 1-row step for large frames):
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py validate_feat_ext_data_shape \
  "{\"data_path\": \"$EFFECTIVE_DATA_PATH\", \"task_type\": \"$TASK_TYPE\", \"variables\": 3, \"data_proc_transforms\": [\"SimpleWindow\"], \"stride_size\": 0.5, \"feat_ext_transform\": [\"FFT_FE\", \"FFT_POS_HALF\", \"DC_REMOVE\", \"ABS\", \"LOG_DB\", \"CONCAT\"], \"frame_size\": 64, \"num_frame_concat\": 1, \"stacking\": \"2D1\", \"log_mul\": 20, \"log_base\": 10, \"log_threshold\": 1e-100, \"shortest_sample_len\": 71, \"target_device\": \"$TARGET_DEVICE\"}"
```
Pass the same parameters (without `shortest_sample_len`) to `generate_feat_ext_section_yaml` in Part C. Transform names are exact and case-sensitive (`Downsample`, `SimpleWindow`, `FFT_FE`, ...): training silently ignores any name it does not know, and the tools reject them.

Understand what each recommended transform does (`references/FE_and_Data_Processing_Transforms/FE_transforms.md`, `Data_processing_transforms.md`; presets are defined in `$TINYML_MODELMAKER_PATH/ai_modules/<module>/constants.py` as `FEATURE_EXTRACTION_PRESET_DESCRIPTIONS`, where `<module>` is the task's module: timeseries, vision, radar or audio). Show the user the full structured output with a summary and **clear point-by-point reasoning** for why each transform or preset suits this dataset.

**Part B, validate:**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py validate_feat_ext_data_shape \
  "{\"data_path\": \"$EFFECTIVE_DATA_PATH\", \"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"data_proc_transforms\": [\"<transforms>\"], \"feature_extraction_name\": \"<preset>\", \"frame_size\": <if SimpleWindow>, \"stride_size\": <if SimpleWindow>, \"shortest_sample_len\": <shortest_sample_len from .tmp_dataset_stats.json>, \"target_device\": \"$TARGET_DEVICE\"}"
```
Pass `shortest_sample_len` so the tool enforces the rows-per-file rule (`rows_check` in the output shows the requirement) and `target_device` so it checks the chain exists in that device's library. Compare `computed_output_shape` with the model's input. If `errors` reports too few rows, a misspelled or missing transform, or a device mismatch, fix it (lower `frame_size` or `num_frame_concat`, correct the name) and re-validate. Read `warnings` too (for example `min_bin` or `stride_size`). Do not continue while `errors` is non-empty.

**Part C, generate:**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_feat_ext_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"data_proc_transforms\": [\"<transforms>\"], \"feature_extraction_name\": \"<preset>\", \"frame_size\": <if SimpleWindow>, \"stride_size\": <if SimpleWindow>, \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $PROJECT_PATH/feat_ext.yaml
```
Pass `frame_size` and `stride_size` only when `SimpleWindow` is used; pass `sampling_rate` / `new_sr` only when `Downsample` is used.

## Step 8C: Select model

1. Read `$PROJECT_PATH/.tmp_dataset_stats.json` for size bucket and distribution.
   **NPU filter:** if `$TARGET_DEVICE` has an NPU and `.tinyml_template.json` says `npu_template: true`, consider only NPU-compliant models. **Rule of thumb, not a hard rule:** NPU models usually end in `_NPU` (e.g. `CLS_13k_NPU`), but there may be exceptions (the ModelZoo has `ElectricalFault_model_40k_t`, an NPU-optimized model whose name has no `_NPU`; its `model_training_id` is `CNN_TS_GEN_BASE_40K_NPU`). So do not decide by the suffix alone: check the model's description / `model_training_id` for NPU-compliant or NPU-optimized wording too. Drop every other model from `ranked_matches` (step 2) and from the `model_descriptions` fallback (step 3), and recommend only from what is left. If nothing is left, tell the user and ask; never fall back to a non-NPU model silently.
2. ```bash
   $VENV_PYTHON $SCRIPTS_DIR/runner.py select_model_for_task \
     "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"target_module\": \"$TARGET_MODULE\", \"variables\": $VARIABLES}"
   ```
   `has_good_match: true` → start from `recommended_model` and show the `ranked_matches` table. `false` → step 3.
3. **Fallback (only when `has_good_match` is false).** Read the file for the task family in `result.fallback.model_descriptions_path` (`classification.py`, `anomalydetection.py`, `regression.py`, `forecasting.py`). Find models that support `$TASK_TYPE` and `$TARGET_DEVICE` (`target_devices` blocks, `enabled_models_list`). For tiny/small datasets prefer fewer parameters to reduce overfitting; for large datasets any complexity works.
4. Show a table: | Model | Param count | Complexity | Merits | Demerits | Ideal use case |. Then recommend with clear point-by-point reasons (dataset size, device, performance needs) and list alternatives with when to prefer each. Tell the user they may pick any model. Store `MODEL_NAME`, and save the table you showed (model, params, merits, demerits, use case, which one you recommended) to `$PROJECT_PATH/.tmp_model_ranking.json` so Step 13B can show it again.

## Step 9: Training section (complete Parts A–D every time)

**A. Recommendations**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_training_recommendations \
  "{\"target_device\": \"$TARGET_DEVICE\", \"dataset_size_bucket\": \"<from Step 8A>\", \"num_gpus\": <0 or N>}"
```
Relay `quantization.reason` and `nas.reason` to the user.

**B. Basic parameters.** Do not assume defaults. Read the real defaults first:
```bash
grep -E "training_epochs|batch_size|learning_rate" "$TINYML_MODELMAKER_PATH/ai_modules/$TARGET_MODULE/params.py" | head -20
```
Ask all four, showing each default inline: epochs, batch size, number of GPUs, learning rate (all optional). If the user does not know the GPU count, run `$VENV_PYTHON -c "import torch; print(torch.cuda.device_count())"` and use that as `NUM_GPUS`.

**C. Quantization.** Present the modes and the recommendation for `$TARGET_DEVICE` (`quantization.recommended_mode` and its reason):
- 0: Float32, no compression, largest and slowest, NPU not used
- 1: standard PyTorch quantization, about 4× smaller, works on all devices
- 2: TI NPU-optimized, required for NPU acceleration (recommended for NPU devices)

Modes 1 and 2 enable Automatic Mixed Precision, which sets per-layer precision automatically, so do not ask about PTQ/QAT or bit widths. If `.tinyml_template.json` says `npu_template: false`, do not offer mode 2: tell the user the imported project is the non-NPU one and set mode 1. Otherwise store the user's answer as `QUANTIZATION_MODE`; it drives the compilation preset in Step 11.

**D. Generate**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_training_section_yaml \
  "{\"enable\": true, \"model_name\": \"$MODEL_NAME\", \"quantization\": $QUANTIZATION_MODE, \"num_gpus\": $NUM_GPUS, \"training_epochs\": <E>, \"batch_size\": <B>, \"learning_rate\": <LR>}" \
  --save-yaml $PROJECT_PATH/training.yaml
```
Include `training_epochs`, `batch_size` and `learning_rate` only if the user gave a value in Part B (omit the key to keep the `params.py` default); always include `num_gpus` (0 if there is no GPU). Part B's answers only reach the config through this command. Do not set `train_output_path` here; Step 13 sets both output paths for every run (it needs the run's date-time). Show the YAML and get confirmation.

**STRICT RULE: output paths are nested, never top-level.** `train_output_path` belongs ONLY inside the `training:` section and `compile_output_path` ONLY inside the `compilation:` section. They must NEVER appear at the root of the config (a root-level key is silently ignored, so outputs land in the wrong place). Values are absolute paths inside the run directory `$PROJECT_PATH/run/<yyyymmdd-hhmmss>/<model_name>/` (see Step 13), which Step 13 writes for you.

```yaml
# CORRECT
training:
  enable: true
  model_name: <model>
  train_output_path: '/abs/project/run/20261009-153012/<model_name>/training'      # inside training:
compilation:
  enable: true
  compile_preset_name: default_preset
  compile_output_path: '/abs/project/run/20261009-153012/<model_name>/compilation' # inside compilation:

# WRONG - never do this
train_output_path: '/abs/project/run/20261009-153012/<model_name>/training'        # root level: NOT allowed
compile_output_path: '/abs/project/run/20261009-153012/<model_name>/compilation'   # root level: NOT allowed
```


## Step 10: Testing section

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_testing_section_yaml '{"enable": true}' \
  --save-yaml $PROJECT_PATH/testing.yaml
```
Defaults suit most users. Ask only about non-default needs: skip training and test an existing model (`"skip_train": true, "model_path": "<path>"`), run on a connected device (`"device_inference": true`), or a separate test set (`"test_data": "<path>"`).

## Step 11: Compilation section

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_compilation_preset_recommendations \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"quantization_mode\": $QUANTIZATION_MODE}"
```
The preset follows `QUANTIZATION_MODE`: mode 2 on an NPU device → `default_preset` (or `compress_npu_layer_data` for tight memory); mode 0/1 on an NPU device → `forced_soft_npu_preset` (NPU needs mode 2); non-NPU device → `default_preset`. Show `recommended_preset` and `recommendation_reason` and let the user accept or choose another of the three. For a non-NPU project (`npu_template: false`) use `forced_soft_npu_preset` and tell the user why.

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_compilation_section_yaml \
  "{\"enable\": true, \"compile_preset_name\": \"<preset>\", \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $PROJECT_PATH/compilation.yaml
```
Do not set `compile_output_path` here; Step 13 sets it with the run directory. The strict nesting rule in Step 9D applies.

## Step 12: Assemble config.yaml

1. `ls -la $PROJECT_PATH/` shows `common.yaml dataset.yaml feat_ext.yaml training.yaml testing.yaml compilation.yaml`. Regenerate any that is missing.
2. The output paths (`training.train_output_path`, `compilation.compile_output_path`) are not set yet; Step 13 writes them into the per-run folder `$PROJECT_PATH/run/<date-time>/<model_name>/`. They must never appear at the config root; Step 13 also moves a stray root-level one into its section.
3. Confirm with the user, then assemble into the project root:
   ```bash
   $VENV_PYTHON $SCRIPTS_DIR/runner.py generate_complete_config_file \
     "{\"task_name\": \"$TASK_NAME\", \"output_dir\": \"$PROJECT_PATH\", \"common_yaml_file\": \"$PROJECT_PATH/common.yaml\", \"dataset_yaml_file\": \"$PROJECT_PATH/dataset.yaml\", \"feature_extraction_yaml_file\": \"$PROJECT_PATH/feat_ext.yaml\", \"training_yaml_file\": \"$PROJECT_PATH/training.yaml\", \"testing_yaml_file\": \"$PROJECT_PATH/testing.yaml\", \"compilation_yaml_file\": \"$PROJECT_PATH/compilation.yaml\"}"
   ```
4. Show the full `$PROJECT_PATH/config.yaml`.
5. **Delete the section files, immediately.** `config.yaml` now holds everything, and leftover section YAMLs could be mistaken for the real config or drift out of date. First confirm each non-empty section file's content made it into `config.yaml`, then remove them:
   ```bash
   $VENV_PYTHON - "$PROJECT_PATH" <<'PYEOF'
   import os, sys, yaml
   root = sys.argv[1]
   cfg = yaml.safe_load(open(f"{root}/config.yaml"))
   files = {"common": "common", "dataset": "dataset", "data_processing_feature_extraction": "feat_ext",
            "training": "training", "testing": "testing", "compilation": "compilation"}
   paths = {k: f"{root}/{f}.yaml" for k, f in files.items()}
   missing = [k for k, p in paths.items() if os.path.exists(p) and os.path.getsize(p) > 0 and k not in cfg]
   if missing:
       sys.exit(f"NOT deleting: config.yaml lacks {missing}; re-run the assembly")
   for p in paths.values():
       if os.path.exists(p): os.remove(p)
   print("section YAMLs deleted:", sorted(os.path.basename(p) for p in paths.values()))
   PYEOF
   ```
   To change a section later, regenerate it with its Step's command (which rewrites the file) and assemble again, or edit `config.yaml` directly.

---

## Step 13: Run training and compilation (Gate G1)

The config to run is always `$PROJECT_PATH/config.yaml`, whether it came from a bundled example (Step 3.5) or Steps 4–12. Never run a config from the modelzoo examples directory; it would write outputs into modelzoo.

**Pre-flight checks**
```bash
ls -lh "$PROJECT_PATH/config.yaml"
$VENV_PYTHON -c "import tinyml_modelmaker" || echo "ERROR: tinyml_modelmaker not importable in $VENV_PYTHON; re-run /tinyml-agent-skills:setup"
cat "$PROJECT_PATH/.tinyml_template.json" 2>/dev/null   # npu_template: false => config needs quantization 1 + forced_soft_npu_preset
```
**Run directory and output paths (mandatory on every route, before every run).** Bundled examples (Step 3.5), user-supplied configs and Steps 4–12 all arrive here, so this is where the paths are set, fresh for each run (a retrain or model switch gets its own folder and never overwrites an earlier one). Layout, the same as the modelzoo's own `run/<date-time>/<model_name>/`:
```
$PROJECT_PATH/
├── config.yaml                                   (stays here)
└── run/<yyyymmdd-hhmmss>/<model_name>/           = $RUN_DIR
    ├── training/                                 <- training.train_output_path
    └── compilation/                              <- compilation.compile_output_path
```
`<model_name>` is `training.model_name` from `config.yaml`. The script `$SCRIPTS_DIR/run_training.py` picks the timestamp (via `new_run_dir.py`), writes both paths **inside their sections** (never at the root; it removes a stray root-level copy), prints `RUN_DIR=...` and then launches the training command, all in one call. **Training is always started through it (the first run, every retry, every re-run): the install-mode command (`run_tinyml_modelzoo.sh` for repo, `-m tinyml_modelmaker.run_tinyml_modelmaker` for wheel) goes after the `--`; never run that command on its own.**

**Every run gets its own new folder, never a reused one.** This covers a retry after an error, a re-run after a config change, a model switch, a different quantization or preset, or a re-run for any other reason. Never launch training by itself with the paths left over from an earlier run (modelmaker clears its training folder on start, so a reused path overwrites the earlier results), and never edit a timestamp by hand. Earlier `run/<timestamp>/` folders stay untouched; the newest is the current run.

**There is no compile-only mode.** Training and compilation run in one pass, so a compilation failure (for example a wrong compiler path) means a full re-run, not a retry of the compile alone. Fix the cause first (the compiler path is checked at G1, before training, so this should be rare), then re-run through `run_training.py` like any other re-run. Never call the modelzoo command directly to retry a compile: modelmaker clears the training folder on start, so a direct run on an earlier run's paths overwrites the training results it was meant to keep. Each re-run also trains a new model, so its metrics can differ from the earlier run; say so when you report it.

After each run, take `RUN_DIR` from the log line `RUN_DIR=...` as a literal value (Step 13A and Step 14 use it; shell variables do not persist between calls). Show the user the path. If you come back later without it (new session), recover the newest run that has artifacts: `ls -dt "$PROJECT_PATH"/run/*/*/ | while read d; do [ -f "${d}compilation/artifacts/mod.a" ] && echo "${d%/}" && break; done`.
If it says `npu_template: false` and `config.yaml` does not have `training.quantization: 1` and `compilation.compile_preset_name: forced_soft_npu_preset`, fix the config (this also covers a user-supplied config) and tell the user.

**GPU check (always, even for bundled examples).** Run `$VENV_PYTHON -c "import torch; print(torch.cuda.device_count())"`. If N > 0 and the user did not pick a different `num_gpus` in Step 9B, set it now (do not just mention it), then show the result:
```bash
$VENV_PYTHON - "$PROJECT_PATH/config.yaml" <N> <<'PYEOF'
import sys, yaml
p, n = sys.argv[1], int(sys.argv[2])
cfg = yaml.safe_load(open(p))
cfg.setdefault("training", {})["num_gpus"] = n
yaml.safe_dump(cfg, open(p, "w"), default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF
grep -n "num_gpus" "$PROJECT_PATH/config.yaml"
```
Tell the user: "[N] GPU(s) detected, so I set `training.num_gpus: [N]` to speed up training; say if you want CPU only." If N is 0, leave `num_gpus` as it is. The config you show in the next step must already contain this value.

**Confirm.** Show the full config and ask: "Ready to start training from `$PROJECT_PATH/config.yaml`? This will take several minutes." Proceed only on yes.

**Find the compiler (G1).** Use the IDE's project tools to locate the compiler for the `$TARGET_DEVICE` family (`TI C2000 CGT` for F28xxx; `TI ARM CGT` or `tiarmclang` for MSPM0, AM26x). In CCStudio use its MCP tools; in VS Code use the CLI project tools (discover which subcommand lists installed compilers, e.g. via the tool's `--help`). Do not search the filesystem. If several versions are installed, **pick the latest version**. Take the root directory that contains `bin/` as `CGT_ROOT` and tell the user which version you chose. If the tools cannot find it, **stop and ask**: "Please give the full path to your CGT root (the folder containing `bin/cl2000` or `bin/armcl`)."

**Run, in ONE bash call.** `CGT_ROOT` lives in your context, not in the shell. Write its literal value into the command; the same goes for every other `$VAR` below (`TRAIN_DIR`, `TRAIN_CMD`, `VENV_PYTHON`, `SCRIPTS_DIR`, `PROJECT_PATH`). An empty variable fails silently. Env var by device: C2000 (F28xxx) → `C2000_CGT_ROOT`; ARM (MSPM0, AM26x) → `ARM_LLVM_CGT_PATH`.

**Build the command from `MODELZOO_INSTALL_MODE` before writing it; do not improvise.** Substitute literal values, and use exactly one of these two forms. `<CGT_VAR>` is `C2000_CGT_ROOT` for C2000 (F28xxx) or `ARM_LLVM_CGT_PATH` for MSPM0 / AM26x.

| Mode | Command (after `cd "<TRAIN_DIR>" &&` and `<CGT_VAR>="<CGT_ROOT>"`) |
|---|---|
| `repo` | `./run_tinyml_modelzoo.sh "<PROJECT_PATH>/config.yaml"` |
| `wheel` | `"<VENV_PYTHON>" -m tinyml_modelmaker.run_tinyml_modelmaker "<PROJECT_PATH>/config.yaml"` |

Wheel-mode rules (each one is a mistake seen before):
- The only required argument is the config path, as the **first positional argument**. Do not pass `--config`, `--config_file` or `--task_type`; the other flags (`--run_name`, `--num_gpus`, `--model_name`, `--input_data_path`, `--target_device`, `--batch_size`) are optional overrides, and the config already carries these values.
- Use the venv's full python path (`<VENV_PYTHON>`), not bare `python` or `python3`, and always with `-m`. Never run `python .../run_tinyml_modelmaker.py` by file path, never `cd` into `site-packages` or the package, and never call `run_tinyml_modelzoo.sh` (a wheel install does not have it).
- `<TRAIN_DIR>` is `$HOME/.tinyml-agent-skills` (a neutral working directory; create it with `mkdir -p` if missing). Outputs go to the absolute paths in the config, not here.
- Check once before running: `"<VENV_PYTHON>" -c "import tinyml_modelmaker"` must exit 0.

```bash
LOG_FILE=$(mktemp /tmp/tinyml_run_XXXXXX.log)
source "$(dirname "$(dirname "<VENV_PYTHON>")")/bin/activate"
<CGT_VAR>="<ACTUAL_CGT_ROOT_PATH>" \
  "<VENV_PYTHON>" "<SCRIPTS_DIR>/run_training.py" "<PROJECT_PATH>" "<TRAIN_DIR>" -- \
  <TRAIN_CMD> "<PROJECT_PATH>/config.yaml" > "$LOG_FILE" 2>&1 &   # creates a NEW run folder, then trains; RUN_DIR=... is in the log
RUN_PID=$!
tail -f "$LOG_FILE" &
TAIL_PID=$!
wait $RUN_PID
TRAIN_EXIT=$?
kill $TAIL_PID 2>/dev/null
echo "Training exit code: $TRAIN_EXIT"
```

Concrete wheel-mode example (MSPM0, ARM compiler; every value is a literal, none left as a variable):
```bash
LOG_FILE=$(mktemp /tmp/tinyml_run_XXXXXX.log)
source "/home/me/.tinyml-agent-skills/venv/bin/activate"
ARM_LLVM_CGT_PATH="/home/me/ti/ti-cgt-armllvm_4.0.0.LTS" \
  "/home/me/.tinyml-agent-skills/venv/bin/python" "/home/me/tinyml-agent-skills/plugins/tinyml-agent-skills/skills/ti-ccstudio-tinyml-workflow/scripts/run_training.py" "/home/me/workspace/pir_detection" "/home/me/.tinyml-agent-skills" -- \
  "/home/me/.tinyml-agent-skills/venv/bin/python" -m tinyml_modelmaker.run_tinyml_modelmaker "/home/me/workspace/pir_detection/config.yaml" > "$LOG_FILE" 2>&1 &
```

Example: for a C2000 device where the IDE's tools returned `/opt/ti/ccs/tools/compiler/ti-cgt-c2000_21.6.0.LTS`, the prefix is `C2000_CGT_ROOT="/opt/ti/ccs/tools/compiler/ti-cgt-c2000_21.6.0.LTS"`.

A non-zero `TRAIN_EXIT` means failure. When it completes, summarize for the user: accuracy/loss from the `run.log` under `$RUN_DIR/training/` (`find "$RUN_DIR/training" -name run.log`) and model size/latency from `$RUN_DIR/compilation/run.log`. Record `RUN_ID` (the timestamp directory name in `RUN_DIR`), `MODEL_ID`, and `QUANTIZATION` (true/false) from the logs.

**On any error:** work out what in the config could have prevented it, explain the cause to the user, propose the change (or take theirs), apply it, and re-run with the same run block above (it creates a new `run/<new timestamp>/<model_name>/` folder, so the failed run is kept and nothing is overwritten). Repeat until training and compilation run cleanly.

**Common issues**
1. *Feature-extraction preset fails on size.* Some classes have too few samples for the presets. If no preset fits, use raw transforms; check available transforms and presets in `$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py`.
2. *Skewed metrics.* Files shorter than the preset expects are skipped silently, so metrics can look unrealistically high. Do not report them as final: stop, fix the config as in issue 1, and re-train.

## Step 13A: Verify artifacts and memory footprint (Gate G2)

Do not start Step 14 until both checks pass.

1. **Artifacts exist:** `ls -la "$RUN_DIR/compilation/artifacts/"`. `mod.a` must be present and non-empty. If not, compilation failed: go back to Step 13's error handling.
2. **Footprint:** `grep -E "FLASH|SRAM|RO Mem|RW Mem" "$RUN_DIR/compilation/run.log"`, then show:
   ```
   MODEL MEMORY FOOTPRINT    Device: $TARGET_DEVICE
     FLASH (RO memory): <SIZE> bytes
     SRAM  (RW memory): <SIZE> bytes
   ```
   Ask the user to confirm it fits their device (check the device datasheet or SDK docs for FLASH/SRAM sizes).

If it does not fit: stop and do not deploy. Reduce size (quantization mode 2, "Memory" NAS optimization, or a smaller model), re-run Step 13, and re-check.

**Next: Step 13B.** If this is a BYOD run (see 13B for how to tell), go straight to Step 13B now. Do not ask about deployment, mention Step 14, or end your turn first. Bundled-example runs go to Step 14.

---

## Step 13B: Review the result with the user (BYOD flow only) — MANDATORY, NEVER SKIPPED (Gate G6)

**This step is mandatory in the BYOD flow and is never skipped.** It is asked after every successful training + compilation run, including re-runs after a retrain or a model switch result. Do not ask about deployment, do not start Step 14, and do not end your turn with a summary until the user has answered this question. Only the BYOD flow has it; it never applies to a bundled example used as-is.

**Is this a BYOD run?** Yes when the user brought their own dataset: `.tinyml_template.json` has `byod: true` (written in Step 4, or in Step 3.5 Option 2), **or** this session built the config from the user's own data in Steps 4–12 or Step 3.5 Option 2, even if the flag is missing (then set it now). No when the user used a bundled ModelZoo example as-is (Step 3.5 Option 1): skip to Step 14, because it is a demonstration and there is nothing to improve. If you cannot tell, check: `grep byod "$PROJECT_PATH/.tinyml_template.json"`, and ask the user whether the dataset is their own.

A model trained on a new dataset is only worth deploying if the user finds the numbers good enough. Show the achieved metrics from the training log (accuracy, F1 or the task's metric, per-class results if available) together with the footprint from 13A, then ask exactly: "Are you happy with this result?" Then **end your turn and wait for the reply.**

- **Yes:** ask whether they want to deploy to a device and continue at Step 14.
- **No, or any sign of dissatisfaction at any point after a run** ("not good", "accuracy is low", "doesn't work", "can I do better"): treat it as "No" right away, even outside this question. Offer to improve it via the following:
  **A different modelzoo model**, chosen from the Step 8C ranking you showed earlier (saved in `.tmp_model_ranking.json`).
---

## Step 14: Deploy to device (only if the user asks)

In the BYOD flow, Step 14 starts only after Step 13B (G6) was answered "yes" for the latest run. `$RUN_DIR` (Step 13) must be known before 14A; if you do not have it, recover it with the command in Step 13. Do 14A → 14E in order. Deployment needs *both* the exclusions (14B) and the copied artifacts (14C); skipping either gives linker errors or a stale model on the device. Show the evidence for each gate before moving on.

Read `references/deployment_procedure.md` before 14B: it has the CCStudio `.cproject` PASS-check script and the details and error lists for build and flash in 14D/14E. Lookups and troubleshooting (device → ccxml, NPU vs non-NPU, on-device verification): `references/device_deployment_guide.md`.

### Step 14A: Verify artifacts exist (Gate G2)

```bash
ls -la "$RUN_DIR/compilation/artifacts/mod.a"
```
If it is missing or empty, training/compilation is incomplete. Stop; do not proceed to 14B. Go back to Step 13.

---

### Step 14B: Keep `run/`, `config.yaml` and `.tinyml_template.json` out of the build (Gate G3)

These three items are work artifacts and configuration, NOT source code: the `run/` folder (all training and compilation outputs), `config.yaml` and `.tinyml_template.json`. If they were built, you would get linker errors (trying to compile `config.yaml`) or duplicate links (`mod.a` linked twice). Exclude nothing else.

**VS Code: no edit needed, but check and inform.** These items are excluded from the build by default, unless they were added in `.ccs/projects.json`. Tell the user that, then look in `.ccs/projects.json` (in the workspace root, or wherever the IDE keeps it; if the file does not exist, say so) for an entry that includes `run/`, `config.yaml` or `.tinyml_template.json` for this project, and tell the user the result: either "none of them are included, so they are excluded" or "`<item>` is included in `.ccs/projects.json`; it will be built unless you remove that entry". Do not edit the file; inform the user and carry on to 14C. No confirmation, no PASS script.

**CCStudio only: do the following. It is mandatory and must complete before Step 14C.** The exclusion list is fixed: `run|config.yaml|.tinyml_template.json`. Do not scan for or add other items.

**MANDATORY, ASKED EVERY SINGLE TIME, NEVER SKIPPED: confirm the exclusion list with the user before editing `.cproject`.** Ask: "These will be excluded from the CCStudio build: `run/`, `config.yaml` and `.tinyml_template.json`. Nothing else is excluded. Is this right?" even when the user has done this before, even on a repeat run. Then **end your turn and wait for the reply**. Do not touch `.cproject`, and do not announce that you are proceeding, until the user has answered. If they ask to add or remove an item, apply it and use that final list wherever the list appears below.

**Step 1 — Update .cproject XML file:**

Read `.cproject` located at `$PROJECT_PATH/.cproject`:
```bash
cat "$PROJECT_PATH/.cproject"
```

Update the `<sourceEntries>` section to exclude the folders and file using the pipe-separated format.

**EXACT placement in XML hierarchy:**
```xml
<configuration id="...">
  <folderInfo id="..." name="/">
    <!-- folder contents here -->
  </folderInfo>
  <!-- sourceEntries is AFTER folderInfo tag, within configuration tag -->
  <sourceEntries>
    <entry excluding="run|config.yaml|.tinyml_template.json" flags="VALUE_WORKSPACE_PATH|RESOLVED" kind="sourcePath" name=""/>
  </sourceEntries>
</configuration>
```

`<sourceEntries>` is a **sibling of `<folderInfo>`**, both under `<configuration>` — **NOT nested inside `<folderInfo>`**. **Place it AFTER the closing `</folderInfo>` tag, BEFORE the closing `</configuration>` tag.**

**Important:** 
- Use pipe (`|`) to separate multiple exclusions (no spaces around pipes)
- This approach excludes folders and files together in a single `excluding` attribute
- The `flags` and `kind` attributes must remain as shown
- If `<sourceEntries>` already exists, keep the names it already excludes (CCStudio lists the inactive `targetConfigs/*.ccxml` there) and add `run`, `config.yaml` and `.tinyml_template.json` to the same pipe-separated list
- If `<sourceEntries>` does not exist, add it as shown above (after `</folderInfo>`, before `</configuration>`)

**Step 2 — Verify the edit:** run the PASS-check script from `references/deployment_procedure.md` (§14B). It checks every `<configuration>` block (Debug, Release, ...) and must print `PASS`. If it reports a block as missing, apply Step 1 to that block and re-run.

**Step 3 — User confirmation in the IDE (MANDATORY):**

> "I've updated the project settings (.cproject file) to exclude `run/`, `config.yaml` and `.tinyml_template.json` from the build. **Please verify in CCStudio's Project Explorer:** they should appear **greyed out** (or with a strikethrough). If not, right-click each and select 'Exclude from Build'. Reply 'confirmed' when all are excluded."

Do not proceed to Step 14C without the user's explicit reply. Both user confirmations (the exclusion list before the edit, and the greyed-out check after it) are required on every CCStudio deployment.

---

### Step 14C: Replace the template's artifacts with the trained ones (Gate G4)

Training outputs sit in `$RUN_DIR` (inside `$PROJECT_PATH/run/`), which is now excluded from the build. The imported project already contains placeholder artifacts from the SDK template, and its build config refers to them **by name and location** (C2000 projects link `artifacts/mod.a`; MSPM0 projects link `model/model.a`). So each trained file must **replace** its placeholder in place and **take the placeholder's file name**. Modelzoo always produces `mod.a`; if the project expects `model.a`, the trained file is written as `model.a`. A second file next to the old one, or under a different name, leaves the old model linked or breaks the link.

**C1. Locate the placeholders** in the imported project (not the SDK, and not our own outputs):
```bash
find_ph() { find "$PROJECT_PATH" -type f -name "$1" -not -path "*/run/*"; }
find_ph "*.a"; find_ph "tvmgen_default.h"; find_ph "test_vector.c"; find_ph "user_input_config.h"
```
Call the full paths found `LIB_PATH`, `HDR_PATH`, `TV_PATH`, `UIC_PATH`. Check that the build really uses the library name you found: `grep -rn "<LIB file name>" "$PROJECT_PATH/.cproject"` should show the linker reference. If `*.a` returns more than one file, work out which one the build links before replacing anything.

- If there is **no placeholder library or header** (e.g. an empty starter), SDK layouts differ, so work it out from the project itself: look at how its build and sources are set up (linker settings, include paths, existing folders such as `model/` or `artifacts/`, what the starter's `main.c` includes or expects) and choose the location and file names that fit, creating the folder if needed and adding the files to the build if the project requires it. Tell the user what you chose and why. If the project gives you no clear answer, ask the user instead of guessing.
- If there is no `test_vector.c` / `user_input_config.h`, the template has no golden-vector files: skip C3 and its checks, **then see C5**. **For a BYOD run C5 is mandatory in every case; if the training run produced no golden vectors, stop and go back to Step 13.**

**C2. Replace the compiled artifacts.** Write the literal paths from C1; `cp -f` overwrites the placeholder and so keeps its name:
```bash
cp -f "$RUN_DIR/compilation/artifacts/mod.a"            "<LIB_PATH>"
cp -f "$RUN_DIR/compilation/artifacts/tvmgen_default.h" "<HDR_PATH>"
```
Tell the user what was replaced and any rename, e.g. "Replaced `model/model.a` with the trained `mod.a` (kept the name `model.a`)".

**C3. Replace the golden vectors (only if C1 found them).** Source is `$RUN_DIR/training/training_quantization/golden_vectors/`, or `$RUN_DIR/training/training_base/golden_vectors/` when quantization is off (confirm with `ls "$RUN_DIR/training"`; other modelzoo versions may name them `quantization`/`base`):
```bash
GV="$RUN_DIR/training/training_quantization/golden_vectors"
[ -d "$GV" ] || GV="$RUN_DIR/training/training_base/golden_vectors"
cp -f "$GV/test_vector.c"       "<TV_PATH>"
cp -f "$GV/user_input_config.h" "<UIC_PATH>"
```

**C4. Verify (BLOCKING). All must pass before Step 14D:**
```bash
ls -lh "<LIB_PATH>" && [ -s "<LIB_PATH>" ]                                          # present and non-empty
grep -c tvmgen "<HDR_PATH>"                                                         # expected TVM symbols
cmp "$RUN_DIR/compilation/artifacts/mod.a"            "<LIB_PATH>"         # now identical to the trained output
cmp "$RUN_DIR/compilation/artifacts/tvmgen_default.h" "<HDR_PATH>"
cmp "$GV/test_vector.c"       "<TV_PATH>"                                           # golden-vector templates only
cmp "$GV/user_input_config.h" "<UIC_PATH>"
```
`cmp` prints nothing when the files are identical. Any difference means the wrong or a partial file was copied: re-copy and re-compare. If the source files are missing, check `ls "$RUN_DIR/compilation/artifacts/"`, `tail -20 "$RUN_DIR/compilation/run.log"` and `ls "$GV"`; a missing golden vector usually means training did not finish.

**C5. Golden-vector test (Gate G4b). MANDATORY AND UNAVOIDABLE FOR EVERY BYOD RUN.** After C4 a BYOD run **cannot proceed to the build** until the project provably runs a golden-vector test on the trained model. (Is it BYOD? Same rule as Step 13B: `.tinyml_template.json` has `byod: true`, or the config was built from the user's own data.) A bundled ModelZoo example used as-is skips C5 and goes to 14D after C4.

1. **Read `references/golden_vector_test.md` in full now**, even if you read it earlier in the session, and follow it. Do not work from memory.
2. **Make sure the test exists.** If the project already has a `test_result` check (an exact or generic template), C3 replaced its vectors; you still verify it in step 3. If it has none (e.g. a bare starter such as `hello_world_ai`), add the test directly into `main.c` as the reference describes (preconditions 1a-1c, backup, diff, user's OK).
3. **Verify the wiring.** Run `gv_verify.py` from the reference. It must print `PASS`: `main.c` (the file that calls the model) defines `test_result`, calls the golden-vector test, includes `tvmgen_default.h` and `user_input_config.h`, and the copied `test_vector.c` provides `raw_input_test`, `model_test_input` and `golden_output`, all in the project and outside the excluded folders. On FAIL, fix and re-run.
4. Back up the original, show the user the diff, and wait for their OK. Only then continue to 14D.

If the test cannot be built for this project (for example the SDK's feature-extraction library lacks a flag the config needs, which the reference's check 1a reports), **stop and tell the user**; do not build without a golden-vector test. 14D's pre-flight re-checks G4b and refuses to build without it.

---

### Step 14D: Build project (Gate G5)

**Pre-flight (MANDATORY):** re-run the 14C-C4 `cmp` checks, for a BYOD run the `gv_verify.py` check of G4b (it must print `PASS`), and, on CCStudio, the 14B PASS check. If either fails, STOP and return to that step; do not attempt a build.

Build the project with the IDE's own tools (see "IDE and tooling"). First end any active debug session for `$PROJECT_NAME`, since it locks the project and can make the build hang. A compiler or linker error is a real build failure: fix the cause. If no tooling can be used at all, stop, give the exact reason, and ask the user to enable it or to build in their IDE and report the result. Gate G5 passes only on a build you saw succeed or one the user confirms finished with no errors.

Details and wording for the prompts: `references/deployment_procedure.md` §14D.

---

### Step 14E: Flash to device

Connect the device by USB/JTAG first. Do not flash if G5 did not pass.

**IMPORTANT -> Before flashing: confirm the target configuration (the `.ccxml`).** The debugger connects through the project's *active* target configuration. If it names the wrong probe or board (for example the bare device `.ccxml` when a LaunchPad is attached), the flash fails with a connection error or, worse, talks to the wrong target, so check it first. Do this every time, before starting the flash.

1. **Read the facts** (both are in the project; do not guess them):
   ```bash
   ls "$PROJECT_PATH"/targetConfigs/*.ccxml
   grep -o 'activeTargetConfiguration value="[^"]*"' "$PROJECT_PATH/.ccsproject"
   ```
2. **Exactly one `.ccxml`:** nothing to ask. If it is not the active one (or no active one is set), set it (step 4). Tell the user in one line, e.g. "Flashing with the project's only target configuration, `MSPM0G5187.ccxml`."
3. **More than one `.ccxml`:** tell the user explicitly which one is active and list the others, then ask which matches their hardware and **wait for the answer**:
   > "This project has these target configurations: `<A>.ccxml`, `<B>.ccxml`. The active one is `<A>.ccxml`. Which matches your hardware? For example, a LaunchPad/EVM with an on-board debug probe usually has its own `*_LaunchPad.ccxml` (or board-named one), while a custom board with an external probe uses the plain device `.ccxml`. Keep the active one, or switch?"

   Recommend one from what the user told you and say why, but do not switch silently.
4. **Switch** only on the user's say-so. In a CCStudio project this is done by editing `.ccsproject`; in other IDEs use the IDE's own way of choosing the target configuration. It is flat XML: every option is a one-line child of the single `<projectOptions>` element, and the set of options differs per device and SDK. You only touch two of them (shown here with placeholders, as an illustration of the shape, not a template to copy):
   ```xml
   <projectOptions>
   	... other options, leave as they are ...
   	<activeTargetConfiguration value="targetConfigs/<chosen>.ccxml"/>
   	<isTargetConfigurationManual value="true"/>
   </projectOptions>
   ```
   If a line exists, change only its `value` (keep the `targetConfigs/` prefix). If it is missing, add it on its own line directly before the closing `</projectOptions>`, never inside another element. This script does it (write the literal file name):
   ```bash
   $VENV_PYTHON - "$PROJECT_PATH/.ccsproject" "<CHOSEN>.ccxml" <<'PYEOF'
   import re, sys
   path, name = sys.argv[1], sys.argv[2]
   s = open(path).read()
   for tag, val in (("activeTargetConfiguration", f"targetConfigs/{name}"), ("isTargetConfigurationManual", "true")):
       line = f'<{tag} value="{val}"/>'
       if re.search(rf'<{tag} value="[^"]*"/>', s):
           s = re.sub(rf'<{tag} value="[^"]*"/>', line, s)
       else:
           s = s.replace("</projectOptions>", f"\t{line}\n</projectOptions>", 1)
   open(path, "w").write(s)
   PYEOF
   grep -E 'activeTargetConfiguration|isTargetConfigurationManual' "$PROJECT_PATH/.ccsproject"
   ```
   The read-back must show the chosen file. In CCStudio, then ask the user to confirm (right-click the `.ccxml` → **Set as Active Target Configuration**; F5 to refresh if it still shows the old one).

Leave the project's existing `excluding` entries for the inactive `.ccxml` files alone; the IDE maintains them.

Then load the program onto the device and run it using the IDE's debug/flash tools: end any existing session for the project, start a debug session (which loads the program; load the `.out` explicitly if it did not), and resume. This takes 30–90 s on hardware; treat it as hung only after 3 minutes. A `device not found` / `no target connected` error is a physical problem: ask the user to check USB/JTAG and retry. If no tooling can be used at all, stop, give the exact reason, and ask the user to enable it or to flash from their IDE and report the result.

Commands and error lists: `references/deployment_procedure.md` §14E.

**After flashing:** in a debug session set a breakpoint after the inference call and watch `test_result`: 1 means the output matches the golden vector, 0 means failure. Troubleshooting: `references/device_deployment_guide.md`.
