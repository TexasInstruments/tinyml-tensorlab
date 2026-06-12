---
name: tinyml-agentic-skill
description: Guides users through end-to-end TinyML model creation, training, compilation, and device deployment. Use when user mentions creating AI solutions, AI models, training for embedded devices, deploying to MCUs, or any tinyML workflow. Covers config creation, model training, compilation, and Code Composer Studio flashing. Always confirm tinyML tensorlab installation path before proceeding.
---

# TinyML Tensorlab Skill

## Overview 

Build, train, compile, and deploy ML models to embedded MCUs using TinyML tensorlab.

---

## CRITICAL: Script and Path Locations

**`runner.py` and `SCRIPTS_DIR` are in THIS SKILL directory, NOT in tinyml-tensorlab.**

When setting up:
```bash
SCRIPTS_DIR=<path-to-this-skill>/scripts
# Example: /home/user/.claude/skills/tinyml-tensorlab-skill/scripts

# runner.py is at:
$SCRIPTS_DIR/runner.py
```

Do NOT use:
- `~/tinyml-tensorlab/scripts/` (wrong — this doesn't exist)
- `~/tinyml-tensorlab/tinyml-modelmaker/scripts/` (wrong)

Use:
- `~/.claude/skills/tinyml-tensorlab-skill/scripts/` (correct — this is the skill)

---

**Workflow (13 steps):**
1. **Session setup** — confirm installation path, setup venv if one is not present, make sure tinyml-tensorlab is fully set-up. Refer `references/setup_guide.md` for setup and installation guide for tinyml-tensorlab.
2. **Requirements** — task type, device, data type, channel count
3. **Common section** — task_type + target_device → save to WORK_DIR
4. **Dataset validation** — validate format, get effective path
5. **Dataset section** — generate YAML → save to WORK_DIR
6. **Feature extraction and Data Processing transforms**
   - Step 6A: Analyze dataset for statistical insights
   - Step 6B: Get recommendations → generate YAML → save to WORK_DIR
7. **Model selection** — analyze dataset size → rank models
8. **Training section** — model name + hyperparams → save to WORK_DIR
9. **Testing section** — testing config → save to WORK_DIR
10. **Compilation section** — preset selection → save to WORK_DIR
11. **Assemble config** — combine all sections → write config.yaml
12. **Run training** — execute run_tinyml_modelzoo.sh
13. **Deploy to device** — create CCS project → flash
**IMPORTANT: After EACH step which generates a section of the config file, pause and show the user the config file (created thus far) and proceed only with user's approval of the config.**
**Reference guides** (read on demand, not upfront):
- `references/setup_guide.md` - Session setup, venv activation, tinyml-tensorlab repo setup and verification.
- `references/config_creation_guide.md` — task types, devices, YAML rules
- `references/example_running_guide.md` — run commands, monitoring, troubleshooting
- `references/device_deployment_guide.md` — CCS project, flashing, validation

---
## Session Setup (do this before Step 1) ***NEVER SKIP THIS***

First, check if a `.env` file exists at the root level of this skill(i.e `tinyml-tensorlab-skill`). 
If it does, get the values of the environment variables (from the list given below) from their corresponding values in the `.env` file, set the values, and export the same in the session **WITHOUT FAIL**.
| Variable          | Source         | Description                              |
|-------------------|----------------|------------------------------------------|
| `SCRIPTS_DIR`     | Setup          | Absolute path to scripts/ folder         |
| `IS_REPO_SETUP`   | Setup          | Indicates if tinyml-tensorlab is setup or not |
| `TINYML_BASE_PATH`| Setup          | Root of tinyml-tensorlab checkout        |
| `TINYML_TENSORLAB_DOCS_PATH`       | Setup          | Path to documentation for tinyml-tensorlab |

If the `.env` file does not exist or if any env variable **whose source is `setup`** still remains un-set, then refer `references/skill_setup_guide.md` for steps to setup the skill. Specifically, use the guide to setup any un-set required variables or if nothing is setup (i.e it is the first setup) then make sure all the above variables are set (**whose source is `setup`**) and the repo is fully ready.

Once the setup is complete, **return here and ensure the following instructions are understood BEFORE proceeding to step 1**.

While working, always create your config files in the /examples directory within tinyml-tensorlab/tinyml-modelzoo, where all other examples are present.

**Session state — track these throughout all steps:**
**All variables with source as `Setup` will be set as per the rules specified above. The rest of the variables will be progressively set up in upcoming steps and are subject to change for each user-given problem statement. Therefore they are not to be added to the `.env` file.

| Variable          | Source         | Description                              |
|-------------------|----------------|------------------------------------------|
| `SCRIPTS_DIR`     | Setup          | Absolute path to scripts/ folder         |
| `IS_REPO_SETUP`   | Setup          | Indicates if tinyml-tensorlab is setup or not |
| `TINYML_BASE_PATH`| Setup          | Root of tinyml-tensorlab checkout        |
| `TASK_NAME`       | Step 2         | Slug used for config subdirectory        |
| `WORK_DIR`        | Step 2         | Temp dir for intermediate section YAMLs  |
| `CONFIG_DIR`      | Step 11        | Permanent dir where config.yaml is saved  |
| `TASK_TYPE`       | Step 2         | e.g., `motor_fault`                      |
| `TARGET_DEVICE`   | Step 2         | e.g., `F28P55`                           |
| `TARGET_MODULE`   | Step 3 output  | `timeseries` or `vision`                 |
| `VARIABLES`       | Step 2/6       | Number of sensor channels                |
| `DATA_PATH`       | Step 2         | User's raw dataset path                  |
| `EFFECTIVE_DATA_PATH` | Step 4     | Path after any auto-reorganization       |
| `QUANTIZATION_MODE` | Step 8C      | 0/1/2 — drives compilation preset       |
| `NAS_ENABLED`     | Step 8D        | true/false                               |
| `TINYML_TENSORLAB_DOCS_PATH`       | Setup          | Path to documentation for tinyml-tensorlab |
---

## Error Handling Rule

**After every runner.py call:** check `success` in the JSON output.
- `"success": true` → proceed to next step
- `"success": false` → read `errors` array, fix the issue, re-run the same step

Never skip to the next step while `success` is false. The `errors` field always describes what went wrong and what to fix.

---

## Step 1: Session setup

See "Session Setup" above. Do not proceed until:
- `check_installation` returns `success: true`
- `IS_REPO_SETUP` is set to `true` or `1`
- All four setup variables (`SCRIPTS_DIR`, `IS_REPO_SETUP`, `TINYML_BASE_PATH`, `TINYML_TENSORLAB_DOCS_PATH`) are set and saved to `.env`
- Virtual environment is activated
---

## Step 2: Understand requirements

Ask the user these questions (all answers needed before proceeding):

1. **Task type** — "What kind of ML task? Classification, anomaly detection, regression, or forecasting?"
   - Use `list_supported_values` to show exact valid strings:
   ```bash
   python3 $SCRIPTS_DIR/runner.py list_supported_values '{"parameter_type": "task_type"}'
   ```

2. **Target device** — "Which MCU are you targeting?"
   ```bash
   python3 $SCRIPTS_DIR/runner.py list_supported_values '{"parameter_type": "target_device"}'
   ```

3. **Data** — "Where is your dataset? (local path or URL)" and "How many sensor channels/variables does it have?"

4. **Task name** — pick a name for the config directory, e.g., `motor_fault_demo`

Set session variables from answers:
```bash
TASK_TYPE=<answer>
TARGET_DEVICE=<answer>
VARIABLES=<answer>
DATA_PATH=<answer>
TASK_NAME=<answer>
WORK_DIR=$(mktemp -d -t tinyml_${TASK_NAME}_XXXXXX)
echo "Work dir: $WORK_DIR"
```

**Task-type disambiguation example:**
```
User: "motor fault detection"
→ Ask: "Classify fault vs healthy (motor_fault / generic_timeseries_classification)
        or detect anomalies (generic_timeseries_anomalydetection)?"
```

---

## Step 3: Generate common section

```bash
python3 $SCRIPTS_DIR/runner.py generate_common_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $WORK_DIR/common.yaml
```

- On `success: true`: note `inferred_module` (set as `TARGET_MODULE`)
- On `success: false`: show `errors`, call `list_supported_values` for the invalid parameter, correct and retry

```bash
# Optional: custom run_name
python3 $SCRIPTS_DIR/runner.py generate_common_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"run_name\": \"{date-time}/{model_name}\"}" \
  --save-yaml $WORK_DIR/common.yaml
```

---

## Step 4: Validate dataset format

This step checks the dataset directory structure and auto-fixes it if possible.
It returns `effective_input_data_path` — which may differ from `DATA_PATH` if data was reorganized.

```bash
python3 $SCRIPTS_DIR/runner.py validate_dataset_section \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$DATA_PATH\", \"task_type\": \"$TASK_TYPE\"}"
```

Read the output carefully:
**Stage 1: Format validation**

If `success: true`:
- Set `EFFECTIVE_DATA_PATH` to `effective_input_data_path` from output (use this, not `DATA_PATH`, from now on):
```bash
EFFECTIVE_DATA_PATH=<value from effective_input_data_path>
```

If `success: false`:
1. Create a copy of the user-given dataset.
2. Manually re-organize the copy into the required structure per `$SCRIPTS_DIR/constants.py` → `EXPECTED_STRUCTURES[{task_family}]`.
3. Set `EFFECTIVE_DATA_PATH` to the reorganized copy's path and re-run `validate_dataset_section`.

---

## Step 5: Generate dataset section

**Always ask user for the below params:**
- Split type: `amongst_files` or `within_files`
**IMPORTANT: If after formatting, each class has only ONE csv(/txt/npy or any supported format) file, then split_type of `within_files` MUST be used**
- Split factor: e.g.`[0.6, 0.3, 0.1]` -> THIS IS JUST AN EXAMPLE

**IMPORTANT: If user does not know or does not specify, then create WITHOUT the above two params. Inform user that they will be picked from params.py (default values).**

```bash
python3 $SCRIPTS_DIR/runner.py generate_dataset_section_yaml \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$EFFECTIVE_DATA_PATH\"}" \
  --save-yaml $WORK_DIR/dataset.yaml
```

With optional split params:
```bash
python3 $SCRIPTS_DIR/runner.py generate_dataset_section_yaml \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$EFFECTIVE_DATA_PATH\", \"split_type\": \"USER_GIVEN_SPLIT_TYPE\" \"split_factor\": {USER_GIVEN_SPLIT_FACTOR}}" \
  --save-yaml $WORK_DIR/dataset.yaml
```

---

## Step 6A: Analyze dataset for statistical insights
Run:
```bash
python3 $SCRIPTS_DIR/runner.py analyse_dataset \
  "{\"formatted_dataset_path\": \"$EFFECTIVE_DATA_PATH\", \"task_family\": \"{classification/anomalydetection/regression/forecasting}\"}"
```
Use the output for downstream tasks - particularly for the feature extraction presets/transforms, data processing presets/transforms, and model selection recommendations. To persist the results for the rest of the session, follow the below steps:
1. Create a temporary file titled `.tmp_dataset_stats.json` within the temporary `$WORK_DIR`. 
2. Store the following JSON in it's exact format in the above file:
{
  "result":{
    "dataset_bucket": <tiny|small|medium|large>,
    "dataset_size": <total_num_samples>
    "task_type": `$TASK_TYPE`,
    "min_sample_or_seq_length": <taken-from-output-of-analyse-dataset>
    ----------------------------------------------------
    if task_family is classification add the below key:
    
    "data_distribution": {
        class_{cls1}:<total_num_of_entries_in_cls1>,
        class_{cls2}:<total_num_of_entries_in_cls2>,
        ... 
    },
    ----------------------------------------------------
    if task_family is anomalydetection add the below key:
    
    "data_distribution": {
        normal:<total_num_of_normal_entries>,
        anomalous:<total_num_of_anomalous_entries>
    },
    ----------------------------------------------------
    if task_family is regression add the below key:

    "data_distribution": {
        datafile1:<total_num_of_entries_in_datafile1>,
        datafile2:<total_num_of_entries_in_datafile2>
    },
    ----------------------------------------------------
    if task_family is forecasting add the below key:
    
    "data_distribution": {
        sequence1:<total_num_of_entries_in_sequence1>,
        sequence2:<total_num_of_entries_in_sequence2>
    },
    ----------------------------------------------------
    "formatted_dataset_path": <path to formatted dataset (step 3 or step 2 if dataset was already in correct format)>
  }
}

## Step 6B: Generate feature extraction and Data Processing transforms

**Part A — get recommendations (required):**
Follow the below flowchart first to intelligently select values for the mentioned params:

**Param Selection FlowChart:**
Is the pattern of the data in frequency content?
|-- Yes --> Use FFT-based preset --> set `prefer_fft` to true
|   |-- Need full spectrum? --> set `need_full_spectrum` to true
|   |-- Reduce features? --> Binning to be used --> `need_full_spectrum` set to false
|-- No --> Use RAW preset --> set `prefer_fft` to false
    |-- Need temporal context? --> Multi-frame --> set `need_temporal_ctx` to true
    |-- Single snapshot? --> 1Frame --> set `need_temporal_ctx` to false

Based on the above flowchart, run the following command:
```bash
python3 $SCRIPTS_DIR/runner.py get_data_proc_feat_ext_recommendations \
  "{\"task_type\": \"$TASK_TYPE\", \"prefer_fft\": <based-on-above-analysis>, \"need_full_spectrum\":"<based-on-above-analysis>, \"need_temporal_ctx\":"<based-on-above-analysis>\", \"min_sample_or_seq_length\":{`min_sample_or_seq_length` from step-6}, variables\": $VARIABLES}"
```
**IMPORTANT - NEVER SKIP THIS CHECK**
Before returning the fetched recommendations to the user, go through the dataset analysis in `$WORK_DIR/.tmp_dataset_stats.json` completely. Based on these statistics ensure the recommended presets/transforms are suited for the dataset.
For example:
If for a classification task, the `min_sample_or_seq_length` is 70, then having a frame_size > 70 in the preset or data transform would be too high and would cause that sample file to be skipped, thus causing data loss. **This would be unacceptable.**
You must therefore suggest presets or transforms having frame/window size <=`min_sample_or_seq_length`. #NAME min_sample_or_seq_length could be causing confusion for agent

Also, understand what each recommended transform does - go through `references/FE_and_Data_Processing_Transforms/FE_transforms.md` and `references/Data_processing_transforms.md`. For presets, consult `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/constants.py` (source of truth for feature extraction and data processing presets). 
Try understanding whether your recommendations actually will be useful for the dataset you are working with and if so, why. **Give CLEAR point-by-point reasoning to the user regarding why your recommended transforms or presets are valid and useful.**

Show user the complete, structured output of `get_data_proc_feat_ext_recommendations` - summarize the same as well while showing it to the user.

If user asks about a specific transform or preset:
```bash
python3 $SCRIPTS_DIR/runner.py get_transform_context \
  '{"transform_or_preset": "<name>"}'
```

**Part B — validate the chosen config:**
```bash
python3 $SCRIPTS_DIR/runner.py validate_feat_ext_data_shape \
  "{\"data_path\": \"$EFFECTIVE_DATA_PATH\", \"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"data_proc_transforms\": [\"<chosen transforms>\"], \"feature_extraction_name\": \"<chosen preset>\", \"frame_size\": <value if SimpleWindow>}"
```

Check `errors` — if `frame_size > min_sequence_length` or required transforms missing, fix before proceeding.

**Part C — generate YAML:**
```bash
python3 $SCRIPTS_DIR/runner.py generate_feat_ext_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"data_proc_transforms\": [\"<chosen transforms>\"], \"feature_extraction_name\": \"<chosen preset>\", \"frame_size\": <value if SimpleWindow>}" \
  --save-yaml $WORK_DIR/feat_ext.yaml
```

Note: only pass `frame_size` if `SimpleWindow` is in `data_proc_transforms`. Only pass `sampling_rate`/`new_sr` if `DownSample` is used.

---

## Step 7: Select model

**Part A — analyze dataset size:**
Refer `$WORK_DIR/.tmp_dataset_stats.json` to get insight on statistical information about the dataset. Basis this, proceed to **Part B (Model recommendation)**.

**Part B — get model recommendations:**
```bash
python3 $SCRIPTS_DIR/runner.py select_model_for_task \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"target_module\": \"$TARGET_MODULE\", \"variables\": $VARIABLES, \"dataset_size_bucket\": \"<from Part A>\", \"modelzoo_path\": \"$TINYML_BASE_PATH/tinyml-modelzoo/examples\"}"
```

**Part C — fetch & display all available models:**

Fetch all available models filtered by `$TASK_TYPE` and number of input variables (`$VARIABLES`):
```bash
python3 $SCRIPTS_DIR/runner.py list_available_models \
  "{\"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"modelzoo_path\": \"$TINYML_BASE_PATH/tinyml-modelzoo/tinyml_modelzoo/models/\", \"model_descriptions_path\": \"$TINYML_BASE_PATH/tinyml-modelzoo/tinyml_modelzoo/model_descriptions/\"}"
```

Display results as a **TABLE** with columns:
| Model Name | Param Count | Complexity | Merits | Demerits | Ideal Use Case |

For each model, include:
- **Merits:** Speed, accuracy, memory efficiency, special capabilities (e.g., "Fast inference on NPU", "Best accuracy")
- **Demerits:** Trade-offs (e.g., "Lower accuracy than larger models", "Requires quantization")
- **Ideal Use Case:** When/where to use (e.g., "Tight memory constraints", "Real-time inference requirement")

**Part D — provide your recommendation:**

Based on dataset size, ranked scores, and device constraints, give **CLEAR, POINT-BY-POINT recommendation** format:

"I recommend **[MODEL_NAME]** because:
- Point 1: [reason specific to this dataset size]
- Point 2: [reason specific to this device]
- Point 3: [reason specific to performance needs]

**Alternatives:**
- [ALT_MODEL] if you prioritize [property]
- [ALT_MODEL2] if you need [property]"

Inform user: "You can accept this recommendation or select any model from the table above."

Store user's choice as `MODEL_NAME`.

---

## Step 8: Generate training section
**DO NOT SKIP ANY OF THE BELOW PARTS. ENSURE STEP 8 IS FULLY DONE, EVERY PART, EVERY SINGLE TIME**
### Part A — get recommendations

```bash
python3 $SCRIPTS_DIR/runner.py get_training_recommendations \
  "{\"target_device\": \"$TARGET_DEVICE\", \"dataset_size_bucket\": \"<from Step 7A>\", \"num_gpus\": <0 or N>}"
```

Read and relay to user:
- `quantization.reason` — explains the recommended mode for their device
- `nas.reason` — explains whether NAS is viable and why

### Part B — ask user about basic training params

Ask:
- How many training epochs?
- Batch size?
- Number of GPUs? (store as `NUM_GPUS`)
- Custom learning rate?

Inform the user that all are optional, user can press `Enter` to use system defaults (taken from the corresponding modules' `params.py`).

**HIGHLY IMPORTANT**
For number of GPUs, if user does not specify or does not know, run the following simple python script:
```python
import torch

print(torch.cuda.device_count())
```
Use the result from the above script to set `NUM_GPUS`.

### Part C — ask user about quantization

Present all three modes clearly, state the recommendation, and ask the user to choose:

```
Quantization reduces model size and enables hardware acceleration.

Modes:
  0 — Float32. No compression. Largest model, slowest inference. NPU will NOT be used.
  1 — Standard PyTorch quantization. 4× smaller, works on all devices.
  2 — TI NPU-optimized. Required for NPU hardware acceleration. [RECOMMENDED for NPU devices]

The recommendation for YOUR device ($TARGET_DEVICE): mode <quantization.recommended_mode>
Reason: <quantization.reason>

When you select mode 1 or 2, Automatic Mixed Precision (AMP) is enabled by default:
<autoquant_explanation>
```

Ask:
- "Which quantization mode do you want? (0 / 1 / 2)" → store as `QUANTIZATION_MODE`

That's it. No need to ask about PTQ/QAT or bit widths — AMP handles per-layer precision automatically.

> Quantization mode will also determine the compilation preset in Step 10 — setting it correctly here matters for end-to-end correctness.

### Part D — generate YAML

Since AMP is default and handles bit widths automatically, just specify the quantization mode:
```bash
python3 $SCRIPTS_DIR/runner.py generate_training_section_yaml \
  "{\"enable\": true, \"model_name\": \"$MODEL_NAME\", \"quantization\": $QUANTIZATION_MODE}" \
  --save-yaml $WORK_DIR/training.yaml
```

Show user the generated YAML so they can confirm the training config before proceeding.

**Note:** If user explicitly wants to disable AMP and use uniform quantization instead, they can set `quantization_method` and bit widths manually, but this is not recommended and rarely needed.

---

## Step 9: Generate testing section

For most users, defaults are correct — just run:
```bash
python3 $SCRIPTS_DIR/runner.py generate_testing_section_yaml \
  '{"enable": true}' \
  --save-yaml $WORK_DIR/testing.yaml
```

Ask only if user has non-default needs:
- Skip training and test an existing model? → add `"skip_train": true, "model_path": "<path>"`
- Run on actual connected device? → add `"device_inference": true`
- Use separate test dataset? → add `"test_data": "<path>"`

---

## Step 10: Generate compilation section

**Part A — get preset recommendation (pass `QUANTIZATION_MODE` from Step 8C):**
```bash
python3 $SCRIPTS_DIR/runner.py get_compilation_preset_recommendations \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"quantization_mode\": $QUANTIZATION_MODE}"
```

The preset recommendation is driven by `QUANTIZATION_MODE`:
- `quantization_mode: 2` + NPU device → `default_preset` (NPU will be used) or `compress_npu_layer_data` (tight memory)
- `quantization_mode: 0 or 1` + NPU device → `forced_soft_npu_preset` (NPU requires mode 2 — forces CPU path)
- Non-NPU device → `default_preset`

Show user `recommended_preset` and `recommendation_reason`. Ask if they want to use it or choose differently.
Available presets: `default_preset`, `forced_soft_npu_preset`, `compress_npu_layer_data`

**Part B — generate YAML:**
```bash
python3 $SCRIPTS_DIR/runner.py generate_compilation_section_yaml \
  "{\"enable\": true, \"compile_preset_name\": \"<chosen preset>\", \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $WORK_DIR/compilation.yaml
```

---

## Step 11: Assemble config.yaml

All section YAML files should now exist in `$WORK_DIR`. Verify before assembling:
```bash
ls -la $WORK_DIR/
# Expected: common.yaml, dataset.yaml, feat_ext.yaml, training.yaml, testing.yaml, compilation.yaml
```

Assemble (using default path under tinyml-modelzoo/examples/):
```bash
python3 $SCRIPTS_DIR/runner.py generate_complete_config_file \
  "{\"task_name\": \"$TASK_NAME\", \"tinyml_base_path\": \"$TINYML_BASE_PATH\", \"common_yaml_file\": \"$WORK_DIR/common.yaml\", \"dataset_yaml_file\": \"$WORK_DIR/dataset.yaml\", \"feature_extraction_yaml_file\": \"$WORK_DIR/feat_ext.yaml\", \"training_yaml_file\": \"$WORK_DIR/training.yaml\", \"testing_yaml_file\": \"$WORK_DIR/testing.yaml\", \"compilation_yaml_file\": \"$WORK_DIR/compilation.yaml\"}"
```

Assemble (using user-specified output directory):
```bash
python3 $SCRIPTS_DIR/runner.py generate_complete_config_file \
  "{\"task_name\": \"$TASK_NAME\", \"output_dir\": \"$CONFIG_DIR\", \"common_yaml_file\": \"$WORK_DIR/common.yaml\", \"dataset_yaml_file\": \"$WORK_DIR/dataset.yaml\", \"feature_extraction_yaml_file\": \"$WORK_DIR/feat_ext.yaml\", \"training_yaml_file\": \"$WORK_DIR/training.yaml\", \"testing_yaml_file\": \"$WORK_DIR/testing.yaml\", \"compilation_yaml_file\": \"$WORK_DIR/compilation.yaml\"}"
```

Before assembling, ask user: *"Where should the final config.yaml be saved?"*
Default is `$TINYML_BASE_PATH/tinyml-modelzoo/examples/$TASK_NAME/config.yaml`.
If user provides a custom path, pass it as `"output_dir"` instead of `"tinyml_base_path"`.

```bash
CONFIG_DIR=<user choice or $TINYML_BASE_PATH/tinyml-modelzoo/examples/$TASK_NAME>
```

On success: config saved permanently to `$CONFIG_DIR/config.yaml`.
Show user the full config: `cat $CONFIG_DIR/config.yaml`

If any `*_yaml_file` is missing from WORK_DIR: go back to the relevant step and re-generate it.

---

## Step 12: Run training & compilation
**SHOW USER FULL CONFIG BEFORE ASKING USER TO ALLOW TRAINING**
Ask user: *"Ready to start training? This will take several minutes."*

Reference `references/example_running_guide.md` for monitoring tips.

**Stream logs live (recommended for long runs):**
```bash
bash $TINYML_BASE_PATH/tinyml-modelzoo/run_with_log_stream.sh \
  $CONFIG_DIR/config.yaml
```

Streams live updates. User sees training progress in real-time instead of waiting until end.

When complete, read the logs and summarize for the user:
- Training accuracy/loss: `{run_path}/training/base/run.log`
- Compiled model size and latency: `{run_path}/compilation/run.log`

From the logs, find and record:
- `RUN_ID` — the run identifier (timestamp-based directory name)
- `MODEL_ID` — the model artifact identifier
- `QUANTIZATION` — whether quantization was applied (true/false)

In case of any errors, analyse and think what could have been done differently in the config to prevent the errors. Then, suggest those modifications to the user, explain in detail the cause of the error and ask the user if your suggested modifications should be applied or if the user has anything they would like to try.
Then proceed to implement the changes (either the ones you recommended or the ones the user gave you, as per what the user chose to do) and re-run training + compilation.
DO THIS UNTIL TRAINING HAPPENS CLEANLY WITHOUT ISSUE.

**COMMON ISSUES TO LOOK OUT FOR:**
1. Feature Extraction preset failed due to size issues. Dataset's class samples may have too few entries for presets. If no presets can fit it, then think of using raw feature extraction transforms. Consult `$TINYML_BASE_PATH/tinyml-modelmaker/tinyml_modelmaker/ai_modules/timeseries/constants.py` for all available transforms and presets. See if any of them or any combination of them could be of use for your usecase and accordingly select them.

2. Due to the above issue, many times files having less number of samples (less than what the presets may expect) may be skipped. As a result the metrics you get may be skewed and show very high values. Do not get confused and report this to the user as the final metrics. Stop the training, go back to the config, and refer point 1 to fix the issue. Once fixed, then train again.

---

## Step 12B: Display compiled model memory footprint (MANDATORY)

**AFTER training completes, ALWAYS extract and display memory usage to user.**

Find and display FLASH (RO Mem) and SRAM (RW Mem) from compilation logs:

```bash
# Extract from compilation log
grep -E "FLASH|SRAM|RO Mem|RW Mem" "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/compilation/run.log"
```

**Display to user in this format:**

```
═══════════════════════════════════════════════════════════════
                    MODEL MEMORY FOOTPRINT
═══════════════════════════════════════════════════════════════
Device: $TARGET_DEVICE

  FLASH (RO Memory): <SIZE> bytes
  SRAM  (RW Memory): <SIZE> bytes

═══════════════════════════════════════════════════════════════
```

**User MUST verify this fits on their device before deployment.**

Get device specs from `references/deployment_sdk_reference.md` or:
- C2000 devices: check device datasheet (typically 256KB–1MB FLASH, 32KB–128KB SRAM)
- MSPM0 devices: check SDK documentation
- AM26x: check MCU+ SDK docs

**If model doesn't fit:**
- Stop. Do NOT proceed to deployment.
- Reduce model size: enable quantization mode 2, use "Memory" NAS optimization, or select smaller model
- Re-run training (Step 12)
- Check memory again before deployment

---

## Step 13: Deploy to device (if user asks)

**Read full guide first:** `references/device_deployment_guide.md`

**Collect from Step 12 logs:**
```bash
RUN_ID=<timestamp dir, e.g., 20240515_143022>
MODEL_ID=<model_name from config, e.g., CLS_4k_NPU>
QUANTIZATION=<true or false>
CCS_INSTALL_PATH=<user-provided, e.g., /opt/ti/ccs1260>
```

**Four scripted steps:**

### Step 13A: Verify artifacts exist
```bash
python3 $SCRIPTS_DIR/runner.py find_run_artifacts \
  "{\"tinyml_base_path\": \"$TINYML_BASE_PATH\", \"task_type\": \"$TASK_TYPE\", \"run_id\": \"$RUN_ID\", \"model_id\": \"$MODEL_ID\", \"quantization\": $QUANTIZATION}"
```
Check `success: true` before proceeding. If false, training/compilation incomplete.

### Step 13B: Create CCS project

**B1. Check SDK installed:**
```bash
python3 $SCRIPTS_DIR/runner.py check_sdk_installation \
  "{\"target_device\": \"$TARGET_DEVICE\"}"
```
Store returned `ai_examples_path` as `$AI_EXAMPLES_PATH`.

**B2. Create project (auto-copies artifacts + golden vectors):**
```bash
python3 $SCRIPTS_DIR/runner.py create_ccs_project '{
  "project_name": "<name>",
  "device_type": "<e.g., f28p55x>",
  "target_device": "$TARGET_DEVICE",
  "run_id": "$RUN_ID",
  "task_type": "$TASK_TYPE",
  "quantization": $QUANTIZATION,
  "model_id": "$MODEL_ID",
  "tinyml_base_path": "$TINYML_BASE_PATH",
  "ccs_templates_path": "$AI_EXAMPLES_PATH"
}'
```
Store returned `project_path` as `$CCS_PROJECT_PATH`.

**B3. VERIFY project structure + artifact authenticity (MANDATORY before Step C):**

**CHECKPOINT: Do NOT proceed to Step 13C until verification passes.**

**Part 1: Check file existence**

List files in project:
```bash
ls -la "$CCS_PROJECT_PATH/"
ls -la "$CCS_PROJECT_PATH/{device_type}/"
ls -la "$CCS_PROJECT_PATH/{device_type}/artifacts/"
```

Verify these files exist. All REQUIRED:
- `$CCS_PROJECT_PATH/application_main.c`
- `$CCS_PROJECT_PATH/{device_type}/artifacts/mod.a`
- `$CCS_PROJECT_PATH/{device_type}/artifacts/tvmgen_default.h`
- `$CCS_PROJECT_PATH/{device_type}/CCS/{device_type}_{project_name}.projectspec`
- `$CCS_PROJECT_PATH/{device_type}/test_vector.c`
- `$CCS_PROJECT_PATH/{device_type}/user_input_config.h`

**If any file missing:**
→ Go to manual fallback below. Do NOT skip this. Missing files = build/flash will fail.

**Part 2: Verify artifact timestamps**

The `create_ccs_project` command also validates that copied artifacts and golden vectors have matching creation timestamps with their sources in ModelMaker. This prevents:
- Files being silently copied from template (stale/wrong artifacts)
- Copy failures falling back to cached template files
- Model mismatch between what you trained and what gets deployed

**Check the create_ccs_project response for:**
- `"timestamp_validation_passed": true` — Safe to proceed. Artifacts are from your actual run.
- `"timestamp_validation_passed": false` — **CRITICAL WARNING.** Artifacts or golden vectors don't match source timestamps.

**If timestamp validation FAILS:**

1. **Do NOT proceed to Step 13C.** The project may contain template files, not your trained model.
2. Manually verify artifacts by comparing mtimes:
   ```bash
   # Source (your run)
   stat "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/compilation/artifacts/mod.a"
   
   # Destination (CCS project)
   stat "$CCS_PROJECT_PATH/{device_type}/artifacts/mod.a"
   ```
   If mtimes differ by more than 0.5 seconds, files are from template (not your actual run).

3. **Manual recovery:**
   - Delete the project: `rm -rf "$CCS_PROJECT_PATH"`
   - Re-run `create_ccs_project` and verify timestamp validation passes
   - If still fails, check that source artifacts exist and training completed successfully

**B4. Manual fallback (if auto-copy failed):**

If after checking above, any files are missing:

1. Verify artifact source paths exist:
   ```bash
   ls "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/compilation/artifacts/"
   ls "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/training/quantization/golden_vectors/"
   ```

2. Copy missing files manually:
   ```bash
   mkdir -p "$CCS_PROJECT_PATH/{device_type}/artifacts"
   cp "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/compilation/artifacts/"* \
      "$CCS_PROJECT_PATH/{device_type}/artifacts/"
   cp "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/training/quantization/golden_vectors/test_vector.c" \
      "$CCS_PROJECT_PATH/{device_type}/"
   cp "$TINYML_BASE_PATH/tinyml-modelmaker/data/projects/$TASK_TYPE/run/$RUN_ID/$MODEL_ID/training/quantization/golden_vectors/user_input_config.h" \
      "$CCS_PROJECT_PATH/{device_type}/"
   ```

3. Re-verify all files now exist:
   ```bash
   ls "$CCS_PROJECT_PATH/{device_type}/artifacts/mod.a"
   ls "$CCS_PROJECT_PATH/{device_type}/test_vector.c"
   ls "$CCS_PROJECT_PATH/{device_type}/user_input_config.h"
   ```

4. **Only after re-verification passes, proceed to Step 13C.**


### Step 13C: Build project
```bash
python3 $SCRIPTS_DIR/runner.py build_ccs_project \
  "{\"ccs_project_path\": \"$CCS_PROJECT_PATH\", \"ccs_install_path\": \"$CCS_INSTALL_PATH\"}"
```
If fails: build manually in CCS (**Project → Build Project**, Ctrl+B).

### Step 13D: Flash to device
Connect device via USB/JTAG, then:
```bash
python3 $SCRIPTS_DIR/runner.py flash_ccs_project \
  "{\"ccs_project_path\": \"$CCS_PROJECT_PATH\", \"ccs_install_path\": \"$CCS_INSTALL_PATH\"}"
```

If fails: flash manually in CCS (**Run → Flash Project**).

**After flashing:** In CCS Debug perspective, set breakpoint after inference, check `test_result == 1` in Watch window (1 = pass, 0 = fail).

See `references/device_deployment_guide.md` for troubleshooting, device/SDK mappings, and full walkthrough.
