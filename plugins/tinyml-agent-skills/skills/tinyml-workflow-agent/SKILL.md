---
name: tinyml-workflow-agent
description: Guides users through end-to-end Tiny ML model creation, training, compilation, and device deployment. Use when user mentions creating AI solutions, AI models, training for embedded devices, deploying to MCUs, or any tiny ML workflow. Triggers on **any workflow step** — dataset analysis, model selection, feature extraction, compilation, deployment. Can start from Step 1 or jump to any intermediate step. If user has a dataset, artifacts, code, metadata, or is at any point in the workflow, this skill handles it. Always confirm tinyml-modelzoo installation path before proceeding.
---

# Tiny ML Tensorlab Skill

## Overview 

Build, train, compile, and deploy ML models to embedded MCUs using Tiny ML tensorlab.

---

## CRITICAL: Script Location

`runner.py` lives in **this skill's `scripts/` directory**, not in tinyml-modelzoo.

```
SCRIPTS_DIR = ~/.claude/plugins/marketplaces/<plugin>/skills/tinyml-workflow-agent/scripts/
```

Session Setup step 1 shows how to discover this path automatically.

---

## CRITICAL RULES — BLOCKING GATES

These two rules apply unconditionally. Read them before any workflow step. They exist because compiler environment state does not persist across shell sessions, and because proceeding to deployment with missing or failed compilation artifacts causes silent, hard-to-debug failures downstream.

### Rule 1: Compiler path MUST be exported in the same shell session as the training script

Before running `run_tinyml_modelzoo.sh`, locate and export the compiler path **in the exact same `bash` call that runs the script.** Exporting env vars in a prior step is not sufficient — shell state does not carry across separate tool calls.

Use `ccs-project → getCompilers` (preferred — CCS already knows what's installed):

```
Call ccs-project → getCompilers
```

From the returned list, find the entry matching `$TARGET_DEVICE` family (e.g., `TI C2000 CGT` for F28xxx, `TI ARM CGT` or `tiarmclang` for MSPM0/AM26x). Extract its installation path. Strip to the root directory containing `bin/`. Store as `CGT_ROOT`.

**If `ccs-project` MCP is unavailable**, fall back to filesystem search:
```bash
# C2000 CGT
find /opt/ti ~/ti $HOME/ti /usr/local/opt/ti 2>/dev/null -name "cl2000" -type f 2>/dev/null | head -1
# ARM CGT
find /opt/ti ~/ti $HOME/ti /usr/local/opt/ti 2>/dev/null -name "armcl" -type f 2>/dev/null | head -1
```

Strip the binary name → get `bin/` → strip `bin/` → get `CGT_ROOT`.

Then export and run in one command:

```bash
export C2000_CGT_PATH=<CGT_ROOT>   # or ARM_CGT_PATH, per device — see references/example_running_guide.md
cd "$TINYML_MODELZOO_PATH" && ./run_tinyml_modelzoo.sh "$CCS_PROJECT_PATH/config.yaml"
```

**If compiler not found by either method:** STOP immediately. Ask the user:
> "Compiler not found automatically. Please provide the full path to your CGT root directory (the folder containing `bin/cl2000` or `bin/armcl`)."

Do NOT run the training script without the compiler path confirmed and exported in the same call.

---

### Rule 2: Compiled artifacts and memory footprint are required before proceeding to deployment

After `run_tinyml_modelzoo.sh` completes, you MUST verify BOTH of the following before moving to Step 14 (deployment) or any other subsequent step:

1. **Compiled artifacts exist and are non-empty:**
   ```bash
   ls -la "$CCS_PROJECT_PATH/compilation/artifacts/"
   ```
   `mod.a` must be present. An empty or missing directory means compilation failed.

2. **Memory footprint summary extracted and shown to user** — Code, RO Data, RW Data, and Total byte counts from the compilation log (Step 13A). User must acknowledge these numbers fit the target device before deployment proceeds.

**If either check fails:** surface the full error, stop, and do NOT proceed to Step 14. A failed or incomplete compilation is never a reason to skip to deployment. Fix training/compilation first (Step 13 error handling), then re-verify.

---

**Workflow (14 steps):**
1. **Session setup** — load `.env`, activate venv, check for updates. If not set up, run `/tinyml-agent-skills:setup` first.
2. **Requirements** — task type and device only.
3. **Early CCS Import** — Find SDK, locate AI examples, import template project into CCS workspace. Set `CCS_PROJECT_PATH`.
3.5. **Check for matching examples** — if match found in modelzoo, offer: (a) use bundled config + dataset, (b) use config as template with custom dataset, (c) build from scratch.
4. **Understand dataset** — if no match or user chose "build from scratch", ask dataset path and task name.
5. **Dataset validation** — validate format, get effective path
6. **Dataset section** — generate YAML → save to `$CCS_PROJECT_PATH`
7. **Feature extraction and Data Processing transforms**
   - Step 7A: Analyze dataset for statistical insights
   - Step 7B: Get recommendations → generate YAML → save to `$CCS_PROJECT_PATH`
8. **Model selection** — analyze dataset size → rank models
9. **Training section** — model name + hyperparams → save to `$CCS_PROJECT_PATH`; output path set to `$CCS_PROJECT_PATH/training/`
10. **Testing section** — testing config → save to `$CCS_PROJECT_PATH`
11. **Compilation section** — preset selection → save to `$CCS_PROJECT_PATH`; output path set to `$CCS_PROJECT_PATH/compilation/`
12. **Assemble config** — combine all sections → write `$CCS_PROJECT_PATH/config.yaml`
13. **Run training** — execute run_tinyml_modelzoo.sh with `$CCS_PROJECT_PATH/config.yaml`
14. **Deploy to device** — copy trained artifacts to project → exclude build folders → build → flash
**Mid-workflow entry:** If the user already has artifacts from a prior step (a config file, a trained model, compiled artifacts), skip directly to the relevant step — do NOT restart from Step 2. Common jump points:
- "I want to use an existing example" → Step 3.5 (after CCS import, pick option 1, 2, or 3)
- "I have a config already" → skip to Step 13 (run training)
- "I have a trained/compiled model, just deploy" → skip to Step 14 (set RUN_ID, MODEL_ID, QUANTIZATION from the user, then proceed)
- "I have a dataset, just build the config" → start from Step 4 (after CCS import in Step 3)
- "Just run training on my existing config" → skip to Step 13

**Session Setup (the checks above — load .env, activate venv, CCS MCP probe) must always run before jumping to any step.** This is a lightweight check, not a re-run of `/tinyml-agent-skills:setup`. If `.env` is missing or incomplete, direct the user to run `/tinyml-agent-skills:setup` first — do not re-run setup yourself.

**IMPORTANT: After EACH step which generates a section of the config file, pause and show the user the config file (created thus far) and proceed only with user's approval of the config.**
**Reference guides** (read on demand, not upfront):
- `references/config_creation_guide.md` — task types, devices, YAML rules
- `references/example_running_guide.md` — run commands, monitoring, troubleshooting
- `references/device_deployment_guide.md` — CCS project, flashing, validation

---
## Session Setup (do this before Step 1) ***NEVER SKIP THIS***

### 1. Discover SCRIPTS_DIR

Find the tinyml-workflow-agent runner (works on all platforms — search under `~/.claude`):
```bash
find ~/.claude -name "runner.py" 2>/dev/null | grep "tinyml-workflow-agent" | head -1
```
Strip `/runner.py` from the result to get `SCRIPTS_DIR`.

If not found, ask the user where the tinyml-agent-skills plugin is installed.

### 2. Load environment

Check if `.env` exists at `~/.tinyml-agent-skills/.env` (same location on all platforms — Linux, macOS, Windows).

If it exists, load and export these variables:

| Variable                    | Description                              |
|-----------------------------|------------------------------------------|
| `IS_REPO_SETUP`             | Indicates if tinyml-modelzoo is setup    |
| `TINYML_MODELZOO_PATH`      | Root of tinyml-modelzoo clone            |
| `TINYML_MODELMAKER_PATH`    | Installed tinyml_modelmaker package root |
| `TINYML_TINYVERSE_PATH`     | Installed tinyml_tinyverse package root  |
| `TINYML_MODELOPT_PATH`      | Installed tinyml_torchmodelopt package root |

**Inform the user:**
> ".env loaded from: `~/.tinyml-agent-skills/.env`"
> "Session variables ready: IS_REPO_SETUP, TINYML_MODELZOO_PATH, TINYML_MODELMAKER_PATH"

**If `.env` is missing or any variable is unset:** stop and tell the user:
> "Run `/tinyml-agent-skills:setup` first. It will create `~/.tinyml-agent-skills/.env` with your configuration."
Do not proceed until setup is done.

### 3. Activate virtual environment

Check if a venv exists at the standard location inside the modelzoo directory:
```bash
ls "$TINYML_MODELZOO_PATH/.venv/bin/activate" 2>/dev/null && echo "venv found" || echo "no venv"
```

- **venv found** — activate it and set `VENV_PYTHON`:
  ```bash
  source "$TINYML_MODELZOO_PATH/.venv/bin/activate"
  VENV_PYTHON="$TINYML_MODELZOO_PATH/.venv/bin/python3"
  ```
- **no venv** — setup was not completed or the venv was deleted. Stop and tell the user to run `/tinyml-agent-skills:setup` first — it will create the venv and install all dependencies.

**`VENV_PYTHON` must be used for ALL subsequent `python3` calls this session** — venv activation does not persist across separate shell tool calls, so explicit binary path is required.

### 4. Check for updates
c/`) poi
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_update_status '{}'
```

- **`initialized: false`** → tell user to run `/tinyml-agent-skills:setup` first. Stop.
- **`mode: "auto"`** → run `$VENV_PYTHON $SCRIPTS_DIR/runner.py check_updates '{}'`
  - `update_available: true` → show message from check_updates output and ask: "Update now?"
    - Yes: `$VENV_PYTHON $SCRIPTS_DIR/runner.py do_update '{"tinyml_modelzoo_path": "$TINYML_MODELZOO_PATH"}'`
    - No: proceed with current version
  - `update_available: false` → proceed normally
- **`mode: "pinned"`** → proceed normally

### 5. Check CCS MCP availability

Probe both CCS MCP servers with lightweight read-only calls:

```
Call ccs-project → getCompilers    (no params)
Call ccs-debug   → getDebugSessions (no params)
```

Record results in session state:

| Variable            | Set when                              |
|---------------------|---------------------------------------|
| `CCS_PROJECT_MCP`   | `true` if `ccs-project` responded     |
| `CCS_DEBUG_MCP`     | `true` if `ccs-debug` responded       |

**If both available:** tell user "CCS MCP servers detected — build and flash will use MCP."
**If none available:** tell user "CCS MCP not detected — build/flash will use headless dslite. Open CCS if you want MCP-based deployment."
**If partial:** note which is missing and which deployment sub-steps will fall back.

Do NOT stop the workflow if MCPs are unavailable — training and compilation proceed regardless. Only Step 14D/14E behavior changes based on these flags.

Config files are created in the **user's working directory** (not in `tinyml-modelzoo/examples/`). All artifacts and outputs are saved relative to this directory, allowing complete isolation of projects.

**Session state — track these throughout all steps. `.env` variables are pre-loaded from setup. Remaining variables set progressively; do NOT write them to `.env`.**

| Variable                    | Source           | Description                              |
|-----------------------------|------------------|------------------------------------------|
| `SCRIPTS_DIR`               | discovered       | Absolute path to scripts/ folder         |
| `VENV_PYTHON`               | Session Setup 3  | Explicit venv python binary — use instead of bare `python3` |
| `IS_REPO_SETUP`             | ~/.tinyml-agent-skills/.env | tinyml-modelzoo setup flag    |
| `TINYML_MODELZOO_PATH`      | ~/.tinyml-agent-skills/.env | Root of tinyml-modelzoo clone |
| `TINYML_MODELMAKER_PATH`    | ~/.tinyml-agent-skills/.env | Installed tinyml_modelmaker package root |
| `TINYML_TINYVERSE_PATH`     | ~/.tinyml-agent-skills/.env | Installed tinyml_tinyverse package root |
| `TINYML_MODELOPT_PATH`      | ~/.tinyml-agent-skills/.env | Installed tinyml_torchmodelopt package root |
| `CCS_PROJECT_MCP`           | Session Setup 5  | true/false — ccs-project MCP available   |
| `CCS_DEBUG_MCP`             | Session Setup 5  | true/false — ccs-debug MCP available     |
| `TASK_NAME`                 | Step 2  | Slug used for config subdirectory        |
| `WORK_DIR`                  | Step 2  | Temp dir for intermediate section YAMLs (deprecated, use CCS_PROJECT_PATH) |
| `USER_WORKING_DIR`          | Step 2  | User's project dir where config.yaml is saved (deprecated, use CCS_PROJECT_PATH) |
| `CCS_PROJECT_PATH`          | Step 3  | Imported CCS project root — all config, training, compilation outputs here |
| `TASK_TYPE`                 | Step 2  | e.g., `motor_fault`                      |
| `TARGET_DEVICE`             | Step 2  | e.g., `F28P55`                           |
| `TARGET_MODULE`             | Step 4  | `timeseries` or `vision`                 |
| `VARIABLES`                 | Step 4  | Number of sensor channels                |
| `DATA_PATH`                 | Step 4  | User's raw dataset path                  |
| `EFFECTIVE_DATA_PATH`       | Step 5  | Path after any auto-reorganization       |
| `QUANTIZATION_MODE`         | Step 11C | 0/1/2 — drives compilation preset       |
| `NAS_ENABLED`               | Step 11D | true/false                               |

---

## Error Handling Rule

**After every runner.py call:** check `success` in the JSON output.
- `"success": true` → proceed to next step
- `"success": false` → read `errors` array, fix the issue, re-run the same step

Never skip to the next step while `success` is false. The `errors` field always describes what went wrong and what to fix.

---

## Step 1: Session setup

Complete the "Session Setup" section above. Do not proceed until:
- `IS_REPO_SETUP` is `true` or `1`
- All variables set: `SCRIPTS_DIR`, `IS_REPO_SETUP`, `TINYML_MODELZOO_PATH`, `TINYML_MODELMAKER_PATH`, `TINYML_TINYVERSE_PATH`, `TINYML_MODELOPT_PATH`
- Virtual environment is activated (or confirmed not needed)
- **[MANDATORY — every session, no exceptions]** Session Setup step 4 ("Check for updates") completed
- **[MANDATORY — every session, no exceptions]** Session Setup step 5 ("Check CCS MCP availability") completed — `CCS_PROJECT_MCP` and `CCS_DEBUG_MCP` set
---

## Step 2: Understand requirements

Ask ONLY these two questions. Do NOT ask about dataset or task name — those come later, only if needed.

1. **Task type** — Ask the user what kind of ML task they want to build. Then read the exact valid task type strings from modelmaker source and show them so the user picks precisely:
   ```bash
   grep -A 50 "TASK_TYPE_TO_MODULE" "$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py"
   ```
   For vision tasks also check: `$TINYML_MODELMAKER_PATH/ai_modules/vision/constants.py`
   **Do NOT infer `task_type` from domain knowledge alone — always look up the defined strings from the source above and confirm with the user.**

2. **Target device** — Ask which MCU. Read valid device names from the same source:
   ```bash
   grep -E "TARGET_DEVICE|ALL_TARGET|SUPPORTED_DEVICE" "$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py" | head -40
   ```

**Task-type disambiguation example:**
```
User: "motor fault detection"
→ Ask: "Classify fault vs healthy (motor_fault / generic_timeseries_classification)
        or detect anomalies (generic_timeseries_anomalydetection)?"
```

Set:
```bash
TASK_TYPE=<answer>
TARGET_DEVICE=<answer>
```

**Proceed immediately to Step 3. Step 3 imports an SDK template into CCS. No dataset questions yet.**

---

## Step 3: Early CCS Import — Import SDK template project

**Run this immediately after getting TASK_TYPE, TASK_NAME and TARGET_DEVICE.**

This step imports an AI example template from the SDK into a new CCS project in the user's workspace. The project becomes the home for all config files, training outputs, and compilation artifacts.

### Step 3A: Find SDK root and locate AI examples

Use `ccs-project → getProducts` (preferred — CCS knows all installed products):

```
Call ccs-project → getProducts
```

From the returned list, find the product matching `$TARGET_DEVICE` family (e.g., C2000Ware for F28xxx, MSPM0 SDK for MSPM0G, MCU+ SDK for AM26x). Extract its installation path as `$SDK_ROOT`.

**If `ccs-project` MCP is unavailable**, fall back to the runner:
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py check_sdk_installation '{"target_device": "$TARGET_DEVICE"}'
```
Store returned `sdk_root` as `$SDK_ROOT`. If `found: false` → show errors (includes download URL), stop until user installs SDK.

Then locate AI examples — do NOT assume a fixed subpath, it varies across SDK families and versions:
```bash
find "$SDK_ROOT" -maxdepth 7 -type d \( -name "ai" -o -name "edgeAI" -o -name "edge_ai" \) 2>/dev/null
```

From the results, identify the directory containing task-like example subdirectories (e.g., `generic_timeseries_classification/`, `motor_fault_ai_npu/`). If multiple candidates, pick the one whose subdirectory names best match task and device. Store as `$AI_EXAMPLES_PATH`.

### Step 3B: Select template

**Template selection within `$AI_EXAMPLES_PATH` — priority order:**

1. **Exact match:** look for a subdirectory whose name contains `$TASK_TYPE` (e.g., `motor_fault_ai_npu` for `motor_fault`).
2. **Generic fallback:** if no exact match, use the generic timeseries template for the task family:

| Task family (`$TASK_TYPE` pattern) | Fallback template directory |
|------------------------------------|-----------------------------|
| `*classification*` | `generic_timeseries_classification` |
| `*anomalydetection*` | `generic_timeseries_anomalydetection` |
| `*regression*` | `generic_timeseries_regression` |
| `*forecasting*` | `generic_timeseries_forecasting` |

Always prefer exact match; only fall back to generic when no subdirectory name contains `$TASK_TYPE`. If generic fallback also absent, stop and tell user — do not guess a different template.

If no directory found: SDK may not include AI examples. Tell user and check SDK version.

Store selected template directory as `$TEMPLATE_DIR` (full path: `$AI_EXAMPLES_PATH/<selected_subdir>`).

### Step 3C: Read template README and confirm requirements

The template directory contains a README. Find and read it:
```bash
find "$TEMPLATE_DIR" -maxdepth 2 -name "README*" -o -name "readme*" 2>/dev/null | head -3
```

Read the README file(s) found. Extract any mentions of:
- Hardware requirements (booster packs, LaunchPads, EVM boards, sensors, specific board revisions)
- Software prerequisites (SDK versions, driver installations, firmware updates)
- Physical connections or jumper settings
- Any other prerequisites before the example can run

Present findings to the user before proceeding:
> "Before importing this project, here's what this example requires:
> **Hardware:** [list any boards, booster packs, sensors mentioned]
> **Software:** [list any SDK versions, drivers, firmware mentioned]
> **Other:** [anything else]
>
> Do you have everything listed above? If not, please set it up before we continue."

**Wait for user confirmation before proceeding to Step 3D.** If the user is missing something, stop and help them resolve it first — do not import the project into CCS until they confirm all requirements are met.

If no README is found or it has no requirements section: skip this check silently and proceed to Step 3D.

### Step 3D: Import template into CCS workspace

Ask user: *"Where should this CCS project be created? (e.g., `/opt/ti/ccs/workspace/my_ml_project` or just the project name and we'll use your default workspace)"*

Set `$WORKSPACE_DIR` and `$PROJECT_NAME` from user's answer.

Use `ccs-project → importProject` to import the selected SDK template:

```
Call ccs-project → importProject
  location:    "$TEMPLATE_DIR/CCS/<device_type>_<template_name>.projectspec"
  destination: "$WORKSPACE_DIR/$PROJECT_NAME"
```

From the response, save:
- `project_path` → `$CCS_PROJECT_PATH`
- The project name used → `$PROJECT_NAME`

**If the template was a generic fallback** (template name ≠ task_type), rename the project to match:
```
Call ccs-project → renameProject
  old name: <template_name>
  new name: <task-specific name>
```
Update `$PROJECT_NAME` to the new name.

Confirm import: call `ccs-project → getProjectDescriptors` — the project must appear by name before proceeding.

**If import fails**, diagnose before acting:
- `"file not found"` / bad path → template path from Step 3B is wrong; fix `$AI_EXAMPLES_PATH` / template selection and retry. Do NOT fall back.
- `"MCP not available"` / `"CCS not running"` / tool missing → MCP cannot be used. **STOP. Tell the user:**
  > "CCS MCP is not available ([exact reason]). The skill requires CCS to be running with the ccs-project MCP enabled. Please open CCS and retry."
  Do not proceed without CCS MCP.

### Step 3 Complete

Set `CCS_PROJECT_PATH` for use in all subsequent steps. All config files, training outputs, and compilation artifacts will be saved under `$CCS_PROJECT_PATH/`.

---

## Step 3.5: Check for matching examples in **modelzoo**

**Run immediately after CCS project is imported.**

List all examples in modelzoo to find reference configs:
```bash
ls $TINYML_MODELZOO_PATH/examples/
```

For each example directory, read its `config.yaml` and check whether **both** of the following match:
1. `task_type` matches `$TASK_TYPE`
2. `target_device` matches `$TARGET_DEVICE`

Dataset URL/path is NOT a required match criterion — users describe use cases semantically, not by dataset URL.

**If one or more examples match:**

Show the user the matching example name(s) and briefly what each does (from its config/readme), then ask:
> "I found an existing example that matches your task and device: `<example_name>`. How would you like to proceed?
> 1. **Use the example as-is** — run training with the bundled example dataset (fastest path, no custom data needed)
> 2. **Use as template, bring my own dataset** — copy example config but point it at your data
> 3. **Build from scratch** — ignore the example, configure everything custom"

**Option 1 — Use example as-is (bundled dataset + config from modelzoo):**

1. Copy example config to CCS project:
   ```bash
   EXAMPLE_PATH="$TINYML_MODELZOO_PATH/examples/<example_name>"
   cp "$EXAMPLE_PATH/config.yaml" "$CCS_PROJECT_PATH/config.yaml"
   ```
2. Update output paths to CCS project (relative paths):
   ```bash
   $VENV_PYTHON -c "
  import yaml
  path = '$CCS_PROJECT_PATH/config.yaml'
  with open(path) as f: cfg = yaml.safe_load(f)
  cfg.setdefault('training', {})['train_output_path'] = './training'
  cfg.setdefault('compilation', {})['compile_output_path'] = './compilation'
  with open(path, 'w') as f: yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
  print('Done.')
  "
   ```
3. Dataset is already defined in the example config (bundled URL or path). Skip Steps 4–12, jump directly to **Step 13** (run training with bundled dataset)

**Option 2 — Use example config as template with custom dataset:**

1. Ask user: dataset path, channel count, task name
2. Copy example config to CCS project (as in Option 1 steps 1–2)
3. Update `dataset.input_data_path` to user's dataset:
   ```bash
   $VENV_PYTHON -c "
  import yaml
  path = '$CCS_PROJECT_PATH/config.yaml'
  with open(path) as f: cfg = yaml.safe_load(f)
  cfg.setdefault('dataset', {})['input_data_path'] = '$USER_DATASET_PATH'
  cfg.setdefault('common', {})['variables'] = $VARIABLES
  with open(path, 'w') as f: yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
  print('Done.')
  "
   ```
4. Skip Steps 4–12, jump directly to **Step 13** (run training with custom dataset + example config)

**Option 3 — Build from scratch (ignore example):**
Proceed directly to **Step 4** (understand dataset and task name).

**If no example matches:**
Skip this step silently and proceed to **Step 4**.

---

## Step 4: Understand dataset and task name

**Run if no matching example was found (Step 3.5) or if user chose "build from scratch".**

Ask these questions:

- **Dataset** — "Where is your dataset? (local path or URL)" and "How many sensor channels/variables does it have?"
- **Task name** — optional: pick a name for the config, e.g., `motor_fault_demo` (used in `config.yaml` metadata)

Set:
```bash
VARIABLES=<answer>
DATA_PATH=<answer>
TASK_NAME=<answer or generic>
```

**Important:** All configs and artifacts are created in `$CCS_PROJECT_PATH` (the imported CCS project root), not in modelzoo/examples. Original example configs are never touched.

---

## Step 5: Generate common section

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_common_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $CCS_PROJECT_PATH/common.yaml
```

- On `success: true`: note `inferred_module` (set as `TARGET_MODULE`)
- On `success: false`: show `errors`, consult `$TINYML_MODELMAKER_PATH/ai_modules/timeseries/constants.py` to verify valid `task_type` and `target_device` values, correct and retry

```bash
# Optional: custom run_name
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_common_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"run_name\": \"{date-time}/{model_name}\"}" \
  --save-yaml $CCS_PROJECT_PATH/common.yaml
```

---

## Step 6: Validate dataset format

This step checks the dataset directory structure and auto-fixes it if possible.
It returns `effective_input_data_path` — which may differ from `DATA_PATH` if data was reorganized.

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py validate_dataset_section \
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

## Step 7: Generate dataset section

**Always ask user for the below params:**
- Split type: `amongst_files` or `within_files`
**IMPORTANT: If after formatting, each class has only ONE csv(/txt/npy or any supported format) file, then split_type of `within_files` MUST be used**
- Split factor: e.g.`[0.6, 0.3, 0.1]` -> THIS IS JUST AN EXAMPLE

**IMPORTANT: If user does not know or does not specify, then create WITHOUT the above two params. Inform user that they will be picked from params.py (default values).**

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_dataset_section_yaml \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$EFFECTIVE_DATA_PATH\"}" \
  --save-yaml $CCS_PROJECT_PATH/dataset.yaml
```

With optional split params:
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_dataset_section_yaml \
  "{\"enable\": true, \"dataset_name\": \"my_dataset\", \"input_data_path\": \"$EFFECTIVE_DATA_PATH\", \"split_type\": \"USER_GIVEN_SPLIT_TYPE\" \"split_factor\": {USER_GIVEN_SPLIT_FACTOR}}" \
  --save-yaml $CCS_PROJECT_PATH/dataset.yaml
```

---

## Step 8A: Analyze dataset for statistical insights
Run:
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py analyse_dataset \
  "{\"formatted_dataset_path\": \"$EFFECTIVE_DATA_PATH\", \"task_family\": \"{classification/anomalydetection/regression/forecasting}\"}"
```
Use the output for downstream tasks - particularly for the feature extraction presets/transforms, data processing presets/transforms, and model selection recommendations. To persist the results for the rest of the session, follow the below steps:
1. Create a temporary file titled `.tmp_dataset_stats.json` within `$CCS_PROJECT_PATH`. 
2. Save the following structure to `$CCS_PROJECT_PATH/.tmp_dataset_stats.json`. Include only the `data_distribution` key that matches the task family:

```json
{
  "result": {
    "dataset_bucket": "<tiny|small|medium|large>",
    "dataset_size": <total_num_samples>,
    "task_type": "<TASK_TYPE>",
    "min_sample_or_seq_length": <from analyse_dataset output>,
    "data_distribution": {
      "<class_or_file_name>": <count>,
      "..."
    },
    "formatted_dataset_path": "<path to formatted dataset>"
  }
}
```

`data_distribution` key names by task family:
- **classification**: `class_<name>` → sample count per class
- **anomalydetection**: `normal` and `anomalous` → counts
- **regression**: `<filename>` → row count per file
- **forecasting**: `<sequence_name>` → entry count per sequence

## Step 8B: Generate feature extraction and Data Processing transforms

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
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_data_proc_feat_ext_recommendations \
  "{\"task_type\": \"$TASK_TYPE\", \"prefer_fft\": <based-on-above-analysis>, \"need_full_spectrum\": <based-on-above-analysis>, \"need_temporal_ctx\": <based-on-above-analysis>, \"shortest_sample_len\": <min_sample_or_seq_length from .tmp_dataset_stats.json>, \"variables\": $VARIABLES}"
```

**`shortest_sample_len` — critical: this is the MEASURED length of the shortest sample in the dataset (rows in the shortest CSV file). It is NOT a minimum requirement — it is an observed constraint. Any preset or frame_size larger than this value will silently skip that file during training, causing silent data loss. The ranker hard-excludes presets where frame_size > shortest_sample_len. Never recommend a frame_size larger than this value.**

Also, understand what each recommended transform does - go through `references/FE_and_Data_Processing_Transforms/FE_transforms.md` and `references/Data_processing_transforms.md`. For presets, consult `$(python3 -c "import tinyml_modelmaker, os; print(os.path.join(os.path.dirname(tinyml_modelmaker.__file__), 'ai_modules', 'timeseries', 'constants.py'))")` (source of truth for feature extraction and data processing presets). 
Try understanding whether your recommendations actually will be useful for the dataset you are working with and if so, why. **Give CLEAR point-by-point reasoning to the user regarding why your recommended transforms or presets are valid and useful.**

Show user the complete, structured output of `get_data_proc_feat_ext_recommendations` - summarize the same as well while showing it to the user.

If user asks about a specific transform or preset, read `references/FE_and_Data_Processing_Transforms/FE_transforms.md` or `references/FE_and_Data_Processing_Transforms/Data_processing_transforms.md` for details.

**Part B — validate the chosen config:**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py validate_feat_ext_data_shape \
  "{\"data_path\": \"$EFFECTIVE_DATA_PATH\", \"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"data_proc_transforms\": [\"<chosen transforms>\"], \"feature_extraction_name\": \"<chosen preset>\", \"frame_size\": <value if SimpleWindow>}"
```

Check `errors` — if `frame_size > shortest_sample_len` or required transforms missing, go back and choose a smaller preset/frame_size before proceeding.

**Part C — generate YAML:**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_feat_ext_section_yaml \
  "{\"task_type\": \"$TASK_TYPE\", \"variables\": $VARIABLES, \"data_proc_transforms\": [\"<chosen transforms>\"], \"feature_extraction_name\": \"<chosen preset>\", \"frame_size\": <value if SimpleWindow>}" \
  --save-yaml $CCS_PROJECT_PATH/feat_ext.yaml
```

Note: only pass `frame_size` if `SimpleWindow` is in `data_proc_transforms`. Only pass `sampling_rate`/`new_sr` if `DownSample` is used.

---

## Step 8: Select model

**Part A — analyze dataset size:**
Refer `$CCS_PROJECT_PATH/.tmp_dataset_stats.json` to get statistical insight on the dataset (size bucket, sample count, distribution). This informs model complexity guidance in Parts B/C.

**Part B — find best matching example:**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py select_model_for_task \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\", \"target_module\": \"$TARGET_MODULE\", \"variables\": $VARIABLES}"
```

Check `has_good_match` in the result:
- **`has_good_match: true`** → a matching example was found. Use `recommended_model` as the starting recommendation. Show the `ranked_matches` table to the user so they can see alternatives.
- **`has_good_match: false`** → no example matched the task type. Proceed to Part C.

**Part C — agent-discovery fallback (only when `has_good_match` is false):**

Read the model_descriptions files from the path in `result.fallback.model_descriptions_path`. Select the file matching the task family:
- Classification tasks → `classification.py`
- Anomaly detection → `anomalydetection.py`
- Regression → `regression.py`
- Forecasting → `forecasting.py`

Read the file(s), identify models that support `$TASK_TYPE` and `$TARGET_DEVICE` (check `target_devices` blocks and `enabled_models_list`). Use dataset stats from `$CCS_PROJECT_PATH/.tmp_dataset_stats.json` to reason about model complexity:
- Small datasets (`dataset_bucket: tiny/small`) → prefer models with fewer parameters (simpler models reduce overfitting)
- Large datasets (`dataset_bucket: large`) → any complexity is fine

**Part D — display options and provide your recommendation:**

Display available models as a **TABLE**:
| Model Name | Param Count | Complexity | Merits | Demerits | Ideal Use Case |

For each model, include:
- **Merits:** Speed, accuracy, memory efficiency, special capabilities (e.g., "Fast inference on NPU", "Best accuracy")
- **Demerits:** Trade-offs (e.g., "Lower accuracy than larger models", "Requires quantization")
- **Ideal Use Case:** When/where to use (e.g., "Tight memory constraints", "Real-time inference requirement")

Give a **CLEAR, POINT-BY-POINT recommendation**:

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

## Step 9: Generate training section
**Complete ALL parts A–D below every time. Do not skip any.**
### Part A — get recommendations

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_training_recommendations \
  "{\"target_device\": \"$TARGET_DEVICE\", \"dataset_size_bucket\": \"<from Step 7A>\", \"num_gpus\": <0 or N>}"
```

Read and relay to user:
- `quantization.reason` — explains the recommended mode for their device
- `nas.reason` — explains whether NAS is viable and why

### Part B — ask user about basic training params

**MANDATORY — do NOT skip or assume defaults. Always ask ALL four questions explicitly before generating any training YAML.**

First read the actual default values from params.py:
```bash
grep -E "training_epochs|batch_size|learning_rate" "$TINYML_MODELMAKER_PATH/ai_modules/$TARGET_MODULE/params.py" | head -20
```

Then ask the user each of the following, showing the real default value from params.py inline:
1. How many training epochs? *(default: \<from params.py\>)*
2. Batch size? *(default: \<from params.py\>)*
3. Number of GPUs?
4. Custom learning rate? *(default: \<from params.py\>)*

Tell them all are optional — pressing Enter keeps the shown default. Wait for responses to all four before proceeding.

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

> Quantization mode will also determine the compilation preset in Step 11 — setting it correctly here matters for end-to-end correctness.

### Part D — generate YAML

Since AMP is default and handles bit widths automatically, just specify the quantization mode:
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_training_section_yaml \
  "{\"enable\": true, \"model_name\": \"$MODEL_NAME\", \"quantization\": $QUANTIZATION_MODE}" \
  --save-yaml $CCS_PROJECT_PATH/training.yaml
```

**IMPORTANT: After generation, update training output path to ABSOLUTE path:**
```bash
# Edit $CCS_PROJECT_PATH/training.yaml and set:
# train_output_path: '$CCS_PROJECT_PATH/training'
# (NOT relative './training' — this ensures outputs go to CCS project, not modelzoo)
```

Show user the generated YAML so they can confirm the training config before proceeding.

**Note:** If user explicitly wants to disable AMP and use uniform quantization instead, they can set `quantization_method` and bit widths manually, but this is not recommended and rarely needed.

---

## Step 10: Generate testing section

For most users, defaults are correct — just run:
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_testing_section_yaml \
  '{"enable": true}' \
  --save-yaml $CCS_PROJECT_PATH/testing.yaml
```

Ask only if user has non-default needs:
- Skip training and test an existing model? → add `"skip_train": true, "model_path": "<path>"`
- Run on actual connected device? → add `"device_inference": true`
- Use separate test dataset? → add `"test_data": "<path>"`

---

## Step 11: Generate compilation section

**Part A — get preset recommendation (pass `QUANTIZATION_MODE` from Step 9C):**
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py get_compilation_preset_recommendations \
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
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_compilation_section_yaml \
  "{\"enable\": true, \"compile_preset_name\": \"<chosen preset>\", \"target_device\": \"$TARGET_DEVICE\"}" \
  --save-yaml $CCS_PROJECT_PATH/compilation.yaml
```

**IMPORTANT: After generation, update compilation output path to ABSOLUTE path:**
```bash
# Edit $CCS_PROJECT_PATH/compilation.yaml and set:
# compile_output_path: '$CCS_PROJECT_PATH/compilation'
# (NOT relative './compilation' — this ensures outputs go to CCS project, not modelzoo)
```

---

## Step 12: Assemble config.yaml

All section YAML files should now exist in `$CCS_PROJECT_PATH`. Verify before assembling:
```bash
ls -la $CCS_PROJECT_PATH/
# Expected: common.yaml, dataset.yaml, feat_ext.yaml, training.yaml, testing.yaml, compilation.yaml
```

**CRITICAL: Before assembling, update training and compilation output paths to ABSOLUTE paths.**

This ensures outputs go to CCS project even though `run_tinyml_modelzoo.sh` runs from modelzoo directory.

Read the training and compilation YAML files and update their paths to absolute paths (use **full `$CCS_PROJECT_PATH`**, NOT relative paths):
```bash
# In $CCS_PROJECT_PATH/training.yaml:
# train_output_path: '$CCS_PROJECT_PATH/training'

# In $CCS_PROJECT_PATH/compilation.yaml:
# compile_output_path: '$CCS_PROJECT_PATH/compilation'
```

**Why absolute paths:** The script runs from `$TINYML_MODELZOO_PATH`, so relative paths like `./training` would create outputs in modelzoo, not in CCS project. Absolute paths ensure outputs always go to CCS project regardless of execution directory.

Confirm with user before assembling.

**CRITICAL: The final config.yaml is saved to `$CCS_PROJECT_PATH`, the imported CCS project root.**

Assemble (output to CCS project):
```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py generate_complete_config_file \
  "{\"task_name\": \"$TASK_NAME\", \"output_dir\": \"$CCS_PROJECT_PATH\", \"common_yaml_file\": \"$CCS_PROJECT_PATH/common.yaml\", \"dataset_yaml_file\": \"$CCS_PROJECT_PATH/dataset.yaml\", \"feature_extraction_yaml_file\": \"$CCS_PROJECT_PATH/feat_ext.yaml\", \"training_yaml_file\": \"$CCS_PROJECT_PATH/training.yaml\", \"testing_yaml_file\": \"$CCS_PROJECT_PATH/testing.yaml\", \"compilation_yaml_file\": \"$CCS_PROJECT_PATH/compilation.yaml\"}"
```

On success: config saved to `$CCS_PROJECT_PATH/config.yaml`.
Show user the full config: `cat $CCS_PROJECT_PATH/config.yaml`

If any `*_yaml_file` is missing from CCS_PROJECT_PATH: go back to the relevant step and re-generate it.

---

## Step 13: Run training & compilation

### MANDATORY CONFIG PATH VERIFICATION

**BLOCKING GATE: Config MUST be from `$CCS_PROJECT_PATH`, NOT modelzoo.**

Before proceeding, verify:
```bash
# This is the ONLY config that should exist and be used
ls -lh "$CCS_PROJECT_PATH/config.yaml"

# This is where training outputs MUST go
ls -lh "$CCS_PROJECT_PATH/training/" 2>/dev/null || echo "Will be created during training"
ls -lh "$CCS_PROJECT_PATH/compilation/" 2>/dev/null || echo "Will be created during training"
```

The config file to run is **ALWAYS** `$CCS_PROJECT_PATH/config.yaml`. Whether it came from:
- Step 3.5 (copied from modelzoo example), or
- Steps 4-12 (generated from scratch)

This config has the correct output paths (`$CCS_PROJECT_PATH/training/` and `$CCS_PROJECT_PATH/compilation/`) pointing to your CCS project. Run it ONLY from CCS project:
```bash
./run_tinyml_modelzoo.sh "$CCS_PROJECT_PATH/config.yaml"
```

**NEVER run:**
```bash
# WRONG — this saves outputs to modelzoo, not your project
./run_tinyml_modelzoo.sh "$TINYML_MODELZOO_PATH/examples/*/config.yaml"

# WRONG — this runs from wrong directory
cd "$TINYML_MODELZOO_PATH/examples/some_example" && ./run_tinyml_modelzoo.sh config.yaml
```

**Ask user to explicitly confirm:** "Ready to start training from `$CCS_PROJECT_PATH/config.yaml`?" User must say yes before proceeding.

**GPU check — ALWAYS run this before training, even when using an existing example:**
```python
import torch
print(torch.cuda.device_count())
```
If result > 0, **always inform the user** regardless of how they got here:
> "A GPU is available on this machine ([N] detected). The config's `training.num_gpus` is currently set to [current value or 'not set']. Setting it to [N] will significantly speed up training. Would you like to update it?"

If yes and the config file already exists (`$CCS_PROJECT_PATH/config.yaml`), update `num_gpus` in the training section before running. If no, proceed with current config.

**SHOW USER FULL CONFIG BEFORE ASKING USER TO ALLOW TRAINING**
Ask user: *"Ready to start training? This will take several minutes."*

Reference `references/example_running_guide.md` for monitoring tips.

**[MANDATORY — see CRITICAL RULE 1] Discover and export compiler path BEFORE running the script:**

**Use `ccs-project → getCompilers` if available (CCS knows all installed SDKs):**
```
Call ccs-project → getCompilers
```
From returned list, find entry matching `$TARGET_DEVICE` family:
- `TI C2000 CGT` → F28xxx devices
- `TI ARM CGT` or `tiarmclang` → MSPM0, AM26x, etc.

Extract installation path, strip to root containing `bin/`, store as `CGT_ROOT`.

**If `ccs-project` MCP unavailable, fall back to filesystem search:**
```bash
# C2000 CGT
find /opt/ti ~/ti $HOME/ti /usr/local/opt/ti 2>/dev/null -name "cl2000" -type f 2>/dev/null | head -1

# ARM CGT
find /opt/ti ~/ti $HOME/ti /usr/local/opt/ti 2>/dev/null -name "armcl" -type f 2>/dev/null | head -1
```
Strip binary → get `bin/` → strip `bin/` → get `CGT_ROOT`.

**If compiler not found by either method:** STOP. Ask user for CGT root path before continuing.

**[MANDATORY] Activate virtual environment BEFORE running the training script:**

The venv must be active so that compiler environment variables and Python paths are correctly set. Activate it explicitly:
```bash
source "$TINYML_MODELZOO_PATH/.venv/bin/activate"
```

**Stream logs live — compiler path, venv activation, and script run MUST be in ONE bash call:**

CRITICAL: Shell variables do NOT persist across separate bash tool calls. `CGT_ROOT` was discovered from the MCP or filesystem search above — it lives in your context, NOT as a shell variable. You MUST substitute the literal discovered path directly into the command. Never write `$CGT_ROOT` as a shell variable in the bash call — it will be empty and Python will silently fall back to the hardcoded default in constants.py.

Determine the correct env var name per `$TARGET_DEVICE`:
- C2000 devices (F28xxx) → `C2000_CGT_PATH`
- ARM devices (MSPM0, AM26x) → `ARM_LLVM_CGT_PATH`

Use inline env var prefix on the script invocation — `cd` first, then prefix the env var directly on `./run_tinyml_modelzoo.sh`. Substitute the LITERAL PATH from your context where `<ACTUAL_CGT_ROOT_PATH>` appears:

```bash
LOG_FILE=$(mktemp /tmp/tinyml_run_XXXXXX.log)
source "$TINYML_MODELZOO_PATH/.venv/bin/activate"
cd "$TINYML_MODELZOO_PATH" && \
  C2000_CGT_PATH="<ACTUAL_CGT_ROOT_PATH>" \
  ./run_tinyml_modelzoo.sh "$CCS_PROJECT_PATH/config.yaml" > "$LOG_FILE" 2>&1 &
RUN_PID=$!
tail -f "$LOG_FILE" &
TAIL_PID=$!
wait $RUN_PID
TRAIN_EXIT=$?
kill $TAIL_PID 2>/dev/null
echo "Training exit code: $TRAIN_EXIT"
```

Example — if `getCompilers` returned `/opt/ti/ccs/tools/compiler/ti-cgt-c2000_21.6.0.LTS` for a C2000 device, you write:
```bash
cd "$TINYML_MODELZOO_PATH" && \
  C2000_CGT_PATH="/opt/ti/ccs/tools/compiler/ti-cgt-c2000_21.6.0.LTS" \
  ./run_tinyml_modelzoo.sh "$USER_WORKING_DIR/config.yaml" > "$LOG_FILE" 2>&1 &
```

**Note:** The script runs from `$TINYML_MODELZOO_PATH`, but passes `$USER_WORKING_DIR/config.yaml` (user's working directory). Output artifacts are placed in `$CCS_PROJECT_PATH/training/` and `$CCS_PROJECT_PATH/compilation/` per the config's relative paths.

User sees live output via `tail -f`. On completion, check `TRAIN_EXIT` — non-zero means failure.

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
1. Feature Extraction preset failed due to size issues. Dataset's class samples may have too few entries for presets. If no presets can fit it, then think of using raw feature extraction transforms. Consult `$(python3 -c "import tinyml_modelmaker, os; print(os.path.join(os.path.dirname(tinyml_modelmaker.__file__), 'ai_modules', 'timeseries', 'constants.py'))")` for all available transforms and presets. See if any of them or any combination of them could be of use for your usecase and accordingly select them.

2. Due to the above issue, many times files having less number of samples (less than what the presets may expect) may be skipped. As a result the metrics you get may be skewed and show very high values. Do not get confused and report this to the user as the final metrics. Stop the training, go back to the config, and refer point 1 to fix the issue. Once fixed, then train again.

---

## Step 13A: Verify compiled artifacts and display memory footprint (BLOCKING GATE — see CRITICAL RULE 2)

**Do NOT proceed to Step 14 until both checks below pass.**

**Check 1 — compiled artifacts exist in CCS project:**
```bash
ls -la "$CCS_PROJECT_PATH/compilation/artifacts/"
```
`mod.a` must be present and non-empty. If missing or directory is empty: compilation failed — do not proceed. Go back to Step 13 error handling.

**Check 2 — extract and display memory footprint:**

Find and display FLASH (RO Mem) and SRAM (RW Mem) from compilation logs:

```bash
# Extract from compilation log in CCS project
grep -E "FLASH|SRAM|RO Mem|RW Mem" "$CCS_PROJECT_PATH/compilation/run.log"
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

Get device memory specs from the SDK documentation for your device family:
- C2000 devices: check device datasheet (typically 256KB–1MB FLASH, 32KB–128KB SRAM)
- MSPM0 devices: check MSPM0 SDK documentation
- AM26x: check MCU+ SDK docs

**If model doesn't fit:**
- Stop. Do NOT proceed to deployment.
- Reduce model size: enable quantization mode 2, use "Memory" NAS optimization, or select smaller model
- Re-run training (Step 13)
- Check memory again before deployment

---

## Step 14: Deploy to device (if user asks)

**MANDATORY BLOCKING GATE — DO NOT SKIP EITHER REQUIREMENT** ⚠️

**Before ANY deployment work, BOTH of these must be completed and verified:**

1. **Exclude build folders (Step 14A/14B)** — `compilation/`, `training/`, `config.yaml` marked as excluded in `.cproject`
2. **Copy trained artifacts (Step 14C)** — `mod.a`, headers, golden vectors copied to correct project locations

**If EITHER is skipped, deployment will fail with linker errors or corrupted binaries.**

---

**Read full guide first:** `references/device_deployment_guide.md`

Compilation artifacts are now in `$CCS_PROJECT_PATH/compilation/artifacts/` (already in the CCS project from training). **Verify that training and compilation artifacts are in `$CCS_PROJECT_PATH/training/`and `$CCS_PROJECT_PATH/compilation/artifacts/` respectively. Once verified, immediately exclude them from build by doing the following:**

### Step 14A: Verify artifacts exist (BLOCKING GATE)

Check that compiled artifacts are present:
```bash
ls -la "$CCS_PROJECT_PATH/compilation/artifacts/mod.a"
```
If not found, training/compilation incomplete. Do not proceed to Step 14B.

---

### Step 14B: Mark `$CCS_PROJECT_PATH/compilation/`, `$CCS_PROJECT_PATH/training/`, and `$CCS_PROJECT_PATH/config.yaml` folders as excluded — MANDATORY

**ABSOLUTE REQUIREMENT BEFORE BUILD:** The `compilation/`, `training/`, and `config.yaml` MUST be excluded from CCS build. They are temporary work artifacts containing training outputs and configuration — NOT source code. Including them in the build will cause linker errors or incorrect binaries.

**This step is NOT optional and must complete successfully before proceeding to Step 14C.**

**Step 1 — Update .cproject XML file:**

Read `.cproject` located at `$CCS_PROJECT_PATH/.cproject`:
```bash
cat "$CCS_PROJECT_PATH/.cproject"
```

Update the `<sourceEntries>` section to exclude the folders and file using the pipe-separated format.

**EXACT placement in XML hierarchy:**
```xml
<configuration id="...">
  <folderInfo id="..." name="/">
    <!-- folder contents here -->
  </folderInfo>
  <sourceEntries>
    <entry excluding="compilation|training|config.yaml" flags="VALUE_WORKSPACE_PATH|RESOLVED" kind="sourcePath" name=""/>
  </sourceEntries>
</configuration>
```

`<sourceEntries>` is a **sibling of `<folderInfo>`**, both under `<configuration>` — NOT nested inside `<folderInfo>`. Place it AFTER the closing `</folderInfo>` tag, BEFORE the closing `</configuration>` tag.

**Important:** 
- Use pipe (`|`) to separate multiple exclusions (no spaces around pipes)
- This approach excludes folders and files together in a single `excluding` attribute
- The `flags` and `kind` attributes must remain as shown
- If `<sourceEntries>` already exists, replace the `excluding` value with the pipe-separated list above
- If `<sourceEntries>` does not exist, add it as shown above (after `</folderInfo>`, before `</configuration>`) i.e after `<folderInfo></folderInfo>` is done and add entry in the manner described above

**Step 2 — User verification in CCS IDE (MANDATORY):**

Ask user to open CCS IDE and verify visually. **Three ways to check:**

**Check 1 — Visual appearance in Project Explorer:**
> "In CCS Project Explorer (left panel), look at the `compilation` and `training` folders. Excluded folders appear **greyed out** or with a **strikethrough** name. Are they greyed out?"

**Check 2 — Right-click context menu:**
> "Right-click on `compilation` folder. A menu appears. Look for 'Exclude from Build' option. If it has a **checkmark** (✓), the folder is excluded. Repeat for `training` folder. Are both checked?"

**Check 3 — Build configuration properties:**
> "Go to Project → Properties → C/C++ Build. In the 'Build Variables' or 'Excluded resources' tab, `compilation/` and `training/` should be listed. Do you see them there?"

**User should confirm ALL THREE are excluded:**
> "I've updated the project settings (.cproject file) to exclude `compilation/`, `training/`, and `config.yaml` from build. **Please verify right now in CCS IDE using the checks above.** If folders are NOT greyed out or NOT marked as excluded, manually right-click each and select 'Exclude from Build' to mark them.
>
> Tell me when you've confirmed:
> - [ ] `compilation/` is greyed out or has checkmark
> - [ ] `training/` is greyed out or has checkmark
> - [ ] `config.yaml` does not appear in build tree or is marked excluded"

**MANDATORY USER ACKNOWLEDGMENT — DO NOT PROCEED WITHOUT THIS**

Ask user to explicitly confirm:
> "I have verified in CCS IDE that:
> - [ ] `compilation/` folder is greyed out / excluded
> - [ ] `training/` folder is greyed out / excluded  
> - [ ] `config.yaml` is marked as excluded
> 
> I acknowledge that if any of these are NOT excluded, the build will fail with linker errors."

**User must type confirmation (e.g., "confirmed" or "yes, all excluded").** Do not proceed to Step 14C without explicit user response.

If exclusions are missing or incorrect: build will fail with linker errors (trying to compile `config.yaml`) or link duplicates (trying to link `mod.a` twice). Stop and ask user to manually exclude them in CCS, then re-verify before proceeding.

---

### Step 14C: Copy trained artifacts to CCS project structure

**CRITICAL CONCEPT:** Training outputs are saved to `$CCS_PROJECT_PATH/compilation/` and `$CCS_PROJECT_PATH/training/` (temporary work folders). **These MUST be copied to the correct artifact directories within the imported project structure so the build can find and link them.** The temporary folders are EXCLUDED from build.

The imported project contains placeholder artifacts from the SDK template in project-specific locations (varies per device family and project structure). Replace them with the trained model outputs.

**C1. Discover artifact locations FROM TEMPLATE:**

First, identify where the SDK template example (from Step 3B) stores its artifacts. Search the template directory:
```bash
# Find where template stores compiled artifacts (typically .a files)
find "$TEMPLATE_DIR" -name "*.a" 2>/dev/null | head -3

# Find where template stores headers
find "$TEMPLATE_DIR" -name "*.h" 2>/dev/null | head -3

# Find golden vectors (if this is a golden-vector example)
find "$TEMPLATE_DIR" -name "test_vector.c" 2>/dev/null
find "$TEMPLATE_DIR" -name "user_input_config.h" 2>/dev/null
```

From these results, identify:
- `$ARTIFACT_TARGET_DIR` — directory where `mod.a` should be placed (relative to project root)
- `$HEADER_TARGET_DIR` — directory where `.h` headers should be placed
- Whether golden vectors exist in the template (if yes, note their target directory as `$GOLDEN_VECTOR_TARGET_DIR`)

**C2. Copy compiled artifacts FROM training output TO project:**

Copy trained artifacts FROM `$CCS_PROJECT_PATH/compilation/artifacts/` (training OUTPUT folder) TO the locations discovered in C1:
```bash
cp "$CCS_PROJECT_PATH/compilation/artifacts/mod.a" "$ARTIFACT_TARGET_DIR/"
cp "$CCS_PROJECT_PATH/compilation/artifacts/tvmgen_default.h" "$HEADER_TARGET_DIR/"
```

**Verify copy succeeded:**
```bash
ls -la "$ARTIFACT_TARGET_DIR/mod.a"
ls -la "$HEADER_TARGET_DIR/tvmgen_default.h"
```

**C3. Copy golden vectors (if template uses them):**

If golden vectors were found in template in C1, copy trained golden vectors FROM `$CCS_PROJECT_PATH/training/quantization/golden_vectors/` TO the same target directory:
```bash
cp "$CCS_PROJECT_PATH/training/quantization/golden_vectors/test_vector.c"    "$GOLDEN_VECTOR_TARGET_DIR/"
cp "$CCS_PROJECT_PATH/training/quantization/golden_vectors/user_input_config.h" "$GOLDEN_VECTOR_TARGET_DIR/"
```

**Verify copy succeeded:**
```bash
ls -la "$GOLDEN_VECTOR_TARGET_DIR/test_vector.c"
ls -la "$GOLDEN_VECTOR_TARGET_DIR/user_input_config.h"
```

**C4. Verify timestamps match:**

Confirm the artifacts were copied correctly:
```bash
stat "$CCS_PROJECT_PATH/compilation/artifacts/mod.a"
stat "$ARTIFACT_TARGET_DIR/mod.a"
```

If mtimes differ by more than 0.5 seconds, the wrong artifact was copied — re-copy and recheck.

**C5. Comprehensive artifact verification — BLOCKING GATE before proceeding to Step 14D**

This checkpoint catches missing or misplaced artifacts BEFORE build, preventing cryptic linker errors. Do NOT proceed to Step 14D until all checks pass.

**Verification Checklist:**

1. **Compiled model artifact (mod.a):**
   ```bash
   ls -lh "$ARTIFACT_TARGET_DIR/mod.a"
   file "$ARTIFACT_TARGET_DIR/mod.a"
   ```
   ✓ File exists  
   ✓ Size > 1KB (empty files indicate copy failed)  
   ✓ File type is ELF object  

2. **TVM header (tvmgen_default.h):**
   ```bash
   ls -lh "$HEADER_TARGET_DIR/tvmgen_default.h"
   grep -c "tvmgen" "$HEADER_TARGET_DIR/tvmgen_default.h"
   ```
   ✓ File exists  
   ✓ Contains expected TVM symbols  

3. **Golden vectors (if template uses them):**
   ```bash
   if [ -n "$GOLDEN_VECTOR_TARGET_DIR" ]; then
     ls -lh "$GOLDEN_VECTOR_TARGET_DIR/test_vector.c"
     ls -lh "$GOLDEN_VECTOR_TARGET_DIR/user_input_config.h"
   fi
   ```
   ✓ Both files exist (if applicable)  
   ✓ Both files non-empty  

4. **Artifact integrity (timestamps):**
   ```bash
   stat "$CCS_PROJECT_PATH/compilation/artifacts/mod.a" | grep Modify
   stat "$ARTIFACT_TARGET_DIR/mod.a" | grep Modify
   ```
   ✓ Source and destination mtimes match (within 1 second)  
   ✓ Indicates successful copy  

5. **Build-excluded folders are actually excluded:**
   ```bash
   grep -l "compilation\|training\|config.yaml" "$CCS_PROJECT_PATH/.cproject" | head -1
   ```
   ✓ `.cproject` contains exclusion rules  

**If ANY check fails:**

| Failure | Action |
|---------|--------|
| `mod.a` missing or < 1KB | Go back to C2. Re-copy from `$CCS_PROJECT_PATH/compilation/artifacts/mod.a`. Verify source exists: `ls "$CCS_PROJECT_PATH/compilation/artifacts/"` |
| `tvmgen_default.h` missing | Go back to C2. Re-copy. Check source path: `tail -20 "$CCS_PROJECT_PATH/compilation/run.log"` |
| Golden vectors missing (but template expects them) | Go back to C3. Verify training generated them: `ls "$CCS_PROJECT_PATH/training/quantization/golden_vectors/"`. If not present, training may have failed. |
| `mtimes` differ > 1 second | Copy may have been interrupted. Re-run copy commands in C2/C3. |
| Exclusion rules missing from `.cproject` | Go back to Step 14A. Re-add `<sourceEntries>` section. |

**PROCEED TO STEP 14D ONLY when all checks above pass.**

### Step 14D: Build project

> **Method priority: MCP → headless bash → manual. Never skip a tier without a concrete reason.**

**PRE-FLIGHT CHECKS — MANDATORY before attempting build**

**Check 1: Exclusions in place:**
```bash
grep -q "compilation\|training\|config.yaml" "$CCS_PROJECT_PATH/.cproject" && echo "✓ Exclusions found" || (echo "✗ EXCLUSIONS MISSING — STOP"; exit 1)
```

**Check 2: Artifacts copied:**
```bash
[ -f "$ARTIFACT_TARGET_DIR/mod.a" ] && [ -s "$ARTIFACT_TARGET_DIR/mod.a" ] && echo "✓ mod.a present" || (echo "✗ mod.a MISSING or EMPTY — STOP"; exit 1)
[ -f "$HEADER_TARGET_DIR/tvmgen_default.h" ] && echo "✓ Headers present" || (echo "✗ Headers MISSING — STOP"; exit 1)
```

**If any check fails: STOP. Do NOT attempt build.** Go back to Step 14A/14B/14C and fix before retrying.

---

**Try 1 — CCS MCP (`CCS_PROJECT_MCP == true` only)**

If `CCS_PROJECT_MCP` is `false` (checked in Session Setup step 5), skip directly to Try 2.

The MCP gives richer error output, runs inside the already-open CCS instance, and respects any active SDK / path configuration the user has set.

Requirements: `CCS_PROJECT_MCP == true` AND B2 (IDE registration) succeeded.

**Pre-build: terminate any active debug sessions for this project**

An active debug session holds a lock on the project. `buildProject` will hang indefinitely waiting for that lock to release. Always clear sessions before building:

```
1. Call ccs-debug → getDebugSessions
2. For each session whose name contains $PROJECT_NAME:
   Call ccs-debug → terminate
     sessionId: <session id>
3. Wait for terminations to complete before proceeding.
```

If `ccs-debug` MCP is not available, skip this pre-check — but if build hangs, ask the user to close any active debug sessions in CCS manually, then retry.

```
Call ccs-project → buildProject
  projectName: $PROJECT_NAME
  outputMode:  "auto"
```

If build succeeds → proceed to Step 14E.

If it fails, identify **why** before moving on:
- `"project not found"` / `"not imported"` → B2 IDE registration did not complete; the project was never registered in CCS.
- `"MCP tool not available"` / tool missing from session → `ccs-project` MCP is not configured in this Claude session.
- `"CCStudio not running"` / connection refused → CCStudio IDE is closed; MCP has nothing to connect to.
- Compiler or linker error → the project *was* built but failed; **do not fall back** — fix the error instead (wrong artifacts, missing header, etc.).

Only fall through to Try 2 when the MCP **could not be used at all** (first three bullets above). A build error that the MCP successfully reported is not a reason to try bash.

**Try 2 — Headless bash (only if MCP was genuinely impossible)**

Before running this, state explicitly why Try 1 could not be used (one of the three reasons above). Then **STOP. Tell the user:**
  > "CCS MCP is not available ([exact reason]). The fallback uses headless build commands which may be less reliable and may miss configurations that CCStudio MCPs handle automatically. It is highly recommended to proceed with the MCP. Do you still want to proceed with the headless commands?"
Only proceed if the user explicitly confirms

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py build_ccs_project \
  "{\"ccs_project_path\": \"$CCS_PROJECT_PATH\", \"ccs_install_path\": \"$CCS_INSTALL_PATH\"}"
```

If this also fails, capture the error message from `errors` in the response.

**Try 3 — Ask user to build manually (absolute last resort)**

Only reach here if both Try 1 AND Try 2 failed. Tell the user exactly why:

> "Automated build could not complete:
> - **MCP (Try 1):** [exact reason — e.g., 'ccs-project MCP not configured in this session']
> - **Headless bash (Try 2):** [exact error — e.g., 'CCS launcher not found at /opt/ti/ccs1260/eclipse/ccstudio']
>
> Please build manually: open CCS → **Project → Build Project** (Ctrl+B)."

---

### Step 14E: Flash to device

Connect device via USB/JTAG first.

> **Method priority: MCP → headless bash/dslite → manual. Never skip a tier without a concrete reason.**

**Try 1 — CCS Debug MCP (`CCS_DEBUG_MCP == true` only)**

If `CCS_DEBUG_MCP` is `false` (checked in Session Setup step 5), skip directly to Try 2.

The MCP flashes through the already-open CCS debug infrastructure, giving live feedback and avoiding dslite configuration issues.

Requirements: `CCS_DEBUG_MCP == true` AND B2 (IDE registration) succeeded AND Step 14C build succeeded.

> Note: `debugProject` typically takes 30–90 seconds on real hardware (JTAG probe negotiation, flash erase/write, target reset). This is normal — do not treat it as a hang unless it exceeds 3 minutes.

```
1. Call ccs-debug → getDebugSessions
   — if a session for $PROJECT_NAME already exists, terminate it first

2. Call ccs-debug → debugProject
   projectName: $PROJECT_NAME
   This starts a debug session directly by project name — no config name resolution needed.
   CCS automatically connects to the target and loads the compiled binary.
   Expected duration: 30–90 seconds. Wait patiently.

3. If program not loaded automatically:
   Call ccs-debug → loadProgram
   programUri: "$CCS_PROJECT_PATH/Debug/$PROJECT_NAME.out"

4. Call ccs-debug → continue   (starts execution)
```

If flash succeeds → go to the post-flash verification below.

If it fails, identify **why** before moving on:
- `"MCP tool not available"` / tool missing → `ccs-debug` MCP is not configured in this session.
- `"CCS not running"` / connection refused → CCS application is closed (pre-check should have caught this).
- `"project not found"` / unknown project → B2 IDE registration did not complete; re-run B2 then retry.
- `"device not found"` / `"no target connected"` → physical connection issue; **do not fall back to dslite** — it will hit the same problem. Ask user to check USB/JTAG and rerun from Try 1.
- Any other runtime error → report it; only proceed to Try 2 if the MCP connection itself was the blocker, not the device.

**Try 2 — Headless bash / dslite (only if MCP was genuinely impossible)**

Before running this, state explicitly why Try 1 could not be used.

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py flash_ccs_project \
  "{\"ccs_project_path\": \"$CCS_PROJECT_PATH\", \"ccs_install_path\": \"$CCS_INSTALL_PATH\"}"
```

If this also fails, capture the error from `errors` in the response.

**Try 3 — Ask user to flash manually (absolute last resort)**

Only reach here if both Try 1 AND Try 2 failed. Tell the user exactly why:

> "Automated flash could not complete:
> - **MCP (Try 1):** [exact reason — e.g., 'ccs-debug MCP not configured in this session']
> - **Headless dslite (Try 2):** [exact error — e.g., 'dslite not found at expected path; verify CCS_INSTALL_PATH']
>
> Please flash manually: open CCS → **Run → Flash Project**."

---

**After flashing:** In CCS Debug perspective, set a breakpoint after the inference call, then check `test_result == 1` in the Watch window (1 = pass, 0 = fail).

See `references/device_deployment_guide.md` for troubleshooting, device/SDK mappings, and full walkthrough.
