# Session Setup Details

`SKILL.md` carries the 4-item checklist. This file has the commands, variable tables and branch handling behind each item. Run all four items every session, including when jumping to a mid-workflow step.

## Contents
1. Discover SCRIPTS_DIR
2. Load environment
3. Verify venv and set the training command
4. Detect the IDE and its tooling
5. Session state variables

## 1. Discover SCRIPTS_DIR

`SCRIPTS_DIR` = `<base directory of this skill>/scripts`. The base directory is given when the skill loads (`Base directory for this skill: ...`). Use that path literally; do not search for it. It is correct for user and project installs and for any plugin version. If the base directory is not in your context, ask the user where the plugin is installed.

Verify: `ls "$SCRIPTS_DIR/runner.py"`.

## 2. Load environment

Read `~/.tinyml-agent-skills/.env` (same path on Linux, macOS, Windows) and export:

| Variable | Description |
|---|---|
| `IS_REPO_SETUP` | tinyml-modelzoo is set up (`true` or `1` required) |
| `TINYML_MODELZOO_PATH` | repo install: modelzoo clone root. wheel install: the venv's `site-packages` (contains `tinyml_modelzoo/`) |
| `MODELZOO_INSTALL_MODE` | `repo` or `wheel` (missing in older `.env` → treat as `repo`) |
| `VENV_PYTHON` | python binary of the environment setup chose (missing in older `.env` → `$TINYML_MODELZOO_PATH/.venv/bin/python3`) |
| `TINYML_MODELMAKER_PATH` | installed tinyml_modelmaker package root |
| `TINYML_TINYVERSE_PATH` | installed tinyml_tinyverse package root |
| `TINYML_MODELOPT_PATH` | installed tinyml_torchmodelopt package root |

Tell the user: ".env loaded from `~/.tinyml-agent-skills/.env`".

If `.env` is missing or any variable is unset, stop: "Run `/tinyml-agent-skills:setup` first." Do not re-run setup yourself.

## 3. Verify venv and set the training command

```bash
[ -x "$VENV_PYTHON" ] && echo "venv found" || echo "no venv"
```

If there is no venv, setup was not completed or the venv was deleted. Stop and send the user to `/tinyml-agent-skills:setup`.

`VENV_PYTHON` must be used for every python call this session. Venv activation does not persist across tool calls. `VENV_ROOT` is `$(dirname "$(dirname "$VENV_PYTHON")")`; activate with `source "$VENV_ROOT/bin/activate"` when a step says so.

| `MODELZOO_INSTALL_MODE` | `TRAIN_DIR` | `TRAIN_CMD` |
|---|---|---|
| `repo` | `$TINYML_MODELZOO_PATH` | `./run_tinyml_modelzoo.sh` |
| `wheel` | `$HOME/.tinyml-agent-skills` | `$VENV_PYTHON -m tinyml_modelmaker.run_tinyml_modelmaker` |

Wheel installs ship no `run_tinyml_modelzoo.sh`, so the skill runs the module directly. Resolve `TRAIN_CMD` to a literal string when you set it (for wheel: the venv's full python path, then `-m tinyml_modelmaker.run_tinyml_modelmaker`), so later steps never write an unresolved `$VENV_PYTHON`. Always invoke it as a module (`-m`), never as `python .../run_tinyml_modelmaker.py` from inside the package directory (the file does a `chdir('..')` that lands in `site-packages`). Extra flags (`--run_name`, `--num_gpus`, `--model_name`, `--input_data_path`, `--target_device`, `--batch_size`) can follow the config path. Windows: `& $env:VENV_PYTHON -m tinyml_modelmaker.run_tinyml_modelmaker <config>`. The config is always `$PROJECT_PATH/config.yaml` with absolute output paths, so results land in the project in both modes; `TRAIN_DIR` is only a neutral working directory.

## 4. Detect the IDE and its tooling

Work out which IDE the user is in (CCStudio or VS Code) and which tools you have for the project operations this skill needs: **import** a `.projectspec`, **build**, **debug/flash**, and optionally list compilers/SDKs. CCStudio typically exposes these as MCP tools; VS Code has CLI project/build/debug tools; TI command-line utilities are a further option. Inspect the tools actually available in the session; do not assume a particular set, and do not persist the result.

Tell the user in one line what will be automated and what, if anything, they will have to do by hand (for example "build and flash will be driven from here; import needs you to open the IDE"). A missing tool never stops training or compilation; it only affects Step 3 import and Steps 14D/14E, where the fallback is to ask the user to perform that step in their IDE and report the result.

## 5. Session state

`.env` variables are preloaded. The rest are set progressively and are never written to `.env`. Config files go in `$PROJECT_PATH`, not in `tinyml-modelzoo/examples/` or the bundled `references/examples/`.

| Variable | Set in | Description |
|---|---|---|
| `SCRIPTS_DIR`, `VENV_PYTHON`, `TRAIN_DIR`, `TRAIN_CMD` | Session Setup | see above |
| `TASK_TYPE`, `TARGET_DEVICE` | Step 2 | e.g. `motor_fault`, `F28P55` |
| `SDK_ROOT`, `AI_EXAMPLES_PATH`, `TEMPLATE_DIR` | Step 3 | SDK and template locations |
| `PROJECT_NAME`, `PROJECT_PATH` | Step 3 | imported project; all config, training and compilation outputs live here |
| `byod` | Step 4 (persisted in `$PROJECT_PATH/.tinyml_template.json`) | true when the user brought their own dataset and built the config in Steps 4–12; Step 13B only runs then |
| `npu_template` | Step 3 (persisted in `$PROJECT_PATH/.tinyml_template.json`) | whether the imported project is the NPU variant; re-read the file instead of relying on memory |
| `TASK_NAME`, `VARIABLES`, `DATA_PATH` | Step 4 | name, channel count, raw dataset path |
| `TARGET_MODULE` | Step 5 | `timeseries` or `vision` |
| `EFFECTIVE_DATA_PATH` | Step 6 | dataset path after any reorganization; use it instead of `DATA_PATH` from then on |
| `MODEL_NAME` | Step 8C | chosen model |
| `QUANTIZATION_MODE` | Step 9 Part C | 0/1/2, drives the compilation preset in Step 11 |
| `RUN_ID`, `MODEL_ID`, `QUANTIZATION` | Step 13 | read from the training logs |
