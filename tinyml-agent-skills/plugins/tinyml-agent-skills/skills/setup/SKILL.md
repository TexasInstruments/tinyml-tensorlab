---
name: setup
description: One-time setup for the TinyML Agent Skill. Run this immediately after installing the plugin. Configures update mode (pinned vs auto-update), discovers SCRIPTS_DIR, verifies tinyml-tensorlab installation, sets up virtual environment, and saves all variables to .env. Must complete before using tinyml-workflow-agent. Always trigger when: user just installed the tinyml plugin, says "set up tinyml", "configure the tinyml skill", "first time setup", or encounters the message "Run /tinyml-agent-skills:setup first".
---

# TinyML Agent Skill — Setup

Run this once after installing the plugin. Re-run any time you move or reinstall tinyml-tensorlab, or want to change your update mode.

---

## Step 1: Discover script paths

Two separate `runner.py` files exist — one for this setup skill, one for the main tinyml-workflow-agent skill. Find both:

**Main skill scripts (SCRIPTS_DIR)** — used for tinyml-tensorlab operations during this session only (not stored in `.env`):
```bash
find ~/.claude -name "runner.py" 2>/dev/null | grep "tinyml-workflow-agent" | head -1
```
Set `SCRIPTS_DIR` from result (strip `/runner.py`, keep the directory).

**Setup skill scripts (SETUP_SCRIPTS_DIR)** — used only during this setup:
```bash
find ~/.claude -name "runner.py" 2>/dev/null | grep "setup/scripts" | head -1
```
Set `SETUP_SCRIPTS_DIR` from result (strip `/runner.py`, keep the directory).

If either is not found, ask the user:
> "Where is the tinyml-agent-skills plugin installed?"

Verify both runners exist:
```bash
ls "$SCRIPTS_DIR/runner.py"
ls "$SETUP_SCRIPTS_DIR/runner.py"
```

---

## Step 2: Choose update mode

Ask the user:

> "How would you like to manage updates for this skill?
>
> 1. **Pinned** — stay on the current version, no automatic updates
> 2. **Auto-update** — check for newer versions at the start of each session"

**NOTE**: Current version can be found in `plugins/tinyml-agent-skills/.claude-plugin/plugin.json`,

Call the setup runner (not the main skill runner) with their choice:
```bash
# Pinned:
UPDATE_RESPONSE=$(python3 "$SETUP_SCRIPTS_DIR/runner.py" set_update_mode '{"mode": "pinned"}')

# Auto-update:
UPDATE_RESPONSE=$(python3 "$SETUP_SCRIPTS_DIR/runner.py" set_update_mode '{"mode": "auto"}')
```

Confirm `success: true` from `UPDATE_RESPONSE` before proceeding.

Extract and store for Step 7:
```bash
UPDATE_MODE=$(echo "$UPDATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('mode'))")
UPDATE_PINNED_VERSION=$(echo "$UPDATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('pinned_version') or '')")
```

---

## Step 3: Confirm tinyml-tensorlab path

Ask user: *"What is the full path to your tinyml-tensorlab directory?"*
(e.g. `/home/username/tinyml-tensorlab`)

```bash
TINYML_BASE_PATH=<user-provided path>

python3 "$SETUP_SCRIPTS_DIR/runner.py" check_installation \
  "{\"tinyml_base_path\": \"$TINYML_BASE_PATH\"}"
```

If `success: false`: show `errors` and `hint`, ask user to correct path. Repeat until `success: true`.

---

## Step 4: Verify packages and virtual environment

First check if a venv exists:
```bash
ls "$TINYML_BASE_PATH/tinyml-modelmaker/venv/bin/activate" 2>/dev/null && echo "venv found" || echo "no venv"
```

- **venv found** — activate it: `source "$TINYML_BASE_PATH/tinyml-modelmaker/venv/bin/activate"`
- **no venv** — the user may have installed packages globally or via pyenv/conda. Do NOT force a venv. Proceed with the current Python environment.

Either way, verify all packages import correctly:
```python
import tinyml_modelmaker
import tinyml_tinyverse
import tinyml_torchmodelopt
import tinyml_modelzoo

print(f"TI Tiny ML ModelMaker: {tinyml_modelmaker.__version__}")
print(f"TI Tiny ML Tinyverse: {tinyml_tinyverse.__version__}")
print(f"TI Tiny ML Model Optimization toolkit: {tinyml_torchmodelopt.__version__}")
print(f"TI Tiny ML Model Zoo: {tinyml_modelzoo.__version__}")
```

**If imports succeed:** skip to Step 5.

**If imports fail:** ask the user how they installed the packages — venv, pyenv, conda, or global pip. If user has not installed the requisite packages, follow `references/setup_guide.md` to set up the environment. Return here once done.

---

## Step 5: Run verification training (first-time only)

**On re-runs:** if `~/.tinyml-agent-skills/.env` already exists with `IS_REPO_SETUP=1`, and the user is only updating update-mode or fixing paths (not re-verifying training), skip this step.

Tell the user: *"Running a one-time verification to confirm training and compilation work correctly. This will take a few minutes."*

Linux:
```bash
cd "$TINYML_BASE_PATH/tinyml-modelzoo"
./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
```

Windows:
```powershell
cd "$TINYML_BASE_PATH\tinyml-modelzoo"
run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
```

Stream output to user.

**If compilation fails due to missing compiler path:** search for the compiler automatically, set the required environment variables, and re-run. If still not found, refer user to:
`https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/user_guide/installation/environment_variables.html#`

**If verification passes:** proceed to Step 6.

---

## Step 6: Set docs path

```bash
TINYML_TENSORLAB_DOCS_PATH="$TINYML_BASE_PATH/docs/source/"
```

Validate it contains RST subdirectories:
```bash
ls "$TINYML_TENSORLAB_DOCS_PATH/getting_started/" 2>/dev/null | head -3
```

If no output or error: path is wrong. Correct value is `$TINYML_BASE_PATH/docs/source/`. Fix before continuing.

---

## Step 7: Write .env file

Config is written to `~/.tinyml-agent-skills/.env` — user-global, survives plugin updates, reinstalls, and cache clears. Works on all platforms (Linux, macOS, Windows).

**CRITICAL: Inform the user of the location:**
> "Configuration will be saved to: `~/.tinyml-agent-skills/.env`"
> "This persists across all sessions and plugin versions."

Call the setup runner to write the file (cross-platform — no bash file operations):
```bash
python3 "$SETUP_SCRIPTS_DIR/runner.py" save_config \
  "{\"tinyml_base_path\": \"$TINYML_BASE_PATH\", \"docs_path\": \"$TINYML_TENSORLAB_DOCS_PATH\", \"update_mode\": \"$UPDATE_MODE\", \"pinned_version\": \"$UPDATE_PINNED_VERSION\"}"
```

Check `success: true`. On success, `env_file` in the response will show the exact path created.

Note: `SCRIPTS_DIR` is NOT stored in `.env` — it is derived automatically from the installed plugin location at the start of each session.

If `success: false`: show `errors`, do not proceed until resolved.

---

## Setup Complete

Tell the user:

> "✓ Setup complete. You can now use `/tinyml-agent-skills:tinyml-workflow-agent` to start building TinyML models."
>
> "Configuration saved to: `~/.tinyml-agent-skills/.env`"
> "This file is loaded automatically by tinyml-workflow-agent on every session start and survives plugin updates."
> "If you move tinyml-tensorlab or change your setup, re-run this setup skill."

Re-run this setup skill any time you:
- Move or reinstall tinyml-tensorlab
- Switch to a different tinyml-tensorlab version
- Want to change pinned vs auto-update mode
- Encounter: "Run `/tinyml-agent-skills:setup` first" (means `~/.tinyml-agent-skills/.env` is missing or incomplete)
