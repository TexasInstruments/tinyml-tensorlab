---
name: setup
description: One-time setup for the TinyML Agent Skill. Run this immediately after installing the plugin. Configures update mode (pinned vs auto-update), discovers SCRIPTS_DIR, verifies tinyml-modelzoo installation, sets up virtual environment, and saves all variables to .env. Must complete before using tinyml-workflow-agent. Always trigger when: user just installed the tinyml plugin, says "set up tinyml", "configure the tinyml skill", "first time setup", or encounters the message "Run /tinyml-agent-skills:setup first".
---

# TinyML Agent Skill — Setup

Run this once after installing the plugin. Re-run any time you move or reinstall tinyml-modelzoo, or want to change your update mode.

---

## Step 1: Discover script paths

Two separate `runner.py` files exist — one for this setup skill, one for the main tinyml-workflow-agent skill. Find both:

**Main skill scripts (SCRIPTS_DIR)** — used for tinyml-modelzoo operations during this session only (not stored in `.env`):
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

## Step 3: Confirm tinyml-modelzoo path

Ask user: *"What is the full path to your tinyml-modelzoo directory?"*
(e.g. `/home/username/tinyml-modelzoo`)

```bash
TINYML_MODELZOO_PATH=<user-provided path>

python3 "$SETUP_SCRIPTS_DIR/runner.py" check_installation \
  "{\"tinyml_modelzoo_path\": \"$TINYML_MODELZOO_PATH\"}"
```

If `success: false`: show `errors` and `hint`, ask user to correct path. Repeat until `success: true`.

---

## Step 4: Set up virtual environment and install packages

### 4A: Detect and activate environment

**Local venv always takes priority.** Use direct path checks — do NOT rely on `find` alone, it can silently return empty and cause incorrect fallthrough.

```bash
# Step 1: Check standard .venv location directly (fastest, most reliable)
if [ -f "$TINYML_MODELZOO_PATH/.venv/bin/python3" ]; then
  VENV_ROOT="$TINYML_MODELZOO_PATH/.venv"
  echo "Found local venv at: $VENV_ROOT"
  VENV_PYTHON="$VENV_ROOT/bin/python3"
else
  # Step 2: Search for any other venv inside modelzoo (non-standard name/location)
  VENV_ROOT=$(find "$TINYML_MODELZOO_PATH" -maxdepth 4 -type f -name "activate" -path "*/bin/activate" 2>/dev/null | head -1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  if [ -n "$VENV_ROOT" ] && [ -f "$VENV_ROOT/bin/python3" ]; then
    echo "Found local venv at: $VENV_ROOT"
    VENV_PYTHON="$VENV_ROOT/bin/python3"
  else
    VENV_ROOT=""
  fi
fi
```

**If no local venv found (`VENV_ROOT` is empty), STOP and ask the user:**

> "No local virtual environment was found inside your tinyml-modelzoo directory. How would you like to proceed?
>
> 1. **Use system/global Python** — use the system Python as-is. Setup will verify packages are present before continuing.
> 2. **Create a new local venv** — a fresh `.venv` will be created inside your modelzoo directory and packages will be installed into it."

- **User chooses option 1 (global Python):**
  ```bash
  VENV_PYTHON="$(which python3)"
  echo "Using system Python: $VENV_PYTHON"
  ```

- **User chooses option 2 (create new venv):**
  ```bash
  python3 -m venv "$TINYML_MODELZOO_PATH/.venv"
  VENV_PYTHON="$TINYML_MODELZOO_PATH/.venv/bin/python3"
  echo "Created new venv: $TINYML_MODELZOO_PATH/.venv"
  ```

```bash
# GUARDRAIL: Lock VENV_PYTHON. Once chosen, it cannot change.
readonly VENV_PYTHON
```

**Use `$VENV_PYTHON` for all subsequent python calls — venv activation does not persist across separate shell calls.**

### 4B: Verify and install packages (VENV_PYTHON is locked)

Use `$VENV_PYTHON` (set in 4A, now locked) — **NEVER test or switch to other Python instances**. If packages are missing, fail setup instead of falling back.

**Step 1: Check if core packages are installed**
```bash
if $VENV_PYTHON -c "import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo" 2>/dev/null; then
  PACKAGES_INSTALLED="yes"
  $VENV_PYTHON -c "
import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo
print('ModelMaker:', tinyml_modelmaker.__version__)
print('Tinyverse:', tinyml_tinyverse.__version__)
print('ModelOpt:', tinyml_torchmodelopt.__version__)
print('ModelZoo:', tinyml_modelzoo.__version__)
"
else
  PACKAGES_INSTALLED="no"
  # GUARDRAIL: If packages missing in chosen VENV_PYTHON, fail setup.
  # Do NOT fall back to global python — priority order is final.
  echo "ERROR: Required packages not found in $VENV_PYTHON"
  echo "Setup cannot proceed. The chosen Python environment does not have TinyML packages installed."
  exit 1
fi
```

**Step 2: Check if AutoNAS is installed**
```bash
if $VENV_PYTHON -c "import autonas" 2>/dev/null; then
  AUTONAS_INSTALLED="yes"
else
  AUTONAS_INSTALLED="no"
fi
```

**Step 3: Handle fresh install vs existing setup**

If `PACKAGES_INSTALLED=no`:
```bash
cd "$TINYML_MODELZOO_PATH"
```

Inform user about AutoNAS MCP:

> "Installing the AutoNAS MCP. This extends TinyML with two powerful features:
>
> 1. **Custom Dataset Support** — use datasets in any format (not just standard TinyML modelzoo format)
> 2. **Neural Architecture Search (NAS)** — automatically create models optimized for your specific use case
>
> You can still use pre-built models from the TinyML modelzoo, but AutoNAS enables you to bring your own datasets and generate custom models tailored to your data.

```bash
$VENV_PYTHON -m pip install -e ".[autonas]"
```

**Step 4: If packages exist but AutoNAS missing, add it**

If `PACKAGES_INSTALLED=yes` AND `AUTONAS_INSTALLED=no`:

> "You have TinyML installed, but AutoNAS is not enabled. Installing AutoNas (same features as explained above).

```bash
cd "$TINYML_MODELZOO_PATH"
$VENV_PYTHON -m pip install -e ".[autonas]"
```

**Step 5: Verify all imports**
```bash
$VENV_PYTHON -c "
import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo
print('ModelMaker:', tinyml_modelmaker.__version__)
print('Tinyverse:', tinyml_tinyverse.__version__)
print('ModelOpt:', tinyml_torchmodelopt.__version__)
print('ModelZoo:', tinyml_modelzoo.__version__)
"
```

If fails, show full pip output and stop — do not proceed until all packages import cleanly.

**Step 6: Initialize AutoNAS workspace (if newly installed)**

```bash
autonas-init "$VENV_PYTHON"
```

### 4C: Discover package paths

After imports succeed, store the installed package root paths — saved to `.env` in Step 6 so the workflow skill can reference modelmaker source files directly.

**CRITICAL: use the correct Python** — bare `python3` may resolve to a system interpreter that finds dev-repo editable installs, giving wrong paths. These packages are dependencies installed via modelzoo, not dev repos.

Use `$VENV_PYTHON` (already set in 4A — venv binary if venv exists, else `which python3`):
```bash
TINYML_MODELMAKER_PATH=$($VENV_PYTHON -c "import tinyml_modelmaker, os; print(os.path.dirname(tinyml_modelmaker.__file__))")
TINYML_TINYVERSE_PATH=$($VENV_PYTHON -c "import tinyml_tinyverse, os; print(os.path.dirname(tinyml_tinyverse.__file__))")
TINYML_MODELOPT_PATH=$($VENV_PYTHON -c "import tinyml_torchmodelopt, os; print(os.path.dirname(tinyml_torchmodelopt.__file__))")
```

Confirm all three are non-empty. If a venv was detected and any path does NOT contain `.venv`, warn the user — a global or dev-repo editable install is shadowing the venv install.

---

## Step 5: Run verification training (optional)

**On re-runs:** if `~/.tinyml-agent-skills/.env` already exists with `IS_REPO_SETUP=1`, and the user is only updating update-mode or fixing paths (not re-verifying training), skip this step.

Ask the user:
> "Would you like to run a quick verification training to confirm your environment works end-to-end? This runs a small example and takes a few minutes. You can skip it and go straight to using the skill."

If user skips: proceed directly to Step 6.

If user confirms, tell them: *"Running verification to confirm training and compilation work correctly. This will take a few minutes."*

**First, find the compiler path** — the verification example targets a C2000 device, so look for the C2000 CGT:

Use `ccs-project → getCompilers` if available — find the `TI C2000 CGT` entry and extract its root (strip to the directory containing `bin/`). Store as `CGT_ROOT`.

If CCS MCP unavailable, search the filesystem:
```bash
find /opt/ti ~/ti $HOME/ti /usr/local/opt/ti 2>/dev/null -name "cl2000" -type f 2>/dev/null | head -1
```
Strip the binary name → `bin/` → strip `bin/` → `CGT_ROOT`.

**If compiler not found:** STOP. Ask the user:
> "Compiler not found. Please provide the full path to your C2000 CGT root (folder containing `bin/cl2000`)."
Do not run verification without the compiler path confirmed.

**Export compiler paths and run:**

CRITICAL: env vars and script invocation MUST be in a single shell command. Each Bash tool call is a separate subshell — `export` in one call does not survive to the next. Python's `constants.py` is imported at process start, so the vars must already be in the environment when the subprocess spawns.

Also CRITICAL: `CGT_ROOT` was discovered from the MCP or filesystem search above — it lives in your context, NOT as a shell variable. You MUST substitute the literal discovered path directly into the command. Do NOT write `$CGT_ROOT` as a shell variable — it will be empty in a new bash call.

The verification example targets a C2000 device — only `C2000_CGT_PATH` is needed. Substitute the LITERAL PATH where `<ACTUAL_CGT_ROOT_PATH>` appears:

Linux:
```bash
cd "$TINYML_MODELZOO_PATH" && \
  C2000_CGT_PATH="<ACTUAL_CGT_ROOT_PATH>" \
  ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
```

Windows:
```powershell
$env:C2000_CGT_PATH="<ACTUAL_CGT_ROOT_PATH>"; cd "$TINYML_MODELZOO_PATH"; run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
```

Stream output to user.

**If verification passes:** proceed to Step 6.

---

## Step 6: Write .env file

Config is written to `~/.tinyml-agent-skills/.env` — user-global, survives plugin updates, reinstalls, and cache clears. Works on all platforms (Linux, macOS, Windows).

**CRITICAL: Inform the user of the location:**
> "Configuration will be saved to: `~/.tinyml-agent-skills/.env`"
> "This persists across all sessions and plugin versions."

Call the setup runner to write the file (cross-platform — no bash file operations):
```bash
python3 "$SETUP_SCRIPTS_DIR/runner.py" save_config \
  "{\"tinyml_modelzoo_path\": \"$TINYML_MODELZOO_PATH\", \"update_mode\": \"$UPDATE_MODE\", \"pinned_version\": \"$UPDATE_PINNED_VERSION\", \"tinyml_modelmaker_path\": \"$TINYML_MODELMAKER_PATH\", \"tinyml_tinyverse_path\": \"$TINYML_TINYVERSE_PATH\", \"tinyml_modelopt_path\": \"$TINYML_MODELOPT_PATH\"}"
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
> "If you move tinyml-modelzoo or change your setup, re-run this setup skill."

Re-run this setup skill any time you:
- Move or reinstall tinyml-modelzoo
- Switch to a different tinyml-modelzoo version
- Want to change pinned vs auto-update mode
- Encounter: "Run `/tinyml-agent-skills:setup` first" (means `~/.tinyml-agent-skills/.env` is missing or incomplete)
