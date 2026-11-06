---
name: setup
description: One-time setup for the TinyML Agent Skill. Run this immediately after installing the plugin. Discovers SCRIPTS_DIR, detects tinyml-modelzoo (repo or pip wheel, offering to install the wheel if missing), sets up virtual environment, and saves all variables to .env. Must complete before using ti-ccstudio-tinyml-workflow. Always trigger when: user just installed the tinyml plugin, says "set up tinyml", "configure the tinyml skill", "first time setup", or encounters the message "Run /tinyml-agent-skills:setup first".
---

# TinyML Agent Skill — Setup

Run this once after installing the plugin. Re-run any time you move or reinstall tinyml-modelzoo.

---

## Install guardrails (apply to every install in Steps 2B and 3B)

Installs (venv creation, `pip install` of the wheel) can take several minutes. **Slowness is not failure.**

- **Never fall back to a different installation** (an existing repo, a previously detected wheel venv, global/system Python, or a different venv) because a new install is slow, times out, or appears hung. The user's choice in Step 3A is final.
- **Do not substitute or retry with other sources** (other URLs, versions, `--no-deps`, different Pythons) to "speed things up".
- **Use long timeouts and wait.** Run each pip command with the Bash `timeout` set to the maximum (600000 ms). If it may exceed that, run it with `run_in_background: true`, redirect output to a log (e.g. `"$WHEEL_VENV/pip-install.log"`), and poll the log/process until it exits. Tell the user it is still installing rather than giving up.
- **Only a non-zero exit code is failure.** A tool timeout, empty output, or a still-running process is not. Before treating a command as failed, check whether the process is still running (`pgrep -f "pip install"`) and wait for it.
- **On real failure: stop.** Show the full output/log, report that setup did not complete, and ask the user how to proceed. Do not continue setup against any other environment, and do not write `.env` for an install that did not finish and pass verification (3W / 3B Step 5).
- **Never present a fallback as success.** If the chosen environment fails verification, say so; do not report "setup complete" using a different environment than the one the user selected.

---

## Step 1: Discover script paths

Two separate `runner.py` files exist, one per skill. Both paths come from skill base directories, not from a search (the base directory is given when each skill loads, as `Base directory for this skill: ...`):

- **`SETUP_SCRIPTS_DIR`** = `<base directory of this setup skill>/scripts`, used only during this setup.
- **`SCRIPTS_DIR`** = `<base directory of ti-ccstudio-tinyml-workflow>/scripts`, used for tinyml-modelzoo operations during this session only (not stored in `.env`). Sibling skill: `../ti-ccstudio-tinyml-workflow/scripts` from this skill's base directory.

If a base directory is not in your context, ask the user:
> "Where is the tinyml-agent-skills plugin installed?"

Verify both runners exist:
```bash
ls "$SCRIPTS_DIR/runner.py"
ls "$SETUP_SCRIPTS_DIR/runner.py"
```

---

## Step 2: Locate tinyml-modelzoo

tinyml-modelzoo can be installed two ways: as a **git repo checkout** (`MODELZOO_INSTALL_MODE=repo`), or as a **pip wheel** inside a venv (`MODELZOO_INSTALL_MODE=wheel`). At the very least the user MUST have the wheel.

### 2A: Detect existing installs

```bash
DETECT=$(python3 "$SETUP_SCRIPTS_DIR/runner.py" detect_modelzoo '{}')
echo "$DETECT"
```

Response has `repos` (git checkouts), `wheels` (pip installs: `python`, `modelzoo_root`, `version`) and `found`.

**ALWAYS stop and ask the user which installation to use — never auto-select, even if only one candidate was found.** Present every candidate as a numbered option, plus two fixed options:

> "I found these tinyml-modelzoo installations:
>
> 1. `<repo path>` — git repo (v…)
> 2. `<repo path>` — git repo (v…)
> 3. `<modelzoo_root>` — pip wheel in venv `<python>` (v…)
> …
> N+1. **Enter a different path** to a tinyml-modelzoo repo
> N+2. **Clean install** — create a new venv (default `~/.tinyml-agent-skills/venv`, or a location of your choice) and install tinyml-modelzoo as a pip wheel"

If `found: false`, say nothing was found and show only the last two options.

**What each answer does:**

| User picks | Set | Then |
|------------|-----|------|
| A repo | `MODELZOO_INSTALL_MODE=repo`, `TINYML_MODELZOO_PATH=<repo>` | Run `check_installation` (below), then Step 3 repo branch (3A, 3B) |
| A pip wheel | `MODELZOO_INSTALL_MODE=wheel`, `VENV_PYTHON=<its python>` | Step 3 wheel branch (3W) |
| Enter a different path | ask for the path, re-run 2A with `{"extra_repo_paths": ["<path>"]}` | Then treat it as one of the rows above |
| Clean install | `MODELZOO_INSTALL_MODE=wheel` | Go to 2B |

For a repo, verify the path; on `success: false` show `errors` and `hint`, ask for a corrected path, and repeat:
```bash
python3 "$SETUP_SCRIPTS_DIR/runner.py" check_installation \
  "{\"tinyml_modelzoo_path\": \"$TINYML_MODELZOO_PATH\"}"
```

### 2B: Offer to install the wheel

Ask where to create the venv, unless the user already gave a location:

> "I'll create a virtual environment and install the tinyml-modelzoo wheel (which brings in modelmaker, tinyverse and modelopt) into it from TI's wheel server. Where should the venv go? Default: `~/.tinyml-agent-skills/venv`."

Set `WHEEL_VENV` to their answer (expand `~`, make it absolute). If the user declines to install at all, stop: setup cannot continue without modelzoo. If `$WHEEL_VENV` already exists and is non-empty, ask whether to reuse it (skip `venv` creation, go straight to `pip install`) or pick another location. Never delete an existing directory. Then:

```bash
TINYML_WHEEL_VERSION=1.5.0
WHEEL_BASE="http://software-dl.ti.com/C2000/esd/mcu_ai/wheel"
WHEEL_VENV=<the chosen absolute path>   # default: $HOME/.tinyml-agent-skills/venv

python3 -m venv "$WHEEL_VENV"            # requires Python 3.10+
VENV_PYTHON="$WHEEL_VENV/bin/python3"    # Windows: $WHEEL_VENV/Scripts/python.exe
$VENV_PYTHON -m pip install --upgrade pip
# One wheel is enough: it pulls in tinyml_modelmaker, tinyml_tinyverse and tinyml_torchmodelopt
$VENV_PYTHON -m pip install "$WHEEL_BASE/tinyml_modelzoo-$TINYML_WHEEL_VERSION-py3-none-any.whl"
```

Follow the **Install guardrails** above: the `pip install` of the wheel is slow (it pulls in torch and other large dependencies), so run it with the maximum timeout or in the background and wait for it to finish. Do NOT fall back to any detected repo/wheel from 3A while it runs. If any pip command exits non-zero, show the full output and stop. On success: `MODELZOO_INSTALL_MODE=wheel`. This venv is now **the** environment (`VENV_PYTHON`) for everything that follows.

---

## Step 3: Set up virtual environment and install packages

Follow the branch for your install mode, then both branches continue at **3C**:

- **Repo mode (`MODELZOO_INSTALL_MODE=repo`):** do **3A** then **3B**. Skip 3W.
- **Wheel mode (`MODELZOO_INSTALL_MODE=wheel`):** `VENV_PYTHON` is already fixed in Step 2B. Skip 3A and 3B and do **3W**.

### 3A: Detect and activate environment (repo mode only — skip in wheel mode)

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

### 3B: Verify and install packages (repo mode only — skip in wheel mode; VENV_PYTHON is locked)

Use `$VENV_PYTHON` (set in 3A, now locked) — **NEVER test or switch to other Python instances**. If packages are missing, fail setup instead of falling back.

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

**Step 2: Install core packages if missing**

If `PACKAGES_INSTALLED=no` (only reachable when the guardrail above is relaxed by the user):
```bash
cd "$TINYML_MODELZOO_PATH"
$VENV_PYTHON -m pip install -e .
```
(This installs tinyml-modelzoo in editable mode; it pulls in modelmaker, tinyverse and modelopt.)

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

### 3W: Verify wheel install (wheel mode only)

```bash
readonly VENV_PYTHON   # locked, as in repo mode
python3 "$SETUP_SCRIPTS_DIR/runner.py" check_wheel_install "{\"venv_python\": \"$VENV_PYTHON\"}"
```

- `success: false` with modelmaker/tinyverse/modelopt listed in `errors` → this is an older or partial modelzoo install (e.g. a venv with only an early modelzoo wheel). Offer to upgrade it: `$VENV_PYTHON -m pip install --upgrade "$WHEEL_BASE/tinyml_modelzoo-$TINYML_WHEEL_VERSION-py3-none-any.whl"` (define `WHEEL_BASE`/`TINYML_WHEEL_VERSION` as in 3B), then re-check. Any other failure: show it and stop.
- On success set `TINYML_MODELZOO_PATH` to `modelzoo_root` from the response (the venv's `site-packages`, which contains `tinyml_modelzoo/`).

Then continue at **3C**.

### 3C: Discover package paths

After imports succeed, store the installed package root paths — saved to `.env` in Step 5 so the workflow skill can reference modelmaker source files directly.

**CRITICAL: use the correct Python** — bare `python3` may resolve to a system interpreter that finds dev-repo editable installs, giving wrong paths. These packages are dependencies installed via modelzoo, not dev repos.

Use `$VENV_PYTHON` (already set in 3A — venv binary if venv exists, else `which python3`):
```bash
TINYML_MODELMAKER_PATH=$($VENV_PYTHON -c "import tinyml_modelmaker, os; print(os.path.dirname(tinyml_modelmaker.__file__))")
TINYML_TINYVERSE_PATH=$($VENV_PYTHON -c "import tinyml_tinyverse, os; print(os.path.dirname(tinyml_tinyverse.__file__))")
TINYML_MODELOPT_PATH=$($VENV_PYTHON -c "import tinyml_torchmodelopt, os; print(os.path.dirname(tinyml_torchmodelopt.__file__))")
```

Confirm all three are non-empty. In wheel mode the paths must be under the wheel venv. If a venv was detected and any path does NOT contain `.venv`, warn the user — a global or dev-repo editable install is shadowing the venv install.

---

## Step 4: Run verification training

**Wheel mode: skip this step.** The verification example needs a repo checkout (`run_tinyml_modelzoo.sh`); the import check in 3W is the verification. Go to Step 5.

**On re-runs:** if `~/.tinyml-agent-skills/.env` already exists with `IS_REPO_SETUP=1`, and the user is only updating or fixing paths (not re-verifying training), skip this step.

Ask the user:
> "Would you like to run a quick verification training to confirm your environment works end-to-end? This runs a small example and takes a few minutes. You can skip it and go straight to using the skill."

If user skips: proceed directly to Step 5.

If user confirms, tell them: *"Running verification to confirm training and compilation work correctly. This will take a few minutes."*

**First, find the compiler path** — the verification example targets a C2000 device, so look for the C2000 CGT:

Use the IDE's project tools to locate the `TI C2000 CGT` installation: in CCStudio its MCP tools, in VS Code the CLI project tools (discover which subcommand lists installed compilers, e.g. via the tool's `--help`). Do not search the filesystem. If several versions are installed, pick the latest. Extract its root (the directory containing `bin/`) and store it as `CGT_ROOT`.

**If compiler not found:** STOP. Ask the user:
> "Compiler not found. Please provide the full path to your C2000 CGT root (folder containing `bin/cl2000`)."
Do not run verification without the compiler path confirmed.

**Export compiler paths and run:**

CRITICAL: env vars and script invocation MUST be in a single shell command. Each Bash tool call is a separate subshell — `export` in one call does not survive to the next. Python's `constants.py` is imported at process start, so the vars must already be in the environment when the subprocess spawns.

Also CRITICAL: `CGT_ROOT` was discovered from the IDE's tools above — it lives in your context, NOT as a shell variable. You MUST substitute the literal discovered path directly into the command. Do NOT write `$CGT_ROOT` as a shell variable — it will be empty in a new bash call.

The verification example targets a C2000 device — only `C2000_CGT_ROOT` is needed. Substitute the LITERAL PATH where `<ACTUAL_CGT_ROOT_PATH>` appears:

Linux:
```bash
cd "$TINYML_MODELZOO_PATH" && \
  C2000_CGT_ROOT="<ACTUAL_CGT_ROOT_PATH>" \
  ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
```

Windows:
```powershell
$env:C2000_CGT_ROOT="<ACTUAL_CGT_ROOT_PATH>"; cd "$TINYML_MODELZOO_PATH"; run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
```

Stream output to user.

**If verification passes:** proceed to Step 5.

---

## Step 5: Write .env file

Config is written to `~/.tinyml-agent-skills/.env` — user-global, survives plugin updates, reinstalls, and cache clears. Works on all platforms (Linux, macOS, Windows).

**CRITICAL: Inform the user of the location:**
> "Configuration will be saved to: `~/.tinyml-agent-skills/.env`"
> "This persists across all sessions and plugin versions."

Call the setup runner to write the file (cross-platform — no bash file operations):
```bash
python3 "$SETUP_SCRIPTS_DIR/runner.py" save_config \
  "{\"tinyml_modelzoo_path\": \"$TINYML_MODELZOO_PATH\", \"tinyml_modelmaker_path\": \"$TINYML_MODELMAKER_PATH\", \"tinyml_tinyverse_path\": \"$TINYML_TINYVERSE_PATH\", \"tinyml_modelopt_path\": \"$TINYML_MODELOPT_PATH\", \"modelzoo_install_mode\": \"$MODELZOO_INSTALL_MODE\", \"venv_python\": \"$VENV_PYTHON\"}"
```

Check `success: true`. On success, `env_file` in the response will show the exact path created.

`.env` also stores `MODELZOO_INSTALL_MODE` (`repo`|`wheel`) and `VENV_PYTHON`, which the workflow skill uses for every python call.

Note: `SCRIPTS_DIR` is NOT stored in `.env` — it is derived automatically from the installed plugin location at the start of each session.

If `success: false`: show `errors`, do not proceed until resolved.

---

## Setup Complete

Tell the user:

> "✓ Setup complete. You can now use `/tinyml-agent-skills:ti-ccstudio-tinyml-workflow` to start building TinyML models."
>
> "Configuration saved to: `~/.tinyml-agent-skills/.env`"
> "This file is loaded automatically by ti-ccstudio-tinyml-workflow on every session start and survives plugin updates."
> "If you move tinyml-modelzoo or change your setup, re-run this setup skill."

Re-run this setup skill any time you:
- Move or reinstall tinyml-modelzoo
- Switch to a different tinyml-modelzoo version
- Encounter: "Run `/tinyml-agent-skills:setup` first" (means `~/.tinyml-agent-skills/.env` is missing or incomplete)
