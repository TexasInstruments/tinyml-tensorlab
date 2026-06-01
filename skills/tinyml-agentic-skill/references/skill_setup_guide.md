This guide explains how the skill must be setup - including steps for verifying the tinyml-tensorlab repository setup, and explains what env variables must be set. It further lays down the rules for storing variables from one-time activities to prevent setup activities from running every time.

1. In order to find the scripts directory and set the `SCRIPTS_DIR` variable (if not yet set), run:
```bash
SCRIPTS_DIR="$(python3 -c "import os; print(os.path.dirname(os.path.abspath('$(find ~/.claude -name runner.py 2>/dev/null | head -1)')))" 2>/dev/null || echo "/home/$(whoami)/.claude/plugins/cache/claude-plugins-official/tinyml-tensorlab-skills/unknown/skills/tinyml-tensorlab-skills/scripts")"
```

Simpler: the scripts directory is always at the path shown in the skill's own directory. Ask the user to confirm the skill path if needed, or hard-code after first discovery:
```bash
SCRIPTS_DIR=<absolute path to the scripts/ folder of this skill>
```

2. In order to figure out where the tinyml-tensorlab repository has been installed by the user, follow the below steps:
**Confirm the tinyML tensorlab installation:**

Ask user: *"What is the full path to your tinyml-tensorlab directory?"*
(e.g., `/home/username/tinyml-tensorlab`)

```bash
TINYML_BASE_PATH=<user-provided path>

# Verify it
python3 $SCRIPTS_DIR/runner.py check_installation \
  "{\"tinyml_base_path\": \"$TINYML_BASE_PATH\"}"
```

If `success: false`, show the user the `errors` and `hint` fields and ask them to correct the path.

3. In order to set up the tinyml-tensorlab repository and it's virtual environment and/or if `IS_REPO_SETUP` is not set, follow the below steps:
**Setup repo and environment for working**

Once the repository is installed and the installation is verified, also verify if the repository has been set-up with a virtual environment and is ready for use. 
First check to find a virtual environment which may be present at the root level of the tinyml-tensorlab repository (installed at `$TINYML_BASE_PATH`).
Try `source .venv/bin/activate` or `source venv/bin/activate`.
Then try running the following script:
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

If all packages import without errors and versions are displayed, your installation is complete.
If `IS_REPO_SETUP` is not set to true/1, then and ONLY THEN, do the below:
Then try running the below to ensure everything is 'good-to-go':
Linux
  ```bash
      cd tinyml-modelzoo
      ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml
  ```
Windows
  ```powershell

      cd tinyml-modelzoo
      run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml
  ```               
**Stream the bash/powershell outputs to the user.**
Verify training and compilation is working properly - **INFORM THE USER THAT YOU ARE RUNNING THIS VERIFICATION - IT WILL HAPPEN ONLY ONCE FOR THE SESSION**
Once done, if everything looks good, store an environment variable **PERMANENTLY** `IS_REPO_SETUP` as true/1. And during the session, anytime a new query starts, you can refer this variable to ensure it is set up.

If compilation fails due to lack of path to compiler, search for the compiler, automatically find path and set the required environment variables and export them in the session as well. Then re-run the above script.

If path cannot be found, inform the user and ask them to install the same. You can refer them to `https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/user_guide/installation/environment_variables.html#`.

If the above verification fails, then you need to set-up the repository and the virtual environment for it. To do so, follow `references\setup_guide.md`.
Once it is set up, set the `IS_REPO_SETUP` variable as true/1.

**Post this, set `TINYML_TENSORLAB_DOCS_PATH` to `$TINYML_BASE_PATH/docs/source/` — the RST source tree is at `docs/source/`, not `docs/`. You can follow `references/documentation_guide.md` to understand how to navigate documentation. Refer it any time you need information not already covered in `references/` or `assets/`.**

Once all of the above have been set up, create a new config file within `references` - title it `.setup_cfg.cfg`.
It must contain all of the below environment variables and their corresponding values:

| Variable          | Source         | Description                              |
|-------------------|----------------|------------------------------------------|
| `SCRIPTS_DIR`     | Setup          | Absolute path to scripts/ folder         |
| `IS_REPO_SETUP`   | Setup          | Indicates if tinyml-tensorlab is setup or not |
| `TINYML_BASE_PATH`| Setup          | Root of tinyml-tensorlab repo        |
| `TINYML_TENSORLAB_DOCS_PATH`       | Setup          | Path to documentation for tinyml-tensorlab |

Make sure all of the above variables are set and correct:
```python
import os
print(os.environ.get("SCRIPTS_DIR"))
print(os.environ.get("IS_REPO_SETUP"))
print(os.environ.get("TINYML_BASE_PATH"))
print(os.environ.get("TINYML_TENSORLAB_DOCS_PATH"))
```
If not, then immediately flag this to the user and do not proceed until the above test passes.

**VALIDATE `TINYML_TENSORLAB_DOCS_PATH` is correct** — it must contain RST subdirectories. Run:
```bash
ls $TINYML_TENSORLAB_DOCS_PATH/getting_started/ 2>/dev/null | head -3
```
If this returns no output or an error, the path is wrong. Correct value is `$TINYML_BASE_PATH/docs/source/`. Fix it before writing to `.env`.

**ALWAYS CREATE A LOCAL `.env` FILE WITH ALL OF THE ABOVE 4 ENV VARIABLES**