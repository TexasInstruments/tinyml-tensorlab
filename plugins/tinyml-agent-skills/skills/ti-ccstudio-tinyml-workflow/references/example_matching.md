# Bundled Example Matching Details (Step 3.5)

Examples ship with this skill in `references/examples/` for both repo and wheel installs. Do not look under `$TINYML_MODELZOO_PATH`.

**What these are.** They are TinyML ModelZoo examples: a modelzoo `config.yaml` plus the example dataset URL/path inside it. They are NOT SDK templates and NOT part of the project imported in Step 3 (that project only receives the outputs). Always go through `references/examples/` first, read the dataset location from the matching example's config (`dataset.input_data_path`), and in everything shown to the user call them "TinyML ModelZoo examples" and say the dataset comes from the example's config. Never ask a dataset question before this step has been presented and answered.

```bash
EXAMPLES_DIR="$SCRIPTS_DIR/../references/examples"
ls "$EXAMPLES_DIR"
```

**Version check (once per session).** Compare the examples' version with the installed modelzoo:

```bash
echo "examples: $(cat "$EXAMPLES_DIR/VERSION")  modelzoo: $($VENV_PYTHON -c 'import tinyml_modelzoo as m; print(m.__version__)')"
```

If they differ, warn that bundled configs may reference models or fields the installed version lacks, and continue only if the user agrees.

**Find matches** (searches every `config*.yaml` in every example folder, including variants like `config_MSPM0.yaml`):

```bash
$VENV_PYTHON $SCRIPTS_DIR/runner.py find_matching_examples \
  "{\"task_type\": \"$TASK_TYPE\", \"target_device\": \"$TARGET_DEVICE\"}"
```

Result lists, each entry with `example_name`, `config_file`, `model_name`:
- `exact`: same task type and device
- `same_task`: same task type, other device (template only; device must change)
- `same_device`: same device, other task type (least useful)

The dataset URL or path is not a match criterion; users describe use cases semantically.

**Always tell the user what was found** (never skip silently).

- `exact` non-empty: show every entry (name, config file, model, one line from the example's `readme*.md`). If more than one, ask which. Then offer:
  > "I found a TinyML ModelZoo example (not the SDK template you imported) that matches your task and device: `<example_name>` (`<config_file>`), with its own example dataset `<input_data_path from that config>`. How would you like to proceed?
  > 1. **Use as-is**: train with this ModelZoo example's config and dataset (fastest, no custom data)
  > 2. **Use as template, bring my own dataset**: copy the ModelZoo example's config, point it at your data
  > 3. **Build from scratch**: ignore the example, configure everything"
- `exact` empty (whatever `same_task` / `same_device` hold): there is no ModelZoo example for this task on this device, so the only path is to build the config from scratch. Do not offer, list or suggest any example, template or "use as-is" option. Tell the user in one line, e.g. "No TinyML ModelZoo example matches this task and device, so we will build the config from scratch," and continue at Step 4.
- The user may name an example directly (e.g. "use motor_bearing_fault"); use it after warning about any task/device mismatch.

`<config_file>` below is the chosen entry's file (usually `config.yaml`, sometimes a device variant). It is always copied to `$PROJECT_PATH/config.yaml`.

## Option 1: use as-is

1. Copy the config, plus any helper scripts (`*.py`) next to it the config relies on:
   ```bash
   EXAMPLE_PATH="$EXAMPLES_DIR/<example_name>"
   cp "$EXAMPLE_PATH/<config_file>" "$PROJECT_PATH/config.yaml"
   ```
2. Do not set output paths here. Step 13 writes `training.train_output_path` and `compilation.compile_output_path` (absolute, inside `$PROJECT_PATH/run/<date-time>/<model_name>/`, nested in their sections, never at the root) before every run. Examples often carry their own relative `train_output_path` / `compile_output_path`; Step 13 overwrites them.
3. If `.tinyml_template.json` says `npu_template: false` (non-NPU project on an NPU device), also set `cfg['training']['quantization'] = 1` and `cfg['compilation']['compile_preset_name'] = 'forced_soft_npu_preset'` with a short `yaml.safe_load` / `yaml.safe_dump` snippet. Example configs often omit these keys, so add them.
4. The dataset is already defined in the config. Skip Steps 4–12 and go to Step 13.

## Option 2: example as template, custom dataset

1. Ask for dataset path, channel count and task name.
2. Do Option 1 steps 1–3 (the non-NPU override included).
3. Set the dataset:
   ```bash
   $VENV_PYTHON -c "
   import yaml
   path = '$PROJECT_PATH/config.yaml'
   with open(path) as f: cfg = yaml.safe_load(f)
   cfg.setdefault('dataset', {})['input_data_path'] = '$USER_DATASET_PATH'
   cfg.setdefault('common', {})['variables'] = $VARIABLES
   with open(path, 'w') as f: yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
   print('Done.')
   "
   ```
4. This is a BYOD run (the user's own dataset), so record it; Step 13B (mandatory result review) reads it:
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
5. Skip Steps 4–12 and go to Step 13.

## Option 3: from scratch

Go to Step 4.
