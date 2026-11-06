# Project Import Details (Step 3A–3D)

Step 3 imports an SDK AI-example template into a new project in the user's IDE (CCStudio or VS Code). That project becomes the home for the config, training outputs and compilation artifacts. Use whatever project tooling the IDE offers for the import (see the "IDE and tooling" section in `SKILL.md`); if there is none you can drive, tell the user why and ask them to import the `.projectspec` themselves, then continue with the project path they give you.

## 3A. Find SDK root and AI examples

Use the IDE's project tools to find the installed SDK matching the `$TARGET_DEVICE` family (C2000Ware for F28xxx, MSPM0 SDK for MSPM0G, MCU+ SDK for AM26x). In CCStudio use its MCP tools; in VS Code use the CLI project tools (discover which subcommand lists installed products/SDKs, e.g. via the tool's `--help`). Do not search the filesystem. If several versions are installed, **pick the latest version** and tell the user which one. Its install path is `$SDK_ROOT`. If the tools cannot find it, ask the user for the SDK path, or to install it first, and stop until you have it.

Locate the AI examples. Do not assume a fixed subpath; it varies by SDK family and version:

```bash
find "$SDK_ROOT" -maxdepth 7 -type d \( -name "ai" -o -name "edgeAI" -o -name "edge_ai" \) 2>/dev/null
```

Pick the directory holding task-like subdirectories (e.g. `generic_timeseries_classification/`, `motor_fault_ai_npu/`); if several, the one whose names best match task and device. Store as `$AI_EXAMPLES_PATH`.

## 3B. Select template

Selection runs in this order. Stop at the first step that yields a template.

1. **Exact match**: a subdirectory whose name contains `$TASK_TYPE` (e.g. `motor_fault_ai_npu` for `motor_fault`). An exact match is a full application of that task, so it is used as-is.
2. **Generic same-task template** (only if there is no exact match):

| `$TASK_TYPE` pattern | Generic directory |
|---|---|
| `*classification*` | `generic_timeseries_classification` |
| `*anomalydetection*` | `generic_timeseries_anomalydetection` |
| `*regression*` | `generic_timeseries_regression` |
| `*forecasting*` | `generic_timeseries_forecasting` |

3. **Bare starter** (only if neither 1 nor 2 exists, and only with the user's OK). A bare starter is a blank project with no application of its own: `empty_*` (e.g. `empty_npu_ai`) or `hello_world_ai*`. Pick the one matching `$TARGET_DEVICE`'s family and NPU status (see 3B-NPU). Its golden-vector test is written into its `main.c` (see `references/golden_vector_test.md`, "Bare starter").
4. **Nothing suitable**: stop and ask (Gate G0). List only the hosts that have a golden-vector check (filter below). If none exists, say so and stop. Never list or suggest any other example project.

**Golden-vector host filter.** A host qualifies only if its folder is a generic or bare starter (rules 2 and 3) and, for rule 2, its sources contain `test_result`:

```bash
ls -1 "$AI_EXAMPLES_PATH" | grep -E 'generic_timeseries_|^empty_|^hello_world_ai' | while read d; do
  grep -rlq "test_result" "$AI_EXAMPLES_PATH/$d" --include=*.c 2>/dev/null && echo "GV-check: $d"
done
```

Only the names this filter prints may be offered. Other example folders (applications such as `motor_bearing_fault_f28p55`) are other applications, even if they have a `test_result` check, and are never offered.

**Gate G0 prompt** (rule 4, or rule 3 when a starter is proposed):
> "No template in this SDK matches `$TASK_TYPE`. [If GV hosts exist:] These SDK starters have a golden-vector check: [list]. [If only a bare starter exists:] For a new application I'd import the blank starter `<name>` as the project and write a golden-vector test into its `main.c`. This changes the starter's code, so I need your OK. Use `<name>`, or stop?"

Wait for an explicit answer. Only an explicit choice sets `$TEMPLATE_DIR`. For a bare starter the GV test is required, not optional: a starter has no `test_result` check and may have no model call (`empty_*` projects have a `<YOUR_..._HERE>` placeholder). Say that before importing.

If `$AI_EXAMPLES_PATH` itself was not found in 3A, stop and tell the user too; the SDK version may be too old to include AI examples.

Store the result as `$TEMPLATE_DIR` (`$AI_EXAMPLES_PATH/<subdir>`).

**NPU vs regular variant.** On an NPU device the SDK may hold two versions of a template, e.g. `hello_world_ai` and `hello_world_ai_npu` (MSPM0), or an `_npu` one only. If both exist for the chosen template, tell the user: "This device has an NPU, so the NPU project is the better fit. Which do you want?" Then stop and wait for the reply: do not import first, and do not announce a choice for them. The decision is theirs; if they pick the regular project, the config is adjusted as described in `SKILL.md` Step 3 (non-NPU project on an NPU device). Devices without an NPU have only regular templates, so nothing to ask.

## 3C. Read template README and confirm requirements

```bash
find "$TEMPLATE_DIR" \( -iname "README*" -o -iname "*readme*" \) -type f 2>/dev/null | head -10
```

**MANDATORY, NEVER SKIPPED (Gate G0b).** This question is asked every single time a template with a README is imported (the only exception is a verified absence of any README, see "No README" below), for every device, task and template (exact match, generic fallback, NPU or regular variant), even if the user said earlier they have the hardware, even if they are in a hurry, even on a repeat run, and even if the README is short or lists nothing. Do not infer the answer from earlier messages, and do not fold it into another question. Do not start 3D until the user has replied to this specific question.

Read it and extract hardware requirements (LaunchPads, booster packs, EVMs, sensors, board revisions), software prerequisites (SDK versions, drivers, firmware), physical connections or jumpers, and any other prerequisites. Present them:

> "Before importing, this example requires: **Hardware:** [...] **Software:** [...] **Other:** [...]. Do you have everything listed? If not, please set it up before we continue."

End your turn after asking and wait for the reply; a recommendation or announcement is not an answer. If the user is missing something, help resolve it first, then re-ask. Cover accelerators and extra hardware (LaunchPad, BoosterPack, EVM, sensor boards, debug probe, cables, jumpers) and software (SDK version, IDE, compiler, drivers, firmware).

**No README.** Only if you are sure the template has none, continue silently to 3D. "Sure" means the case-insensitive search above (whole template tree, any extension) printed nothing, and the template directory itself exists and is non-empty. If the search errored, the directory is missing or empty, or you are in any doubt, you are not sure: look again (`ls -R "$TEMPLATE_DIR" | head -50`) and do not skip the question.

**README exists but lists no requirements.** Still ask, generically: "The template README lists no requirements. For `$TARGET_DEVICE`, do you have the LaunchPad/EVM, any BoosterPack or sensor board this example uses, a USB cable/debug probe, and your IDE plus the SDK and compiler installed?" 

## 3D. Import into the workspace

Ask: "Where should this project be created? (full path, or just a name to use your default workspace)". Set `$WORKSPACE_DIR` and `$PROJECT_NAME`.

**Find the `.projectspec` first. Never construct its path from a pattern or a fixed folder name.** Its location and file name vary by SDK, board and template (`CCS/`, `ticlang/`, `nortos/ticlang/`, `freertos/ticlang/`, `ti_arm_clang/`...; names like `generic_f29h85x.projectspec`), so a guessed path gives `file not found`. In the same turn, before the first import attempt, search the template recursively:

```bash
find "$TEMPLATE_DIR" -name "*.projectspec" -type f 2>/dev/null | sort
```

Import only a file this command printed. If it prints nothing, do not import and do not guess: look for what the template does contain (`ls -R "$TEMPLATE_DIR" | head -50`, and `find "$TEMPLATE_DIR" -name "*.ccsproject" -o -name ".cproject" -o -name "project*.js"`), tell the user the template has no importable `.projectspec`, and ask how to proceed. Never fall back to a different template.

Pick exactly one file from the results:
1. Keep only files under the sub-directory for `$TARGET_DEVICE` (match the device or board in the path, case-insensitively, e.g. `f29h85x`, `mspm0g5187`, `LP_MSPM0G5187`, `am263x`). One template folder often holds several devices.
2. Drop files whose name starts with `test_` or that are clearly variants for another purpose.
3. Prefer a path containing the toolchain the project uses (`ticlang` / `ti_arm_clang`) if several remain.
4. If exactly one remains, use it. If zero remain, or more than one is genuinely ambiguous, show the list and ask the user which; never guess, and never fall back to another template.

Store the chosen file as `$PROJECTSPEC`. If the path has spaces, quote it.

Import `$PROJECTSPEC` as a project at `$WORKSPACE_DIR/$PROJECT_NAME` using the IDE's tools, whichever they are.

Save the resulting project path as `$PROJECT_PATH` and the project name as `$PROJECT_NAME`.

On an NPU device, record the choice in `$PROJECT_PATH/.tinyml_template.json` right after the import, so it survives later steps and sessions:
```bash
$VENV_PYTHON - "$PROJECT_PATH" "<TEMPLATE_DIR>" <true|false> <<'PYEOF'
import json, sys
root, template_dir, npu = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
json.dump({"template_dir": template_dir, "npu_template": npu}, open(f"{root}/.tinyml_template.json", "w"), indent=2)
PYEOF
```
`npu_template` is true if the imported template name ends in `_npu`, else false. Devices without an NPU do not need the file. It is kept out of the build in Step 14B.

If the template name differs from the task type (generic fallback or a template the user picked), rename the project to a task-specific name using the IDE's tools (or move the folder and re-import) and update `$PROJECT_NAME`.

Confirm the project is registered with the IDE under its name (whatever listing the tools offer, or the user confirming it appears in the IDE). Later build and flash steps depend on that registration.

On failure:
- `file not found` / bad path → the `.projectspec` path was wrong. Re-run the `find` above, pick the file from its output (never a guessed path), and retry. Do not fall back to another template.
- IDE tooling unavailable (IDE not running, tool not configured) → stop and tell the user the exact reason; ask them to start or enable it, or to import the `.projectspec` themselves and give you the project path.
