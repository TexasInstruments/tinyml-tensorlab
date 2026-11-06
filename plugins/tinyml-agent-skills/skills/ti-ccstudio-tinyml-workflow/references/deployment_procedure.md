# Deployment Procedure (Steps 14A–14E mechanics)

Read this in full when `SKILL.md` Step 14 tells you to. `SKILL.md` holds the checklist and the gates; this file holds the exact commands and fallbacks. Lookup tables and troubleshooting live in `device_deployment_guide.md`.

Shell variables do not persist between tool calls. Substitute literal values for `$PROJECT_PATH`, `$ARTIFACT_TARGET_DIR`, etc. in every command, or re-derive them in the same call.

## Contents
- 14B: PASS-check script for the `.cproject` exclusions (CCStudio only)
- 14D: build (whatever the IDE offers, or the user builds)
- 14E: flash (whatever the IDE offers, or the user flashes)

The exclusion edit (14B) and the copy/verify commands (14C) are written out in full in `SKILL.md`; they are not repeated here.

---

## 14B: PASS-check for the `.cproject` exclusions (CCStudio only)

In VS Code there is no script to run: follow the VS Code paragraph of `SKILL.md` Step 14B (tell the user the items are excluded by default, check `.ccs/projects.json` for them, report what you find, do not edit it). In CCStudio, run after editing `.cproject` as described in `SKILL.md` Step 14B. It checks every `<configuration>` block (Debug, Release, ...) and prints `PASS` only if each excludes `run`, `config.yaml` and `.tinyml_template.json`:

```bash
$VENV_PYTHON - "$PROJECT_PATH/.cproject" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
need = {"run", "config.yaml", ".tinyml_template.json"}
confs = list(ET.parse(sys.argv[1]).getroot().iter("configuration"))
if not confs:
    sys.exit("FAIL: no <configuration> in .cproject")
bad = []
for c in confs:
    ex = set()
    for se in c.iter("sourceEntries"):
        for e in se.iter("entry"):
            ex |= set(e.get("excluding", "").split("|"))
    if need - ex:
        bad.append((c.get("id"), sorted(need - ex)))
sys.exit(f"FAIL: missing exclusions {bad}") if bad else print("PASS: every configuration excludes", sorted(need))
PYEOF
```

On FAIL, apply the `SKILL.md` Step 14B edit to the listed configuration(s) and re-run. Do not continue until it prints PASS.

---

## 14D — Build

The goal: the project builds cleanly with the trained artifacts. Do it with whatever the IDE gives you (CCStudio: its project/build tools; VS Code: the CLI project and build tools), in the same environment the project was imported into. No particular tool is prescribed.

- **Clear stale debug sessions first.** An active debug session holds a lock on the project and can make a build hang. Terminate any session for `$PROJECT_NAME` with the IDE's debug tooling before building. If you cannot, and the build hangs, ask the user to close debug sessions and retry.
- **A compiler or linker error is a real build failure.** Fix the cause (wrong artifacts, missing header, exclusions on CCStudio); do not switch tools to get around it.
- **Tooling could not be used at all** (project not found or not registered, tool not configured, IDE not running): this is not a build error. Stop, give the exact reason, and ask the user either to enable/start the tooling so you can retry, or to build in their IDE and report the result:

> "I can't build automatically ([exact reason]). Please either enable/start the IDE's project tooling and tell me when it is ready, or build the project in your IDE and tell me whether it finished with no errors."

Gate G5 passes on a build you observed succeed, or on the user's explicit confirmation of a clean build.

---

## 14E — Flash

The device must be connected by USB/JTAG. The goal: the built program is loaded onto the device and running, with the correct target configuration. Use whatever debug/flash tooling the IDE offers.

Typical sequence (adapt to the tools you have):
1. End any existing debug session for `$PROJECT_NAME`.
2. Start a debug/flash session for the project (it loads the program). If the program did not load, load `$PROJECT_PATH/<build config>/$PROJECT_NAME.out` explicitly.
3. Resume/run the program.

Starting a session normally takes 30–90 s on hardware (probe negotiation, erase/write, reset); treat it as hung only after 3 minutes.

On failure:
- `device not found` / `no target connected` → physical problem, not a tooling problem. Ask the user to check USB/JTAG and retry.
- project not found / IDE not running / tool not configured → tooling unusable. Stop, give the exact reason, and ask the user to enable it, or to flash in their IDE and report the result:

> "I can't flash automatically ([exact reason]). Please either enable/start the IDE's debug tooling and tell me when it is ready, or flash the program from your IDE and tell me the result."

- other runtime error → report it as is.

**After flashing:** see "Verify model on device" in `device_deployment_guide.md` (`test_result == 1` means pass).
