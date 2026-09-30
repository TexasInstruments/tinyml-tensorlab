# How to build and publish the modelmaker/tinyverse/torchmodelopt wheels

## Problem

`tinyml-modelzoo` is pushed to GitHub as source only, so external customers can
`git clone` it and `pip install -e .` to get an editable modelzoo plus example
configs. That install only works if modelzoo's `pyproject.toml` dependency on
`tinyml_modelmaker` resolves to a real, downloadable wheel — and that wheel's
own dependencies (`tinyml_tinyverse`, `tinyml_torchmodelopt`, `ti_mcu_nnc`)
must resolve the same way. This guide covers building those wheels from your
local source checkouts and getting them onto TI's CDN at the exact URL each
`pyproject.toml` already points to.

## Solution

Run `tinyml-modelmaker/build_wheels.sh` to build the wheels from local
source, then upload the `.whl` files to
`https://software-dl.ti.com/C2000/esd/mcu_ai/wheel/` yourself (the script
cannot do this last step — that location isn't writable from a dev machine).

## Steps

1. **Bump versions consistently.** Every `pyproject.toml` involved
   (`tinyml-modelmaker`, `tinyml-tinyverse`, `tinyml-modeloptimization/torchmodelopt`,
   `tinyml-modelzoo`) pins sibling deps by exact version in the wheel filename,
   e.g. `tinyml_tinyverse-1.5.0-py3-none-any.whl`. If you're cutting a new
   release, bump `version = "..."` in all of them to the same string before
   building — a mismatch here is the single most common way this pipeline
   breaks.

2. **Build the wheels.** From `tinyml-modelmaker/`:
   ```bash
   pip install build   # once, if not already installed
   ./build_wheels.sh
   ```
   This builds `tinyml_torchmodelopt`, `tinyml_modelzoo`, `tinyml_tinyverse`,
   and `tinyml_modelmaker` from whatever's checked out at
   `../tinyml-tinyverse`, `../tinyml-modeloptimization/torchmodelopt`, and
   `../tinyml-modelzoo` (override with `TINYVERSE_DIR`, `TORCHMODELOPT_DIR`,
   `MODELZOO_DIR` env vars if your layout differs). Output lands in
   `./dist_wheels` (override with `OUT_DIR`).

   **You only need 3 of the 4 wheels for this use case.** `tinyml_modelzoo`
   is also built here, but that's for a *different* consumer — the
   `tinyml-mlbackend` Docker image (see its own how-to guide) — not for the
   GitHub-source-plus-editable-install flow this guide is about. Don't
   publish the modelzoo wheel thinking customers need it; they get modelzoo
   from GitHub, not from the CDN.

3. **Verify the build.**
   ```bash
   ls -la dist_wheels/*.whl
   ```
   You should see `tinyml_modelmaker-<version>-py3-none-any.whl`,
   `tinyml_tinyverse-<version>-py3-none-any.whl`,
   `tinyml_torchmodelopt-<version>-py3-none-any.whl`, and
   `tinyml_modelzoo-<version>-py3-none-any.whl` (the last one only relevant
   per the note above).

4. **Upload the 3 wheels to the CDN.** Using whatever access you have to
   `software-dl.ti.com`, place the modelmaker/tinyverse/torchmodelopt wheels
   at:
   ```
   https://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelmaker-<version>-py3-none-any.whl
   https://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_tinyverse-<version>-py3-none-any.whl
   https://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_torchmodelopt-<version>-py3-none-any.whl
   ```
   The filename must match exactly what's hardcoded in the consuming
   `pyproject.toml` files — a version or filename typo here 404s every
   customer install, silently, until someone tries it.

5. **Confirm the URLs are live** before telling anyone to install:
   ```bash
   curl -sI https://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelmaker-<version>-py3-none-any.whl | head -1
   ```
   Expect `HTTP/2 200`. Repeat for tinyverse and torchmodelopt.

6. **Push `tinyml-modelzoo` to GitHub** (manual, outside these scripts — this
   guide only covers the wheel side). Its `pyproject.toml` already pins
   `tinyml_modelmaker @ https://software-dl.ti.com/.../tinyml_modelmaker-<version>-py3-none-any.whl`,
   so once step 4 is live, customers get the whole stack transitively.

7. **Sanity-test the customer path end to end** before calling it done:
   ```bash
   python -m venv /tmp/customer-test && source /tmp/customer-test/bin/activate
   git clone https://github.com/TexasInstruments/tinyml-modelzoo.git /tmp/modelzoo-test
   cd /tmp/modelzoo-test && pip install -e .
   python -c "import tinyml_tinyverse, tinyml_modelmaker, tinyml_torchmodelopt"
   ```
   If this venv has never seen your local sibling checkouts, a successful
   import here proves the CDN wheels — not a stray local install — are what
   actually got pulled in.

## Variations

- **Testing before publishing anywhere:** point `HOST_DIR` at a local
  directory and serve it with `python -m http.server`, then temporarily
  patch the relevant `pyproject.toml`/CDN references to that local server's
  URL. `build_wheels.sh`'s header comments cover this via `HOST_BASE_URL`.
- **Re-releasing after a hotfix to just one repo:** you still need to rebuild
  and republish all 3 wheels if any of them changed, and bump the version in
  every `pyproject.toml` that pins it — partial updates leave stale pinned
  versions pointing at an old wheel.

## See also

- `docker-build.md` — the 4th wheel (`tinyml_modelzoo`) this script also
  builds is for that flow, not this one.
- `local-dev-multi-repo-install.md` — for your own dev loop across these
  repos, you don't want wheels at all; use editable installs instead.
