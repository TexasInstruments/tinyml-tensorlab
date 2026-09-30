#!/bin/bash
#
# Build tinyml_modelmaker + its sibling wheels (including tinyml_modelzoo)
# from LOCAL source, and optionally copy them into a directory served to
# your wheel host.
#
# pyproject.toml's tinyml_tinyverse/tinyml_torchmodelopt/tinyml_modelzoo deps
# now point at TI's official wheel CDN (software-dl.ti.com/C2000/esd/mcu_ai/wheel/)
# by default, so a plain build already produces wheels whose metadata matches
# where they're actually hosted - no patching needed for that case.
#
# The patch-before-build/restore-after step below is a safety net for the
# case where pyproject.toml points somewhere else at build time (e.g. a
# branch that still has git+github deps, or testing against a different
# host via HOST_BASE_URL): it rewrites a working copy of pyproject.toml so
# the sibling-dep lines point at HOST_BASE_URL instead, builds against that,
# then restores the original file. Nothing you `git diff` after running this
# should show any change either way - if the current lines already match
# HOST_BASE_URL, the patch pattern simply doesn't match and nothing happens.
#
# Re-run this any time tinyverse/torchmodelopt/modelmaker/modelzoo source
# changes; nothing here detects staleness for you. The built wheels land in
# OUT_DIR (./dist_wheels by default) either way - software-dl.ti.com is not
# writable from here, so getting them onto the actual CDN is a separate,
# manual step outside this script (upload/copy via whatever access you have
# to that location). HOST_DIR is only for copying into a LOCAL directory you
# control (e.g. one served by a local http.server for testing, or a staging
# dir before you upload it yourself) - it is never a URL.
#
# tinyml_modelzoo IS built here now (previously excluded, since modelmaker/
# tinyverse used to depend on it as a loose local-editable constraint) -
# building it lets consumers (e.g. tinyml-mlbackend's docker image) install
# the whole stack via `pip install tinyml_modelzoo` with zero sibling repos
# checked out locally.
#
# Usage:
#   ./build_wheels.sh
#   TINYVERSE_DIR=/path/to/tinyml-tinyverse ./build_wheels.sh
#   HOST_DIR=/path/to/local/staging/dir ./build_wheels.sh
#   HOST_BASE_URL=http://other-host:8100/wheels ./build_wheels.sh   # only affects the safety-net patch, not where files get copied

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELMAKER_DIR=${MODELMAKER_DIR:-"$SCRIPT_DIR"}
TINYVERSE_DIR=${TINYVERSE_DIR:-"$SCRIPT_DIR/../tinyml-tinyverse"}
TORCHMODELOPT_DIR=${TORCHMODELOPT_DIR:-"$SCRIPT_DIR/../tinyml-modeloptimization/torchmodelopt"}
MODELZOO_DIR=${MODELZOO_DIR:-"$SCRIPT_DIR/../tinyml-modelzoo"}
OUT_DIR=${OUT_DIR:-"$SCRIPT_DIR/dist_wheels"}
HOST_DIR=${HOST_DIR:-}
HOST_BASE_URL=${HOST_BASE_URL:-"https://software-dl.ti.com/C2000/esd/mcu_ai/wheel"}

if ! python -c "import build" >/dev/null 2>&1; then
    echo "Error: the 'build' package is not installed in this Python environment."
    echo "Run: pip install build"
    exit 1
fi

mkdir -p "$OUT_DIR"

# Paths we may have patched, so the EXIT trap can restore them no matter how
# this script ends (success, build failure, Ctrl-C).
PATCHED_FILES=()
restore_patched() {
    for f in "${PATCHED_FILES[@]}"; do
        if [ -f "$f.orig" ]; then
            mv "$f.orig" "$f"
        fi
    done
}
trap restore_patched EXIT

patch_pyproject_for_local_build() {
    local dir="$1"
    local pyproject="$dir/pyproject.toml"
    local version
    version=$(grep -m1 '^version = ' "$pyproject" | sed -E 's/version = "(.*)"/\1/')

    if ! grep -q 'git+https://github.com/TexasInstruments/tinyml-tensorlab' "$pyproject"; then
        return  # already local-hosted (or has no such deps) - nothing to patch
    fi

    cp "$pyproject" "$pyproject.orig"
    PATCHED_FILES+=("$pyproject")
    sed -i -E \
        -e "s|tinyml_modelzoo @ git\+https://github\.com/TexasInstruments/tinyml-tensorlab\.git@[^\"]*|tinyml_modelzoo>=${version}|" \
        -e "s|tinyml_tinyverse @ git\+https://github\.com/TexasInstruments/tinyml-tensorlab\.git@[^\"]*|tinyml_tinyverse @ ${HOST_BASE_URL}/tinyml_tinyverse-${version}-py3-none-any.whl|" \
        -e "s|tinyml_torchmodelopt @ git\+https://github\.com/TexasInstruments/tinyml-tensorlab\.git@[^\"]*|tinyml_torchmodelopt @ ${HOST_BASE_URL}/tinyml_torchmodelopt-${version}-py3-none-any.whl|" \
        "$pyproject"
    echo "  (patched $pyproject for this build only - will restore on exit)"
}

for dir_name in "TORCHMODELOPT_DIR:$TORCHMODELOPT_DIR" "MODELZOO_DIR:$MODELZOO_DIR" "TINYVERSE_DIR:$TINYVERSE_DIR" "MODELMAKER_DIR:$MODELMAKER_DIR"; do
    label="${dir_name%%:*}"
    dir="${dir_name#*:}"
    if [ ! -f "$dir/pyproject.toml" ]; then
        echo "Error: $label ($dir) has no pyproject.toml - check the path."
        exit 1
    fi
    echo "=== Building wheel from $dir ==="
    patch_pyproject_for_local_build "$dir"
    python -m build --wheel --outdir "$OUT_DIR" "$dir"
done

echo ""
echo "Built wheels:"
ls -la "$OUT_DIR"/*.whl

if [ -n "$HOST_DIR" ]; then
    if [ ! -d "$HOST_DIR" ]; then
        echo "Error: HOST_DIR ($HOST_DIR) does not exist."
        exit 1
    fi
    echo ""
    echo "=== Copying to $HOST_DIR ==="
    cp "$OUT_DIR"/*.whl "$HOST_DIR/"
    ls -la "$HOST_DIR"/*.whl
fi
