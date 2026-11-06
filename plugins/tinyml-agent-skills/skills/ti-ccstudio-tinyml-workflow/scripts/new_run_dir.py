#!/usr/bin/env python3
"""Point config.yaml at a brand-new run folder and print it.

Usage: new_run_dir.py <PROJECT_PATH>

Creates the path layout the modelzoo uses, inside the project:
    <PROJECT_PATH>/run/<yyyymmdd-hhmmss>/<model_name>/training
    <PROJECT_PATH>/run/<yyyymmdd-hhmmss>/<model_name>/compilation
and writes them to training.train_output_path / compilation.compile_output_path
(inside their sections, never at the config root). Run it before EVERY training
run so a re-run never reuses or overwrites an earlier run folder.
Last output line: RUN_DIR=<absolute path>
"""
import datetime
import os
import sys

import yaml


def main(root):
    path = os.path.join(root, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model = (cfg.get("training") or {}).get("model_name")
    if not model:
        sys.exit("training.model_name missing in config.yaml; set it first")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(root, "run", stamp, model)
    while os.path.exists(run_dir):  # same-second re-run: never reuse a folder
        stamp = (datetime.datetime.strptime(stamp, "%Y%m%d-%H%M%S")
                 + datetime.timedelta(seconds=1)).strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(root, "run", stamp, model)
    os.makedirs(run_dir)  # reserve the folder so a same-second re-run gets a different one
    for key in ("train_output_path", "compile_output_path"):
        if cfg.pop(key, None) is not None:
            print(f"REMOVED root-level {key}")
    cfg.setdefault("training", {})["train_output_path"] = os.path.join(run_dir, "training")
    cfg.setdefault("compilation", {})["compile_output_path"] = os.path.join(run_dir, "compilation")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print("RUN_DIR=" + run_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
