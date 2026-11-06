#!/usr/bin/env python3
"""Create a brand-new run folder, then launch training in it. One call, no skipping.

Usage: run_training.py <PROJECT_PATH> <TRAIN_DIR> -- <training command...>

Does new_run_dir.py's work (fresh run/<yyyymmdd-hhmmss>/<model_name>/ folder, config.yaml
paths rewritten), prints RUN_DIR=<path>, then runs the training command with TRAIN_DIR
as the working directory and the caller's environment (e.g. C2000_CGT_ROOT). Exits with
the command's exit code. Because the folder is created in the same call that starts
training, a re-run can never reuse an earlier run's paths.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import new_run_dir


def main(argv):
    if len(argv) < 5 or argv[3] != "--":
        sys.exit(__doc__)
    root, train_dir, cmd = argv[1], argv[2], argv[4:]
    new_run_dir.main(root)  # prints RUN_DIR=...
    sys.stdout.flush()
    os.makedirs(train_dir, exist_ok=True)
    sys.exit(subprocess.run(cmd, cwd=train_dir).returncode)


if __name__ == "__main__":
    main(sys.argv)
