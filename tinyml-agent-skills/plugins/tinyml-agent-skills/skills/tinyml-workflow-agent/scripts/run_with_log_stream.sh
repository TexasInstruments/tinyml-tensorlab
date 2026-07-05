#!/bin/bash
# Run Tiny ML example with streaming log output
# Usage: ./run_with_log_stream.sh <config_path> [log_file]

set -e

CONFIG_PATH="$1"
LOG_FILE="${2:-run.log}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config_path> [log_file]"
  echo "Example: $0 /path/to/config.yaml"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found at $CONFIG_PATH"
  exit 1
fi

# Extract base directory from config path
CONFIG_DIR=$(dirname "$CONFIG_PATH")
cd "$CONFIG_DIR"

# Start training in background, log to file
echo "[$(date)] Starting training from $CONFIG_PATH" > "$LOG_FILE"
echo "Logging to: $LOG_FILE" >&2

# Extract TINYML_BASE_PATH from config or environment
TINYML_BASE_PATH="${TINYML_BASE_PATH:-.}"

bash "$TINYML_BASE_PATH/tinyml-modelzoo/run_tinyml_modelzoo.sh" "$CONFIG_PATH" >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "Training started (PID: $TRAIN_PID)"
echo "::TRAIN_PID::$TRAIN_PID" >&2
echo "::LOG_FILE::$LOG_FILE" >&2

# Wait for process
wait $TRAIN_PID
TRAIN_EXIT=$?

echo "[$(date)] Training complete (exit: $TRAIN_EXIT)"
exit $TRAIN_EXIT
