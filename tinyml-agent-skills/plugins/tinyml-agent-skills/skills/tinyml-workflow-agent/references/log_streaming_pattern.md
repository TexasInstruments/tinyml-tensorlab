# Log Streaming During Training

Pattern for showing live training progress while example runs in background.

---

## Log File Locations

### Training Phase

```
{TINYML_BASE_PATH}/tinyml-modelmaker/data/projects/{TASK_TYPE}/run/{RUN_ID}/{MODEL_NAME}/training/base/run.log
```

**Example:**
```
~/tinyml-tensorlab/tinyml-modelmaker/data/projects/motor_fault/run/20240519_143022/ds_cnn/training/base/run.log
```

Created at training start. First entries after 1-2 min (data load phase).

### Compilation Phase

```
{TINYML_BASE_PATH}/tinyml-modelmaker/data/projects/{TASK_TYPE}/run/{RUN_ID}/{MODEL_NAME}/compilation/base/run.log
```

**Example:**
```
~/tinyml-tensorlab/tinyml-modelmaker/data/projects/motor_fault/run/20240519_143022/ds_cnn/compilation/base/run.log
```

Created when compilation phase starts (after training done). Use same `/loop` pattern to monitor.

---

## Quick Pattern

```bash
# Terminal 1: Start training
bash /path/to/run_tinyml_modelzoo.sh config.yaml > /tmp/start.log 2>&1 &
TRAIN_PID=$!

# Extract RUN_ID + MODEL_NAME from config (or parse output)
# Then tail the log
tail -f ~/tinyml-tensorlab/tinyml-modelmaker/data/projects/TASK_TYPE/run/RUN_ID/MODEL_NAME/training/base/run.log

# Terminal 2: Monitor process
while kill -0 $TRAIN_PID 2>/dev/null; do
  sleep 30
done
echo "Training finished"
```

---

## In Claude Code (Recommended)

### Step 1: Start training, capture metadata

Run config. Output logs show:
```
[INFO] RUN_ID: 20240519_143022
[INFO] MODEL_NAME: ds_cnn
[INFO] TASK_TYPE: motor_fault
```

Agent extracts `{RUN_ID}`, `{MODEL_NAME}`, `{TASK_TYPE}`.

### Step 2: Construct log path

```python
LOG_PATH = f"{TINYML_BASE_PATH}/tinyml-modelmaker/data/projects/{TASK_TYPE}/run/{RUN_ID}/{MODEL_NAME}/training/base/run.log"
```

### Step 3: Use `/loop 45s` to tail log

Prompt:
```
Monitor training:
- LOG_PATH: {path}
- Check if exists: test -f {LOG_PATH}
  - If not: say "Log not created yet, retrying..."
  - If yes: tail -30 {LOG_PATH}
- Check process: kill -0 {TRAIN_PID}
  - If running: loop continues
  - If done: show "Training complete" + final 50 lines
```

Result: User sees progress every 45s. Loop exits when training done.

---

## What User Sees

**Initial:**
```
Training started (PID: 12345)
Log path: ~/tinyml-tensorlab/tinyml-modelmaker/data/projects/motor_fault/run/20240519_143022/ds_cnn/training/base/run.log
Polling every 45s...
```

**Update 1 (45s):**
```
=== Training Progress ===
[INFO] Phase 1: Data loading...
[INFO] Loaded 150 training samples
[INFO] Phase 2: Training model...
[EPOCH 1/100] Loss: 0.8234, Acc: 0.71
```

**Update 2 (90s):**
```
[EPOCH 10/100] Loss: 0.4123, Acc: 0.92
[EPOCH 11/100] Loss: 0.3987, Acc: 0.94
```

**When done:**
```
Training complete.
=== Final Output ===
[SUCCESS] Training completed!
Model saved to: ~/tinyml-tensorlab/tinyml-modelmaker/data/projects/motor_fault/run/20240519_143022/ds_cnn/training/model.pt
[INFO] Phase 3: Compilation...
[SUCCESS] Compilation complete!
```

---

## Single Process, Sequential Phases

Training + compilation run as single Python process. Same PID stays alive throughout.

**Flow:**
1. PID starts → `run_tinyml_modelmaker.py` begins
2. Phase 1 (training): logs to `/training/base/run.log`
3. Phase 1 ends → same PID continues
4. Phase 2 (compilation): logs to `/compilation/base/run.log`
5. Phase 2 ends → PID dies

**Monitoring:**
- Watch same PID for full lifecycle
- Monitor training log initially
- Detect phase switch: when training log stops updating + compilation log exists
- Switch to compilation log
- Exit when PID dies

**Detection marker:**
```bash
grep -q "\[SUCCESS\].*[Tt]raining\|[Cc]ompleted" {TRAINING_LOG}
test -f {COMPILATION_LOG}
# Both true → switch phase
```

**Loop logic:**
```
While PID alive:
  if training log has completion marker AND compilation log exists:
    current_log = compilation_log
    phase = "Compilation"
  else:
    current_log = training_log
    phase = "Training"
  
  tail current_log
  sleep 45s

When PID dies:
  Show final output from current_log (training or compilation)
```

---

## Agent Implementation

### 1. Extract metadata from training start

Parse stdout/stderr for:
```
RUN_ID = grep -o "run/[^/]*" output | cut -d/ -f2
MODEL_NAME = grep -o "model.*" output | head -1
TASK_TYPE = from config (or extract from log path pattern)
```

### 2. Construct log path

```bash
LOG_PATH="${TINYML_BASE_PATH}/tinyml-modelmaker/data/projects/${TASK_TYPE}/run/${RUN_ID}/${MODEL_NAME}/training/base/run.log"
```

### 3. Start loop (auto-detects phase)

Use `/loop 45s` with prompt template:

```
Check process: kill -0 {TRAIN_PID} 2>/dev/null
- Dead? → Show "Process complete" + final 50 lines, exit loop

Detect phase:
- test -f {TRAINING_LOG} → log_path = training_log, phase = "Training"
- grep success in {TRAINING_LOG} AND test -f {COMPILATION_LOG} → 
    log_path = compilation_log, phase = "Compilation"

Check log:
- test -f {log_path}? 
  - No: "Waiting for {phase} log..."
  - Yes: tail -30 {log_path}

Continue loop while PID alive
```

### 4. Loop auto-exits when done

No manual stop. PID dies → loop detects → shows final output → done. Handles phase switch automatically during loop.

---

## Edge Cases

### Log not created yet (first 1-2 min)
```bash
if [ ! -f "$LOG_PATH" ]; then
  echo "Waiting for log to be created (data loading phase)..."
  # Loop retries
fi
```

### Process hung (log static for N loops)
- Track last line count
- If unchanged 3 loops running → suggest checking process manually
- User can `ps aux | grep {PID}` or inspect full log

### Large log (after many epochs)
- Tail only last 30-40 lines per update
- Avoids context bloat in conversation

---

## Implementation Checklist

- [ ] Extract `RUN_ID`, `MODEL_NAME`, `TASK_TYPE` from training start output
- [ ] Construct log path: `{TINYML_BASE_PATH}/tinyml-modelmaker/data/projects/{TASK_TYPE}/run/{RUN_ID}/{MODEL_NAME}/training/base/run.log`
- [ ] Start training in background (capture PID)
- [ ] Use `/loop 45s` to poll log + check process status
- [ ] Loop checks `kill -0 $PID` to detect completion
- [ ] On loop exit: show final 50-100 lines of log
- [ ] Handle "log not created yet" case gracefully

---

## Alternative: Manual Log Inspection

If loop doesn't work or user prefers manual:

```bash
# User runs in terminal while training runs
tail -f ~/tinyml-tensorlab/tinyml-modelmaker/data/projects/TASK_TYPE/run/RUN_ID/MODEL_NAME/training/base/run.log
```

Works but requires user to manage terminal. Loop pattern is better UX.
