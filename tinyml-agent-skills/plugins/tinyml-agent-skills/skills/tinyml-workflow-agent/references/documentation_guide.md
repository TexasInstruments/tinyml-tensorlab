# Documentation Navigation Guide

Agent reference for navigating `$TINYML_TENSORLAB_DOCS_PATH`. This guide maps all doc sections + scenarios to specific files. Use RST files (never HTML).

---

## SOURCE PRECEDENCE RULES

**CRITICAL:** If both RST docs and .md files in `/references` or `/assets` cover same topic:

1. **Check `/references/*.md` or `/assets/*.md` FIRST** — these take precedence (authoritative constants/procedures)
2. **RST is reference** — if question remains or for conceptual depth, consult RST
3. **If conflicting**, md file wins (it's maintained tighter than RST)

**Why:** .md files are task workflows (config_creation_guide.md, device_deployment_guide.md). RST is architectural background. For operational decisions, use .md.

### Precedence hierarchy (highest to lowest):
1. `/references/*.md` (task guides, procedures)
2. `/assets/*.md` (constants, presets, defaults)
3. RST docs (conceptual, background, examples)

---

## COMPLETE DOC MAP

### `/getting_started` — Onboarding
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Overview + toc | Starting fresh, need structure |
| `quickstart.rst` | 5-min hello world | Brand new user, minimal setup |
| `first_example.rst` | Full walkthrough (motor fault) | Want end-to-end example |
| `understanding_config.rst` | Config file fields + validation | Need field reference (but prefer `/references/config_creation_guide.md`) |
| `running_examples.rst` | How to execute bash command | Ready to run, need syntax |

**Precedence:** Use `/references/config_creation_guide.md` for config help (preferred), then `/references/example_running_guide.md` for execution.

---

### `/task_types` — Task-Specific Config
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Task selector matrix | User unsure which task type fits their data |
| `timeseries_classification.rst` | TS classification: models, default params, data format | Building TS classification config |
| `timeseries_regression.rst` | TS regression: models, params, multi-output support | Building TS regression config |
| `timeseries_forecasting.rst` | TS forecasting: horizon, lookback, models, params | Building TS forecasting config |
| `anomaly_detection.rst` | Anomaly: Normal/Anomaly data split, models, params | Building anomaly config |
| `image_classification.rst` | Vision: image size, normalization, models | Building image classification config |

**Precedence:** Consult `/assets/timeseries_default_params.md` + `/assets/vision_default_params.md` for exact defaults (authoritative). RST for conceptual explanation + model examples.

---

### `/byod` — Custom Data Formats
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | BYOD overview + metadata.json schema | Understanding custom data workflow |
| `classification_format.rst` | Folder structure: `classes/class_1/`, `classes/class_2/`, metadata | Preparing classification data |
| `regression_format.rst` | Files layout: `files/`, annotations.json, metadata | Preparing regression data |
| `forecasting_format.rst` | Forecasting layout + horizon + lookback rules | Preparing forecasting data |
| `anomaly_detection_format.rst` | Normal/Anomaly folders + metadata | Preparing anomaly data |
| `data_splitting.rst` | Train/val/test split strategies + config options | Understanding split behavior |

**Precedence:** RST is authoritative here (no .md equivalent). Cross-check `/task_types/{task_type}.rst` for task-specific requirements.

---

### `/byom` — Pre-Trained Models (No Training)
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | BYOM workflow overview | User has ONNX model, wants to skip training |
| `adding_models.rst` | ONNX import rules + supported ops | Integrating custom ONNX |
| `compilation_only.rst` | Config: training.enable: false, dataset.enable: false | Ready to compile pre-trained model |

**Precedence:** RST is authoritative. Reference `/references/config_creation_guide.md` § "BYOM Workflow" for config pattern.

---

### `/devices` — Hardware + Targets
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Device selector + family overview | Unsure which device family to use |
| `device_overview.rst` | All devices: SRAM, flash, speed specs | Checking memory constraints |
| `c2000_family.rst` | C2000 (F28P55, F2837, F2837, F29H85, F28P65): memory, compiler, SDKs | Targeting C2000 device |
| `mspm0_family.rst` | MSPM0 (G3507, G5187, G3517) + MSPM33 (C32, C34): specs, SDKs | Targeting MSPM0/MSPM33 device |
| `connectivity_devices.rst` | SimpleLink (CC2755, CC1352, CC1354, CC35X1), Sitara (AM13E2, AM263, AM263P, AM261) | Targeting connectivity or Sitara |
| `npu_guidelines.rst` | NPU devices (F28P55, AM13E2, MSPM0G5187): quantization modes, speed gain (10-25×), compile presets, memory tricks | NPU device specifics, quantization req, speed expectations |

**Precedence:** `/references/device_deployment_guide.md` for deployment steps. RST for device specs. Check `/references/installation_env_variables_guide.md` for compiler env vars.

---

### `/features` — Advanced Model & Data Engineering
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Features overview (NAS, quantization, FE, NAS, post-training) | Understanding advanced features |
| **Quantization** |
| `quantization.rst` | Quantization theory: why (size, speed, NPU req), config params (mode 0/1/2, method PTQ/QAT, schemes) | Understanding quantization config |
| `quantization_examples.rst` | Example configs + results per device | Seeing quantization impact |
| **Feature Extraction** |
| `feature_extraction.rst` | FE theory: data transforms (SimpleWindow, DownSample, Normalize, AddNoise, etc.), FE presets naming convention | Understanding FE + transform list |
| **Neural Architecture Search** |
| `neural_architecture_search.rst` | NAS: automatic architecture discovery, preset sizes (s/m/l/xl/xxl), optimization modes | User wants auto model design |
| **Post-Training Analysis** |
| `goodness_of_fit.rst` | Model accuracy analysis: metrics, confusion matrix, per-class performance | Evaluating trained model |
| `post_training_analysis.rst` | Quantization error analysis, layer-wise statistics | Debugging quantization issues |

**Precedence:** `/assets/timeseries_module_constants.md` + `/assets/vision_module_constants.md` list ALL presets (authoritative — use for lookups). `/assets/timeseries_data_proc_feat_ext_consts.md` lists all transforms + defaults. RST is conceptual. For specific preset name or transform behavior, check assets first.

---

### `/deployment` — Hardware Flashing + Integration
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Deployment workflow steps (A→D) | Starting deployment phase |
| `npu_device_deployment.rst` | CCS setup for NPU devices (F28P55, AM13E2, MSPM0G5187) + Resource Explorer + flashing | Deploying to NPU device |
| `non_npu_deployment.rst` | CCS setup for non-NPU (CPU-only) devices | Deploying to non-NPU device |
| `ccs_integration.rst` | Code Composer Studio integration details | CCS-specific configuration |

**Precedence:** `/references/device_deployment_guide.md` is scripted step-by-step (preferred for actual execution). RST is reference/theory. Use `/references/device_deployment_guide.md` for Steps A→D, RST for background.

---

### `/examples` — Reference Implementations (26+ tasks)
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | All examples directory | Finding example by domain |
| **Pre-Built Tasks** (have data + config) |
| `arc_fault.rst` | Arc fault detection (electrical current) | Reference electrical fault detection |
| `motor_bearing_fault.rst` | Motor bearing fault (vibration) | Reference bearing fault detection |
| `ecg_classification.rst` | ECG classification (heartbeats) | Reference ECG/biomedical signal |
| `electrical_fault.rst` | Electrical fault detection | Reference electrical domain |
| `fan_blade_fault_classification.rst` | Fan blade fault | Reference fan/mechanical fault |
| `forecasting_pmsm_rotor.rst` | PMSM rotor speed forecast | Reference forecasting task |
| `gas_sensor.rst` | Gas sensor anomaly detection | Reference gas/sensor domain |
| `grid_stability.rst` | Grid stability classification | Reference power/grid domain |
| `hvac_indoor_temp_forecast.rst` | HVAC temperature forecast | Reference HVAC/building domain |
| `blower_imbalance.rst` | Blower imbalance detection | Reference imbalance/mechanical |
| `pir_detection.rst` | PIR motion detection | Reference motion/binary classification |
| `induction_motor_speed_prediction.rst` | Induction motor speed (regression) | Reference motor control domain |
| `nilm_classification.rst` | NILM (Non-Intrusive Load Monitoring) | Reference NILM/energy domain |
| `washing_machine_regression.rst` | Washing machine state regression | Reference appliance domain |
| **Generic Tasks** (no data, config template) |
| `generic_classification.rst` | Generic TS classification template | Building custom TS classification |
| `generic_regression.rst` | Generic TS regression template | Building custom TS regression |
| `generic_forecasting.rst` | Generic TS forecasting template | Building custom TS forecasting |
| `generic_anomaly_detection.rst` | Generic anomaly detection template | Building custom anomaly task |
| `image_classification_example.rst` | Generic image classification template | Building custom vision task |
| `har_activity_recognition.rst` | HAR (Human Activity Recognition) | Reference HAR domain |
| `mosfet_temp_prediction.rst` | MOSFET temperature regression | Reference semiconductor domain |
| `torque_measurement_regression.rst` | Torque measurement regression | Reference torque/mechanical |
| `mnist_image_classification.rst` | MNIST example (vision) | Reference vision task |
| `grid_fault_detection.rst` | Grid fault detection | Reference grid/power domain |
| `anomaly_detection_example.rst` | Generic anomaly example walkthrough | Understanding anomaly workflow |
| `forecasting_example.rst` | Generic forecasting example walkthrough | Understanding forecasting workflow |
| `ac_arc_fault.rst` | AC arc fault (electrical) | Reference AC electrical fault |

**Use when:** User wants reference implementation, data source, or example config for specific domain. Docs include download links for data.

---

### `/introduction` — Conceptual Background
| File | Content | Use When |
|------|---------|----------|
| `what_is_tensorlab.rst` | High-level Tiny ML Tensorlab overview | New user needs context |
| `architecture.rst` | System architecture: ModelMaker → ModelZoo → Deployment | Understanding system design |
| `terminology.rst` | Glossary: NPU, quantization, compile, feature extraction, etc. | Confused by term |

**Use when:** Onboarding or architectural questions. Not needed for operational tasks.

---

### `/installation` — Setup + Environment
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Installation overview | Starting setup |
| `prerequisites.rst` | System requirements (Python, pip, git) | Checking if system ready |
| `linux_setup.rst` | Linux installation steps (venv, pip install) | Installing on Linux |
| `windows_setup.rst` | Windows installation steps | Installing on Windows |
| `developer_installation.rst` | Dev mode (git clone, editable install) | Contributing to tensorlab |
| `environment_variables.rst` | Compiler env vars by device family | Setting up build environment |

**Precedence:** `/references/setup_guide.md` + `/references/installation_env_variables_guide.md` are operational steps (preferred). RST is reference.

---

### `/appendix` — Reference Tables
| File | Content | Use When |
|------|---------|----------|
| `config_reference.rst` | Full YAML config schema + defaults | Validating config |
| `model_zoo_reference.rst` | All models by task type + architecture details | Choosing model |
| `changelog.rst` | Version history + breaking changes | Upgrading versions |

**Precedence:** `/references/config_creation_guide.md` is better for config help. RST appendix for deep reference.

---

### `/troubleshooting` — Error Diagnosis
| File | Content | Use When |
|------|---------|----------|
| `index.rst` | Troubleshooting overview | Starting diagnosis |
| `common_errors.rst` | Error messages + root causes + fixes | Hit error during training/compilation/deploy |
| `faq.rst` | Frequently asked questions | Common setup/usage questions |

**Use when:** User hits error or has setup question. Cross-reference device-specific sections if needed.

---

## REFERENCE FILES (Higher Priority)

Always check these BEFORE RST docs:

| File | Topic | Use When |
|------|-------|----------|
| `/references/config_creation_guide.md` | Config creation + validation | **User creating config** — preferred over RST |
| `/references/device_deployment_guide.md` | Hardware deployment steps A→D | **User deploying to device** — preferred over RST |
| `/references/example_running_guide.md` | How to run examples | **User running config** — command syntax |
| `/references/setup_guide.md` | Python env + venv + installation | **User setting up repo** — preferred over RST |
| `/references/installation_env_variables_guide.md` | Compiler env vars per device | **User setting up compiler** — preferred over RST |
| `/assets/timeseries_module_constants.md` | ALL TS feature extraction presets | **User choosing FE preset** — authoritative lookup |
| `/assets/vision_module_constants.md` | ALL vision feature extraction presets | **User choosing FE preset (vision)** — authoritative lookup |
| `/assets/timeseries_default_params.md` | Task-specific default params (TS) | **User checking defaults** — authoritative |
| `/assets/vision_default_params.md` | Task-specific default params (Vision) | **User checking defaults** — authoritative |
| `/assets/timeseries_data_proc_feat_ext_consts.md` | All data transforms + feature extraction | **User choosing transform** — authoritative |
| `/assets/vision_data_proc_feat_ext_consts.md` | Vision-specific transforms | **User choosing transform (vision)** — authoritative |

---

## Scenario → Doc Path Mapping

---

## Scenario → Doc Path Mapping

### Scenario 1: User is brand new

**"I've never used Tiny ML before"**

1. Check: `/references/setup_guide.md` (environment setup) — required first
2. Then: `/getting_started/quickstart.rst` (5 min hello world)
3. Then: `/getting_started/first_example.rst` (full walkthrough)
4. Decide task type: `/task_types/index.rst`
5. If custom data: `/byod/index.rst`

---

### Scenario 2: Creating config file

**"Help me set up a motor fault detection config"**

**PRECEDENCE:**
1. **PRIMARY:** `/references/config_creation_guide.md` (complete config guide + validation)
2. Task specifics: `/task_types/timeseries_classification.rst` (models, details)
3. Data format: `/byod/classification_format.rst`
4. Defaults: `/assets/timeseries_default_params.md` (authoritative task defaults)
5. Device: `/devices/npu_guidelines.rst` or `/devices/c2000_family.rst` (quantization, memory)

---

### Scenario 3: User has custom dataset

**"I have sensor data for anomaly detection. How do I format it?"**

**PRECEDENCE:**
1. **PRIMARY:** `/byod/anomaly_detection_format.rst` (folder structure)
2. Metadata schema: `/byod/index.rst`
3. Split options: `/byod/data_splitting.rst`
4. Task defaults: `/assets/timeseries_default_params.md` (anomaly detection params)

---

### Scenario 4: User is stuck on data format

**"My classification data isn't loading. What's wrong?"**

1. Check: `/byod/classification_format.rst` (exact layout: `classes/class_1/`, `classes/class_2/`, metadata.json)
2. Schema: `/byod/index.rst` (metadata.json structure)
3. Error: `/troubleshooting/common_errors.rst` (data loading errors)

---

### Scenario 5: Choosing a task type

**"I have electrical current data. Should I use classification or regression?"**

1. Decision matrix: `/task_types/index.rst` (by data type + expected output)
2. Classification details: `/task_types/timeseries_classification.rst`
3. Regression details: `/task_types/timeseries_regression.rst`
4. Forecasting details: `/task_types/timeseries_forecasting.rst`

---

### Scenario 6: Device constraints & memory

**"Will my model fit on MSPM0G3507? It's running out of memory."**

1. Device specs: `/devices/mspm0_family.rst` (MSPM0G3507 = 40KB SRAM)
2. NPU tricks: `/devices/npu_guidelines.rst` (compression, NPU speedup)
3. Troubleshoot: `/references/device_deployment_guide.md` § "Device out of memory"

---

### Scenario 7: NPU vs CPU device

**"I'm targeting F28P55. Does it have NPU? What's the speedup?"**

1. **PRIMARY:** `/devices/npu_guidelines.rst` (NPU devices: F28P55 = has NPU, 10-25× speedup, quantization: 2 required)
2. Reference: `/devices/c2000_family.rst` (non-NPU C2000 variants for comparison)

---

### Scenario 8: Deployment to hardware

**"Training done. How do I flash to device?"**

1. **PRIMARY:** `/references/device_deployment_guide.md` (Steps A→D scripted procedures)
2. Background: `/deployment/index.rst` (workflow overview)
3. Device-specific RST:
   - F28P55 (NPU): `/deployment/npu_device_deployment.rst`
   - MSPM0 (non-NPU): `/deployment/non_npu_deployment.rst`
4. Errors: `/troubleshooting/common_errors.rst`

---

### Scenario 9: Pre-trained ONNX model (skip training)

**"I have ONNX. Compile + deploy only."**

1. Workflow: `/byom/index.rst` (BYOM overview)
2. ONNX rules: `/byom/adding_models.rst` (supported ops, constraints)
3. Config pattern: `/byom/compilation_only.rst` (training.enable: false, dataset.enable: false)
4. **ALSO:** `/references/config_creation_guide.md` § "BYOM Workflow"

---

### Scenario 10: Feature extraction + transforms

**"What's `Generic_1024Input_FFTBIN_64Feature_8Frame`? How to choose?"**

**PRECEDENCE:**
1. **PRIMARY:** `/assets/timeseries_module_constants.md` (all preset names + exact params — authoritative lookup)
2. **PRIMARY:** `/assets/timeseries_data_proc_feat_ext_consts.md` (all transforms + behaviors)
3. Background: `/features/feature_extraction.rst` (theory + how transforms work)
4. Task-specific: `/task_types/{task_type}.rst` (recommended presets per task)

---

### Scenario 11: Quantization config

**"When do I use quantization: 0 vs 1 vs 2? What's PTQ vs QAT?"**

**PRECEDENCE:**
1. Config rules: `/references/config_creation_guide.md` § "Compilation Section" (quantization field options)
2. **PRIMARY:** `/devices/npu_guidelines.rst` (device → quantization requirement: C2000 non-NPU = 1, F28P55 NPU = 2)
3. Theory: `/features/quantization.rst` (why + how quantization works)
4. Examples: `/features/quantization_examples.rst` (config examples + results)

---

### Scenario 12: Data transforms (SimpleWindow, DownSample, Normalize, etc.)

**"Which transforms should I use? What do they do?"**

**PRECEDENCE:**
1. **PRIMARY:** `/assets/timeseries_data_proc_feat_ext_consts.md` (all transforms + exact params)
2. Explanation: `/features/feature_extraction.rst` (transform descriptions + use cases)
3. Task defaults: `/assets/timeseries_default_params.md` (recommended transforms per task)

---

### Scenario 13: Compilation presets

**"What's `default_preset` vs `compress_npu_layer_data` vs `forced_soft_npu_preset`?"**

**PRECEDENCE:**
1. Config reference: `/references/config_creation_guide.md` § "Compilation Section" (preset table)
2. Device context: `/devices/npu_guidelines.rst` (when to use which preset)
3. RST background: `/features/` (optimization strategies)

---

### Scenario 14: Debugging compilation error

**"Compilation failed: 'Undefined symbol mod_inference'"**

1. **PRIMARY:** `/troubleshooting/common_errors.rst` (error message + fix)
2. Build setup: `/references/installation_env_variables_guide.md` (compiler env vars)
3. Device-specific: `/devices/{family}.rst` (if device-related)
4. Last resort: `/references/device_deployment_guide.md` § "Contingency Plan" (use CCS MCP servers)

---

### Scenario 15: Post-training analysis

**"How accurate is my model? How do I evaluate it?"**

1. Metrics: `/features/goodness_of_fit.rst` (confusion matrix, per-class metrics)
2. Analysis: `/features/post_training_analysis.rst` (layer-wise quantization error, debugging)

---

### Scenario 16: Neural Architecture Search (NAS)

**"Can I auto-generate the best model?"**

→ `/features/neural_architecture_search.rst` (preset sizes: s/m/l/xl/xxl, optimization: memory or compute)

---

### Scenario 17: GUI model design

**"I want to design model visually instead of picking from modelzoo"**

1. Overview: `/model_composer/overview.rst` (drag-drop builder)
2. Quickstart: `/model_composer/getting_started_gui.rst` (how to use GUI)
3. Export: `/model_composer/exporting_models.rst` (ONNX output for deployment)

---

### Scenario 18: Reference implementation for domain

**"I want to see a working example for motor fault detection"**

→ `/examples/motor_bearing_fault.rst` (includes data link, config example, expected results)

Then check other task-specific examples:
- Arc fault: `/examples/arc_fault.rst`
- ECG: `/examples/ecg_classification.rst`
- HVAC forecast: `/examples/hvac_indoor_temp_forecast.rst`
- Anomaly: `/examples/anomaly_detection_example.rst`
- etc.

---

### Scenario 19: Running a config file

**"How do I execute the config?"**

**PRECEDENCE:**
1. **PRIMARY:** `/references/example_running_guide.md` (bash command syntax + pre-run checks)
2. Background: `/getting_started/running_examples.rst`

---

### Scenario 20: Environment setup (compiler paths, venv)

**"I need to set up my environment before running examples"**

**PRECEDENCE:**
1. **PRIMARY:** `/references/setup_guide.md` (venv creation, repo setup)
2. **PRIMARY:** `/references/installation_env_variables_guide.md` (compiler env vars by device)
3. RST reference: `/installation/` (detailed setup for Linux/Windows)

---

## FOR AGENTS: Triage Decision Tree

```
User asks about...
├─ Config creation? → /references/config_creation_guide.md (PRIMARY)
├─ Data format? → /byod/{task_type}_format.rst
├─ Feature extraction / transforms? → /assets/timeseries_module_constants.md (PRIMARY)
├─ Quantization? → /references/config_creation_guide.md (PRIMARY) + /devices/npu_guidelines.rst
├─ Device specs / memory? → /devices/{family}.rst
├─ NPU specifics? → /devices/npu_guidelines.rst (PRIMARY)
├─ Deployment / flashing? → /references/device_deployment_guide.md (PRIMARY)
├─ Error / debugging? → /troubleshooting/common_errors.rst (PRIMARY)
├─ Environment setup? → /references/setup_guide.md (PRIMARY) + /references/installation_env_variables_guide.md
├─ Running example? → /references/example_running_guide.md (PRIMARY)
├─ Model evaluation? → /features/goodness_of_fit.rst
├─ NAS / model design? → /features/neural_architecture_search.rst or /model_composer/
├─ New user onboarding? → /getting_started/ + /references/setup_guide.md
├─ Task type selection? → /task_types/index.rst
├─ Reference implementation? → /examples/{domain}.rst
└─ General? → /introduction/ (architecture, terminology)
```

When RST + .md both exist for topic: **Check .md first** (operational), then RST (background).

---

## Maintenance Notes

- **RST files:** `$TINYML_TENSORLAB_DOCS_PATH` — authoritative for conceptual content
- **Constants/procedures:** `/references/*.md` + `/assets/*.md` — **authoritative for lookups + operations**
- **Precedence:** Always check .md files first for operational decisions
- **No HTML:** Use RST files only, never HTML output
- **This guide:** Navigation layer for agents — not doc duplication
