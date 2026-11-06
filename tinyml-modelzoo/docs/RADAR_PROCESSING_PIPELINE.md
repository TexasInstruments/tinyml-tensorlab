# Radar Data Pre-Processing Pipeline — Complete Reference Guide

> **Complete guide to the Radar Data Pre-Processing Pipeline in Tensorlab**

**Important Note : This pipeline ONLY works on Processed Signal Chain Output and is mainly having inbuilt presets needed to preprocess Radar dataset through filterings, points/radar frames quality checks, windowing etc. and make it ready for Training**

---

## Table of Contents

1. [Identify Your Use Case Scenario](#identify-your-use-case-scenario)
2. [Converting mmWave Data Recorder Captures](#converting-mmwave-data-recorder-captures)
3. [Feature Extraction Presets](#feature-extraction-presets)
4. [Inline Override: Customizing Presets in YAML](#inline-override-customizing-presets-in-yaml)
5. [Use Case : Pre-Cleaned / Ready To Train Datasets](#use-case--pre-cleaned--ready-to-train-datasets)
6. [All Configurable Parameters Reference](#all-configurable-parameters-reference)

---

## Identify Your Use Case Scenario

| Your Data Situation | Recommended Path |
|---------------------|------------------|
| **Already have per-frame features** (CSV columns = model input features, no bin*, point_*, track_* columns) | Use **Generic_Radar_SimpleWindowOnly** preset - add inline args - frame_size, stride_size for windowing customisation |
| **On-device Signal Chain Output from mmWave Data Recorder** (columns MUST match expected names: bin*, point_x*, track_velocity_*, etc.) | Use Generic_248Input_Posture_and_Fall_Detection or Generic_392Input_PointCloudWithTrack_SimpleWindow or Generic_130Input_Radar_SurfaceClassification Presets and override params |


---

## Converting mmWave Data Recorder Captures

If your data comes directly from the mmWave Data Recorder, use the `annotate_radar_dataset.py` script to convert it into a Tensorlab-ready dataset ZIP.

**Input Structure Needed** (mmWave Data Recorder output):
>**Important Modification needed in binData Folder from User's end** : To develop 'class label' named folders (eg. falling, walking etc.) and classify the session folders in binData accordingly.
```
binData/
├── falling/
│   ├── session_001/
│   │   ├── replay_*.csv
│   │   └── replay_*.json
│   └── ...
├── sitting/
├── standing/
├── walking/
└── laying/
```

**Command**:
```bash
python examples/radar_posture_and_fall_detection/annotate_radar_dataset.py \
  --input /path/to/binData \
  --output /path/to/my_radar_dataset.zip
```

**Output Structure** (ready for Tensorlab):
```
my_radar_dataset.zip
└── classes/
    ├── falling/
    │   ├── session_001.csv
    │   └── ...
    ├── sitting/
    ├── standing/
    ├── walking/
    └── laying/
```

Then update `config.yaml`:
```yaml
dataset:
  input_data_path: /path/to/my_radar_dataset.zip
```

Refer to the [Radar Posture & Fall Detection README](https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/radar_posture_and_fall_detection/README.md) for detailed instructions specific to the application.

---

## Feature Extraction Presets

### Preset 1: Generic_248Input_Posture_and_Fall_Detection 
**Use Case**: Optimized for human posture and fall detection (matches RF_DualStream_PointCloud_19K).

```python
Generic_248Input_Posture_and_Fall_Detection = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window'],
        frame_size=8,
        stride_size=0.125,
        radar_point_cloud_data_selector=['y', 'z','snr'],
        radar_point_cloud_coordinate_system='relative',
        radar_point_cloud_relative_coordinate_filter=['y'],
        radar_point_cloud_spatial_filter_bounds=[None, None, -10.0, 10.0, -10.0, 10.0],
        radar_track_data_selector=['z', 'velocity', 'acceleration'],
        radar_track_velocity_bounds=[-10.0, 10.0],
        radar_track_acceleration_bounds=[-10.0, 10.0],
        radar_track_spatial_filter_bounds=[None, None, None, None, -10.0, 10.0],
        radar_max_points_per_frame=8,
        radar_min_points_per_window=12,
        radar_point_cloud_structure_strategy='highest and lowest',
        radar_point_cloud_sort_by='z',
        radar_frames_quality_index='per_window',
        radar_feature_ordering='frame_major',
    ),
    common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION),
)
```

**Per-Frame Feature Layout (31 features)**:


| Index | Feature | Source |
|-------|---------|--------|
| 0 | track_height_z | Track |
| 1 | track_velocity_x | Track |
| 2 | track_velocity_y | Track |
| 3 | track_velocity_z | Track |
| 4 | track_acc_x | Track |
| 5 | track_acc_y | Track |
| 6 | track_acc_z | Track |
| 7-30 | 8 points x [y, z, snr] | Point Cloud (24 features) |

**Total per frame**: 7 (track) + 24 (8 points x 3) = **31 features**  
**Window (8 frames)**: 31 x 8 = **248 features** → Matches model input!

---

### Preset 2: Generic_130Input_Radar_SurfaceClassification
**Use Case**: Surface Classification using FFT range bins only.

```python
Generic_130Input_Radar_SurfaceClassification = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Radar Range Data', 'Simple Window'],
        radar_range_bins_bounds=[6, 31],
        frame_size=5,
        stride_size=0.2,
        radar_frames_quality_index=None,
        variables=1,
    ),
    common=dict(task_type=TASK_TYPE_RADAR_SURFACE_CLASSIFICATION),
)
```
**Result**: 26 bins (6-31 inclusive) x 5 frames = 130 features

---

### Preset 3: Generic_Radar_SimpleWindowOnly
**Use Case**: Pre-cleaned/refined datasets where CSV already has correct per-frame features. Only windowing is needed.

```python
Generic_Radar_SimpleWindowOnly = dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Simple Window'],
        frame_size=8,
        stride_size=0.125,
    ),
    common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION),
)
```
**Notes**:
- No range bin, point cloud, track, or scaling stages run
- Input CSV must already have the correct per-frame feature columns

---

### Preset 4: Generic_392Input_PointCloudWithTrack_SimpleWindow
**Use Case**: Complete CSV processing with full feature extraction and quality filtering.  
**Resulting Input Tensor**: `(N, 8 x 49) = (N, 392)` — 5 PC features x 8 points + 9 track features = 49/frame.

```python
Generic_392Input_PointCloudWithTrack_SimpleWindow=dict(
    data_processing_feature_extraction=dict(
        data_proc_transforms=['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window'],
        frame_size=8,
        stride_size=0.125,
        radar_point_cloud_data_selector=['x','y', 'z','doppler','snr'],
        radar_point_cloud_coordinate_system='relative',
        radar_point_cloud_relative_coordinate_filter=['x','y','z'],
        radar_point_cloud_spatial_filter_bounds=[None, None, None, None, None, None],
        radar_track_data_selector=['x', 'y', 'z', 'velocity', 'acceleration'],
        radar_track_velocity_bounds=[None, None],
        radar_track_acceleration_bounds=[None, None],
        radar_track_spatial_filter_bounds=[None, None, None, None, None, None],
        radar_max_points_per_frame=8,
        radar_min_points_per_window=12,
        radar_point_cloud_structure_strategy='highest only',
        radar_point_cloud_sort_by='z',
        radar_frames_quality_index='per_window',
        radar_feature_ordering='frame_major',
    ),
    common=dict(task_type=TASK_TYPE_RADAR_POINT_CLOUD_CLASSIFICATION), )
```

---

## Inline Override: Customizing Presets in YAML

> **Power User Feature**: Use a preset as base, then override specific parameters inline.

### How It Works
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  
  # Override ANY parameter from the preset:
  frame_size: 16                    # Change window size
  stride_size: 0.25                 # Change stride
  radar_min_points_per_window: 20   # Stricter quality
  radar_max_points_per_frame: 12    # More points per frame
  radar_point_cloud_data_selector: ['x', 'y', 'z', 'snr']  # Add x coordinate
```



### Override Example: Custom Fall Detection Config
```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
  
  # Custom: 16 frames, no overlap, stricter quality, all point features
  frame_size: 16
  stride_size: 1.0
  radar_min_points_per_window: 20
  radar_point_cloud_data_selector: ['x', 'y', 'z', 'doppler', 'snr']
  radar_point_cloud_sort_by: 'y'  
  
  # Keep rest from preset (relative coords, track features, etc.)
```
___

## Use Case : Pre-Cleaned / Ready To Train Datasets

> **Scenario**: You already have a dataset where each row = one frame with the exact features your model expects. No bin*, point_*, track_* columns exist. You just need **windowing** (sliding window with stride).

### When to Use
- Features already extracted externally (MATLAB, Python, custom scripts) with no pre-processing needed
- CSV columns = final per-frame features (e.g., 31 columns per frame)
- No radar-specific column names present
- You only need: frame_size, stride, no quality filtering support

### Use the Preset

```yaml
data_processing_feature_extraction:
  feature_extraction_name: "Generic_Radar_SimpleWindowOnly"
  # That's it! Only "Simple Window" stage runs.
```

### Inline Config (Full Control)

```yaml
data_processing_feature_extraction:
  data_proc_transforms:
    - "Simple Window"          # ONLY this stage runs
  frame_size: 8                # Frames per window (MUST match model)
  stride_size: 0.5             # Stride as fraction of frame_size (0-1)
  # Feature ordering in flattened window
  radar_feature_ordering: 'frame_major'  # or 'feature_major'
```

### Important Notes
| Parameter | Requirement |
|-----------|-------------|
| Column names | **Don't matter** — pipeline treats all columns as features |
| Quality filtering | Only supportive when column names present as per Tensorlab standards |

---

## All Configurable Parameters Reference

> **Note**: Using a preset via feature_extraction_name is highly recommended. Any parameter whose value matches the preset's default **does not need to be specified in your YAML**. Only override parameters that differ from the preset defaults.

### Data Processing Feature Extraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| feature_extraction_name | str | None | Preset Name (Eg. Generic_248Input_Posture_and_Fall_Detection for Posture and Fall Detection) |
| data_proc_transforms | list | [] | **Main pipeline gate** — tokens: "Radar Range Data", "Radar Point Cloud Data", "Radar Track Data", "Scaling", "Simple Window" |
| frame_size | int | 1 | **Frames per window (critical for model input shape)** |
| stride_size | float | 0.125 | Stride as fraction of frame_size (0-1) |



---

### 1. Range Data Parameters

Active only when "Radar Range Data" is in `data_proc_transforms` (e.g., surface classification).

| Parameter | Type | Default | Description | When to Use |
|-----------|------|---------|-------------|-------------|
| radar_range_bins_bounds | list[2] | [min_bin, max_bin] | Range bin selection to crop near/far bins. | **Required** if "Radar Range Data" in data_proc_transforms. Set to crop near/far bins. **Optional**: omit to keep all bins. |

---

### 2. Point Cloud Data Parameters

Active only when "Radar Point Cloud Data" is in `data_proc_transforms`.

| Parameter | Type | Default | Description | When to Use |
|-----------|------|---------|-------------|-------------|
| radar_point_cloud_data_selector | list[str] | [x, y, z, doppler, snr] | Point features to keep. Choose a subset to reduce dimensionality (e.g., y,z,snr for fall detection). | **Required** if "Radar Point Cloud Data" active. |
| radar_point_cloud_coordinate_system | str | absolute or relative | Coordinate frame. Use relative to make point cloud position-invariant. | **Required** if "Radar Point Cloud Data" in data_proc_transforms. |
| radar_point_cloud_relative_coordinate_filter | list[str] | [x, y, z] | Axes to make relative (e.g., y for distance-only). | **Required** when coordinate_system: relative. |
| radar_point_cloud_spatial_filter_bounds | list[6] | [min_x, max_x, min_y, max_y, min_z, max_z] | Spatial gate to drop points outside the sensing zone. | **Optional**. Use None to disable. |
| radar_point_cloud_non_spatial_filter_bounds | list[4] | [min_dop, max_dop, min_snr, max_snr] | Doppler/SNR gate to drop noisy or implausible points. | **Optional**. Use None to disable. |
| radar_max_points_per_frame | int | Max points per frame after selection | Cap on points kept per frame. | **Required** if "Radar Point Cloud Data" active. |
| radar_point_cloud_structure_strategy | str | Point selection strategy | Options: highest only, lowest only, highest and lowest. | **Required** if "Radar Point Cloud Data" active. |
| radar_point_cloud_sort_by | str | Feature to sort points by: x, y, z, doppler, snr | Determines sort order for the point selection strategy. | **Required** if "Radar Point Cloud Data" active. |

---

### 3. Track Data Parameters

Active only when "Radar Track Data" is in `data_proc_transforms`.

| Parameter | Type | Default | Description | When to Use |
|-----------|------|---------|-------------|-------------|
| radar_track_data_selector | list[str] | [x, y, z, velocity, acceleration] | Track features to keep. Select subset to control features/frame (e.g., z, velocity, acceleration). | **Required** if "Radar Track Data" active. |
| radar_track_velocity_bounds | list[2] | [min_vel, max_vel] | Filter out physically implausible track velocities (ghost tracks). Eg: [-10, 10]. | **Optional**. |
| radar_track_acceleration_bounds | list[2] | [min_acc, max_acc] | Filter out implausible accelerations. Eg: [-10, 10]. | **Optional**. |
| radar_track_spatial_filter_bounds | list[6] | [min_x, max_x, min_y, max_y, min_z, max_z] | Gate track position to the sensing zone. | **Optional**. Use None to disable. |
| radar_data_scaling_track_velocity | float | Scale factor for track_velocity_* columns | Independent of "Radar Track Data". Use when velocity magnitude is too small vs position features (e.g., 30). | **Optional**. You **NEED** to specify "Scaling" pre processing token in data_proc_transforms |

---

### 4. Simple Window Parameters

Active only when "Simple Window" is in `data_proc_transforms`.

| Parameter | Type | Default | Description | When to Use |
|-----------|------|---------|-------------|-------------|
| frame_size | int | 1 | Frames per window. Critical for model input shape. | Always when windowing. |
| stride_size | float | 0.125 | Stride as fraction of frame_size (0-1). | Always when windowing. |


---

### 5. Quality Indexing Parameters

Controls frame/window-level quality filtering.

| Parameter | Type | Default | Description | When to Use |
|-----------|------|---------|-------------|-------------|
| radar_frames_quality_index | str | per_radar_frame, per_window, or unset | Quality filtering mode. None = disabled. per_radar_frame works standalone. per_window needs "Simple Window". | **Optional** |
| radar_max_points_per_frame | int | Max points per frame after selection | Cap on points kept per frame (shared with point cloud). | **Required** if "Simple Window", "Point Cloud Data" active and Quality filtering needed. |
| radar_min_points_per_frame | int | Min real points required per frame (per_radar_frame mode) | Drop individual frames with < N real points. | **Required** only if radar_frames_quality_index: per_radar_frame. Set >0 to drop sparse frames. Make sure "Point Cloud Data" is active in data_proc_transforms |
| radar_min_points_per_window | int | Min total real points across window (per_window mode) | Drop entire windows with < N total points across all frames. | **Required** only if radar_frames_quality_index: per_window. Set >0 to drop low-quality windows. Make sure "Point Cloud Data" is active in data_proc_transforms|

**Comparison:**

| Mode | Parameters | Drops | Requires "Simple Window" |
|------|------------|-------|-------------------------|
| unset / empty | N/A | Nothing | No |
| per_radar_frame | radar_min_points_per_frame | Individual frames with < N real points | No (works standalone) |
| per_window | radar_min_points_per_window | Entire windows with < N total points across all frames | Yes |

---

### 6. Feature Ordering Parameter

| Parameter | Type | Default | Description | When to Use |
|-----------|------|---------|-------------|-------------|
| radar_feature_ordering | str | frame_major or feature_major | Flattening order for the window tensor. Must match the model's expectation. | **Optional**. Set when the model expects a specific feature layout. |


---


*Generated for Tensorlab EdgeAI — Making mmWave Radar ML Accessible*