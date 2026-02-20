# Arc Fault Dataset Labeling Tool
### - Laavanaya Dhawan, Nathan Nohr, Adithya Thonse 
Frame-based and Window-based Label Generation for ML Training
# 1. Overview
This tool converts raw arc-fault measurement CSV files into labeled datasets suitable for machine-learning training.
It supports two labeling strategies:
- Frame Mode – produces contiguous ARC / NORMAL waveform segments using a state-machine approach.
- Window Mode – produces fixed-length ARC / NORMAL windows using sliding-window logic.
The output of both modes is a set of .txt files containing waveform samples, organized under:

```text
     <dataset_path>/classes/
```
Each .txt file corresponds to one labeled training example.
# 2. Installation
**Python version**: Python 3.8 or newer is recommended
## Required Python packages
Install dependencies using pip:
```python
pip install numpy pandas
```
## 3. Input CSV Format (Required)
### 1.  Frame Mode CSVs (4 columns)
Frame mode expects CSV files with exactly 4 columns in the following order:


|      Column Index (1-based)          |Signal                          |Description                          |
|----------------|-------------------------------|-----------------------------|
|1|Time            |Sample time (seconds or ticks)|
|2          |Bandpass Current           |Load current (ADC or scaled units)    |
|3          |Arc Voltage |Arc-gap voltage|
|4         |LOG300 |LOG300 log-amplifier envelope output|

✔ Column names are not required
✔ Non-numeric rows are automatically ignored
### 2. Window Mode CSVs (Expected layout)
Window mode expects a directory structure such as:
```text
csv_folder/
├── arc/
│   ├── compressor/
│   │   ├── *.csv
│   ├── snps/
│   │   ├── *.csv
└── normal/
    ├── compressor/
    │   ├── *.csv
    ├── snps/
    │   ├── *.csv
```
or 

```text
csv_folder/
├── arc/
│   │   ├── *.csv
│   │   ├── *.csv
└── normal/
    │   ├── *.csv
    │   ├── *.csv
```
Each CSV must contain at least(LOG300 is not used in window mode):
x → current
y → arc voltage

# 4. Frame Mode
## Purpose
Frame mode generates variable-length ARC and NORMAL waveform segments, closely matching real arc events.
This mode is intended for LOG300-based ML training.
## How labeling works (Frame Mode)
1. **Frame segmentation**: The input waveform is divided into fixed-length frames (for example, 512 samples per frame).
Per-frame voltage analysis
2. **For each frame, the arc-gap voltage is analyzed:**
- Peak voltage (Vmax)
- RMS voltage (Vrms)
A frame is marked as arc-like if:
```
Vmax ≥ V_MIN, and Vrms ≤ V_MAX
MIN_V = 110.0
MAX_V = (2**11) * (0.2) / 0.707
NORM_V = 110.0
```
3. **Initial frame classification**: Based on the voltage conditions:
- Frames satisfying the arc condition are marked as potential ARC frames
- A frame is marked as a normal candidate if its peak voltage stays below a defined normal threshold.
4.  **Temporal smoothing across frames**
Real arcing events typically persist across multiple consecutive half-cycles and do not appear as isolated single-frame spikes.
To avoid false detections caused by noise or switching transients, temporal smoothing is applied.A sliding window of consecutive frames is evaluated. If enough frames within this window satisfy the arc condition, all frames in that window are promoted to ARC. 
This effectively suppresses short, isolated spikes while preserving sustained arc behavior. Any frame that overlaps with a confirmed arc region is explicitly removed from the normal set, even if it individually looked normal.
This guarantees that normal data is “pure”, meaning it does not contain samples immediately before, during, or after an arc event.
Consecutive ARC frames are merged into a single ARC segment
Consecutive NORMAL frames are merged into a single NORMAL segment. Transitions between ARC, NORMAL, and NONE states are tracked explicitly. Only segments longer than a minimum duration are saved, ensuring that the dataset contains meaningful training examples rather than very short fragments.
5. **NOTE**: Segments shorter than 8 frames are discarded to remove transients.

## What is saved
In frame mode, the labeling logic always uses the voltage channel to determine whether a frame corresponds to an arc or normal condition. However, the signal that gets saved for each labeled segment is configurable.
This is controlled using the --frame_save_col argument:
The value specifies the 1-based CSV column index to save as the output waveform.
This allows the same labeling logic to be reused for different front-end architectures.
Typical usage:
- **LOG300 datasets (4-column CSVs)**: Use --frame_save_col 4 to save the LOG300 envelope signal.
- **LPF / current-only datasets (3-column CSVs)**: Use --frame_save_col 2 to save the current waveform.
The script assumes:
Column 1: Time
Column 3: Voltage (used for labeling logic)
Only the saved waveform channel changes; the arc/normal decision logic remains identical.

```
Saved signal: LOG300 (column 4)/ Bandpass Current(column 2)
Used for labeling: Arc voltage (column 3)

Each output .txt file contains only LOG300 samples.
```
## Frame Mode Output
```
dataset_path/
└── classes/
    ├── arc_dir/
    │   ├── arc_<name>_<id>_<length>.txt
    └── normal_dir/
        ├── normal_<name>_<id>_<length>.txt
 ```
Each file represents one labeled training example.
## How to run (Frame Mode)

```python 
python label_tool.py \
  --mode frame \
  --dataset_path <output_dataset_path> \
  --csv_folder <csv_input_path> \
  --frame_size 512 \
  --frame_thresh 4 \
  --frame_smooth_frames 6
  --frame_save_col 4
  ```
  # 5. Window Mode
## Purpose

1. **Frame segmentation**:The input waveform is first divided into fixed-length frames (for example, 1024 samples per frame).
2. **Sliding window formation**:A window is defined as a fixed number of consecutive frames (for example, 8 frames per window).
This window slides forward one frame at a time across the signal.
3. **Per-frame voltage analysis**:For each frame inside the window, the arc-gap voltage is analyzed:
Peak voltage (Vmax)
RMS voltage (Vrms)
A frame is considered arc-like if:
```
Vmax ≥ V_MIN, and Vrms ≤ V_MAX
```
Currently these thresholds are fixed as follows: 
```
V_MIN = (2**11) * 0.12
V_MAX = (2**11) * (0.34) / 0.707
```
4. **Frame counting inside the window**
For every window position:The algorithm counts how many frames inside the window are arc-like. This count represents how persistent the arc behavior is over that time span. 
5. **Temporal smoothing at the window level**: Instead of making a decision based on a single frame, the algorithm applies temporal smoothing across the entire window:
- If the number of arc-like frames inside the window is greater than or equal to a configurable threshold. The entire window is labeled as ARC
- Otherwise, if the voltage levels are sufficiently low and stable:
The window is labeled as NORMAL.Normal windows are selected based on the final frame voltage in the window. Avoids capturing decay/transient tails after an arc.
- Windows that do not clearly satisfy either condition are discarded

Once a window is labeled, the corresponding waveform samples are extracted. Each window is saved as a fixed-length training example
ARC and NORMAL windows are stored separately.
## What is saved
> Saved signal: Current (x)
Used for labeling: Arc voltage (y)
LOG300: Not used
## Window Mode Output (flattened)
```
dataset_path/
└── classes/
    ├── arc/
    │   ├── compressor_arc.txt
    │   ├── snps_arc.txt
    └── normal/
        ├── compressor_arc_normal.txt
        ├── compressor_pure_normal.txt
```
Application-level folders are automatically merged and flattened.
## How to run (Window Mode)

```python
python label_tool.py \
  --mode window \
  --dataset_path <output_dataset_path> \
  --csv_folder <csv_input_path> \
  --frame_size 1024 \
  --frame_thresh 5 \
  --window_frames 8
```
Optional dataset balancing:
```
--balanced
```
# 6. LOG300 Usage Explanation
- LOG300 captures the high-frequency arc noise envelope, providing sensitivity to arc activity that is not visible in low-frequency current or voltage signals.
- LOG300 is not used to decide labels. Arc and normal labels are determined using arc-voltage statistics, ensuring deterministic and hardware-independent ground truth.
- In Frame Mode, LOG300 is saved as the ML input waveform, allowing models to learn arc signatures from high-frequency envelope behavior.
# 7. Final Note: Frame Mode vs Window Mode
Both modes use arc-voltage thresholds to determine ARC vs NORMAL, but they differ in how time is aggregated:
- Frame Mode applies temporal smoothing across consecutive frames and then merges them into variable-length ARC/NORMAL segments, preserving the natural duration of arc events.
- Window Mode evaluates fixed-size sliding windows and labels or discards each window independently, keeping only high-confidence ARC or NORMAL windows of uniform length.
> In short:
Frame Mode preserves event continuity.
Window Mode prioritizes fixed-size, high-confidence samples.