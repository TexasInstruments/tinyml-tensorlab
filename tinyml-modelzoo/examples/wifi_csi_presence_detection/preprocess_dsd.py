import re
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# Input CSI recordings and output directory.
# Download wifi_presence_detection_dsd.zip from:
#   https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wifi_presence_detection_dsd.zip
# Then set IN_ROOT to either:
#   - the path to the downloaded .zip  (zip file will be auto-extracted on first run)
#   - the already-extracted wifi_presence_detection_dsd/Lab/ directory
IN_ROOT  = Path(r"/path/to/wifi_presence_detection_dsd.zip")
OUT_ROOT = Path("preprocessed_wifi_presence_detection_dsd")


def _resolve_input_root(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    if not path.exists():
        raise FileNotFoundError(f"Zip not found: {path}")
    extract_to = path.parent / path.stem
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {path.name} -> {extract_to} ...")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(extract_to)
    lab_dir = extract_to / "Lab"
    if not lab_dir.exists():
        raise FileNotFoundError(f"Expected Lab/ dir not found after extraction: {lab_dir}")
    return lab_dir

# All 13 activity tokens; only no_presence_np_empty maps to class 0
TOKENS_ALL = (
    "no_presence_np_empty",
    "presence_p_motion",
    "presence_p_still_left",
    "presence_p_still_right",
    "presence_p_typing_left",
    "presence_p_typing_right",
    "presence_p_typing_back_left",
    "presence_p_typing_back_right",
    "presence_p_sitting_left",
    "presence_p_sitting_right",
    "presence_p_sitting",
    "p_sitting_left",
    "presence_p_still",
)

TOKEN_TO_LABEL  = {t: (0 if t == "no_presence_np_empty" else 1) for t in TOKENS_ALL}

CSV_STEM_RE = re.compile(
    rf"^(?P<prefix>.*?)(?P<token>{'|'.join(map(re.escape, TOKENS_ALL))})_"
    rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}})_(?P<time>\d{{4}})$",
    re.IGNORECASE,
)

LABEL_TO_FOLDER = {0: "class_0_no_presence", 1: "class_1_presence"}

# Train/val/test date assignment — same split as main.py

TRAIN_DAYS = {"2025-12-22", "2026-01-22", "2025-08-01", "2025-08-02", "2025-08-03", "2025-12-21"}
TEST_DAYS  = {"2025-12-20", "2026-01-23"}

VAL_SIZE   = 0.2
SEED       = 1337

# Signal parameters
Fs              = 128.0
WIN_SEC     = 2.0                          # physical capture window length
MODEL_SEC   = 1.0                          # model input length after decimation
CHUNK_RAW   = int(WIN_SEC   * Fs)          # 256 samples
CHUNK_MODEL = int(MODEL_SEC * Fs)          # 128 samples
OVERLAP     = 0.30
STRIDE      = int(CHUNK_RAW * (1 - OVERLAP))  # ~179 samples between windows
TRIM_SEC    = 10.0                         # skip first 10 s of each recording (transient)
DROP_COLS   = {"tx_mac", "rx_mac", "packet_no", "hw_seq"}

# 52 usable subcarriers — hardware index 26 is the DC subcarrier, so we skip it
CSI_USED_IDX = np.array(list(range(0, 26)) + list(range(27, 53)), dtype=int)
N_SC         = len(CSI_USED_IDX)  # 52


def parse_stem(stem):
    m = CSV_STEM_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    # normalise token to its canonical casing
    tl = d["token"].lower()
    for t in TOKENS_ALL:
        if t.lower() == tl:
            d["token"] = t
            break
    return d

def interpolate_to_grid(tw, Xw_cplx, fs, win_sec):
    """Resample |CSI| magnitude onto a uniform time grid of (win_sec * fs) points."""
    if len(tw) < 2:
        return None

    order = np.argsort(tw)
    tw, Xw_cplx = tw[order], Xw_cplx[order]

    Xm = np.abs(Xw_cplx).astype(np.float64)
    T, S = Xm.shape

    # fill any NaN gaps with linear interpolation before resampling
    for s in range(S):
        bad = np.isnan(Xm[:, s])
        if bad.any():
            idx  = np.arange(T)
            good = ~bad
            Xm[:, s] = np.interp(idx, idx[good], Xm[good, s]) if good.any() else 0.0

    N  = int(round(win_sec * fs))
    tq = tw[0] + np.arange(N) / fs
    Xi = np.empty((N, S), dtype=np.float64)
    for s in range(S):
        Xi[:, s] = np.interp(tq, tw, Xm[:, s])

    return Xi.astype(np.float32)

def extract_csi(df):
    """Pull complex CSI columns and select the 52 usable subcarriers."""
    sub_cols = [c for c in df.columns if c.startswith("sub_")]
    if not sub_cols:
        return None

    idx_map = {}
    for c in sub_cols:
        try:
            idx_map[int(c.split("_")[1])] = c
        except ValueError:
            pass

    used = [idx_map[i] for i in CSI_USED_IDX if i in idx_map]
    if len(used) != N_SC:
        return None

    return df[used].replace("i", "j", regex=True).astype(complex).to_numpy()


def process_file(csv_path):
    """Drop metadata cols, extract 52-subcarrier CSI magnitudes, resample to uniform grid."""
    df = pd.read_csv(csv_path, low_memory=False)
    ts_col = next((c for c in ("timestamp", "mcu_timestamp") if c in df.columns), None)
    if ts_col is None:
        return None

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore", inplace=True)

    t_raw = df[ts_col].to_numpy(dtype=float)
    # timestamps are microseconds when > 1e5, otherwise already in seconds
    t = (t_raw - t_raw[0]) / 1e6 if np.nanmax(t_raw) > 1e5 else t_raw - t_raw[0]

    keep = t >= TRIM_SEC
    if keep.sum() < CHUNK_RAW:
        return []

    df = df.loc[keep].reset_index(drop=True)
    t  = t[keep]
    t_raw_trimmed = t_raw[keep]
    Xc = extract_csi(df)
    if Xc is None or Xc.shape[0] < CHUNK_RAW:
        return []

    windows = []
    tqs = []
    for start in range(0, len(t) - CHUNK_RAW + 1, STRIDE):
        t0   = t[start]
        mask = (t >= t0) & (t < t0 + WIN_SEC)
        if not mask.any():
            continue
        Xi = interpolate_to_grid(t[mask], Xc[mask], Fs, WIN_SEC)
        if Xi is None or Xi.shape[1] != N_SC:
            continue
        # interpolation can land ±1 sample off — force exactly 256
        if Xi.shape[0] != CHUNK_RAW:
            src = np.linspace(0, Xi.shape[0] - 1, CHUNK_RAW)
            tmp = np.empty((CHUNK_RAW, N_SC), dtype=np.float32)
            for s in range(N_SC):
                tmp[:, s] = np.interp(src, np.arange(Xi.shape[0]), Xi[:, s])
            Xi = tmp
        
        N   = int(round(WIN_SEC * Fs))
        t_start = t_raw_trimmed[start] / 1e6 if np.nanmax(t_raw_trimmed) > 1e5 else t_raw_trimmed[start]
        tq  = t_start + np.arange(N) / Fs
        tq = (tq).astype(np.float32)

        windows.append(Xi)
        tqs.append(tq)
    # N   = int(round(WIN_SEC * Fs))
    # t_start = t_raw[0] / 1e6 if in_microseconds else t_raw[0]
    
    return tqs, windows

def build_entries_from_output():
    """Rebuild entries list by scanning already-processed output files."""
    entries = []
    folder_to_label = {v: k for k, v in LABEL_TO_FOLDER.items()}
    for label_folder, label in folder_to_label.items():
        for out_path in sorted((OUT_ROOT / "classes" / label_folder).glob("*.csv")):
            info = parse_stem(out_path.stem)
            if info is None:
                print(f"  WARNING: could not parse stem: {out_path.name}")
                continue
            rel_path = out_path.relative_to(OUT_ROOT / "classes").as_posix()
            entries.append((rel_path, label, info["token"], info["date"], info.get("prefix", ""), out_path.name))
    return entries


def main(annotations_only=False):
    global IN_ROOT
    if not annotations_only:
        IN_ROOT = _resolve_input_root(IN_ROOT)

    classes_root     = OUT_ROOT / "classes"
    annotations_root = OUT_ROOT / "annotations"
    classes_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    if annotations_only:
        print("Annotations-only mode: scanning existing output files...")
        entries = build_entries_from_output()
        print(f"Found {len(entries)} output files\n")
    else:
        if not IN_ROOT.exists():
            print(f"ERROR: Input root not found: {IN_ROOT}")
            return

        csv_files = [(p, parse_stem(p.stem)) for p in IN_ROOT.rglob("*.csv")]
        csv_files = [(p, info) for p, info in csv_files if info is not None]
        print(f"Found {len(csv_files)} CSV files\n")

        entries = []
        n_files = 0

        for csv_path, info in csv_files:
            label  = TOKEN_TO_LABEL[info["token"]]
            folder = classes_root / LABEL_TO_FOLDER[label]
            folder.mkdir(exist_ok=True)

            tqs, Xi = process_file(csv_path)
            if Xi is None or len(Xi) == 0:
                print(f"  [{info['token']}] {csv_path.name}: skipped")
                continue

            fname    = f"{csv_path.stem}.csv"
            out_path = folder / fname
            cols = ["time"] + [f"sub_{i}" for i in CSI_USED_IDX]
            pd.DataFrame(np.column_stack([np.concatenate(tqs, axis=0), np.concatenate(Xi, axis=0)]), columns=cols).to_csv(out_path, index=False)

            rel_path = out_path.relative_to(OUT_ROOT / "classes").as_posix()
            entries.append((rel_path, label, info["token"], info["date"], info.get("prefix", ""), csv_path.name))

            n_files += 1
            if n_files % 10 == 0:
                print(f"  {n_files} files processed...")

    # build metadata and assign splits
    df = pd.DataFrame(entries, columns=["file", "label", "condition", "date", "user", "source_file"])

    df_pool = df[df["date"].isin(TRAIN_DAYS)]
    df_test = df[df["date"].isin(TEST_DAYS)]

    unknown = df[~df["date"].isin(TRAIN_DAYS) & ~df["date"].isin(TEST_DAYS)]
    if not unknown.empty:
        print(f"WARNING: {len(unknown)} windows from unrecognised dates: {sorted(unknown['date'].unique())}")
    
    if df_pool.empty:
        df_train = df_val = df_pool
    else:
        groups = df_pool["date"].fillna("NA").values
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
        tr_idx, va_idx = next(gss.split(df_pool.index.values, groups=groups))
        df_train = df_pool.iloc[tr_idx]
        df_val   = df_pool.iloc[va_idx]

    train_set = set(df_train.index)
    val_set   = set(df_val.index)
    test_set  = set(df_test.index)

    def assign_split(i):
        if i in train_set: return "train"
        if i in val_set:   return "val"
        if i in test_set:  return "test"
        return "unassigned"

    df["split"] = [assign_split(i) for i in df.index]

    def write_list(path, items):
        path.write_text("\n".join(items) + "\n", encoding="utf-8")

    write_list(annotations_root / "file_list.txt",            df["file"].tolist())
    write_list(annotations_root / "instances_train_list.txt", df_train["file"].tolist())
    write_list(annotations_root / "instances_val_list.txt",   df_val["file"].tolist())
    write_list(annotations_root / "instances_test_list.txt",  df_test["file"].tolist())

    df.to_csv(annotations_root / "metadata.csv", index=False)

    print(f"\nVal sources (GroupShuffleSplit): {sorted(df_val['source_file'].unique())}")
    print(df["split"].value_counts().to_string())
    print(f"\nDone -> {OUT_ROOT}")


if __name__ == "__main__":
    import sys
    main(annotations_only="--annotations-only" in sys.argv)