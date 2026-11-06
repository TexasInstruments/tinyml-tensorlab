import numpy as np
import pandas as pd
from pathlib import Path

# This preprocessing script was used to preprocess the wifi_presence_detection_dsi dataset.

# NOTE: You don't have to run this script again on the wifi_presence_detection_dsi dataset since it has already been preprocessed.
# This script is provided as a reference for the user to understand how the dsi dataset was preprocessed prior to model training.

IN_ROOT  = Path("/path/to/captured/csi/dataset")
OUT_ROOT = Path("preprocessed_wifi_presence_detection")

LABEL_TO_FOLDER = {0: "class_0_no_presence", 1: "class_1_presence"}
FOLDER_TO_LABEL = {"no_presence": 0, "presence": 1}

Fs          = 128.0
WIN_SEC     = 2.0
DROP_COLS   = {"tx_mac", "rx_mac", "packet_no", "hw_seq"}

CSI_USED_IDX = np.array(list(range(0, 26)) + list(range(27, 53)), dtype=int)
N_SC         = len(CSI_USED_IDX)  # 52


def interpolate_to_grid(tw, Xw_cplx, fs, win_sec):
    if len(tw) < 2:
        return None, None

    order = np.argsort(tw)
    tw, Xw_cplx = tw[order], Xw_cplx[order]

    Xm = np.abs(Xw_cplx).astype(np.float64)
    T, S = Xm.shape

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
    df = pd.read_csv(csv_path, low_memory=False)

    ts_col = next((c for c in ("timestamp", "mcu_timestamp") if c in df.columns), None)
    if ts_col is None:
        return None, None

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore", inplace=True)

    t_raw = df[ts_col].to_numpy(dtype=float)
    in_microseconds = np.nanmax(t_raw) > 1e5
    t = (t_raw - t_raw[0]) / 1e6 if in_microseconds else t_raw - t_raw[0]

    Xc = extract_csi(df)
    if Xc is None:
        return None, None

    Xi = interpolate_to_grid(t, Xc, Fs, WIN_SEC)
    if Xi is None:
        return None, None

    N   = int(round(WIN_SEC * Fs))
    t_start = t_raw[0] / 1e6 if in_microseconds else t_raw[0]
    tq  = t_start + np.arange(N) / Fs
    return (tq).astype(np.float32), Xi


def main():
    classes_root = OUT_ROOT / "classes"
    classes_root.mkdir(parents=True, exist_ok=True)

    csv_files = []
    for class_dir in sorted((IN_ROOT / "classes").iterdir()):
        if not class_dir.is_dir() or class_dir.name not in FOLDER_TO_LABEL:
            continue
        for p in sorted(class_dir.glob("*.csv")):
            csv_files.append((p, class_dir.name))
    print(f"Found {len(csv_files)} CSV files\n")

    n_files = 0
    for csv_path, class_name in csv_files:
        label  = FOLDER_TO_LABEL[class_name]
        folder = classes_root / LABEL_TO_FOLDER[label]
        folder.mkdir(exist_ok=True)

        tq, Xi = process_file(csv_path)
        if Xi is None:
            print(f"  [{class_name}] {csv_path.name}: skipped")
            continue

        cols = ["time"] + [f"sub_{i}" for i in CSI_USED_IDX]
        fname = f"{csv_path.stem}.csv"
        pd.DataFrame(np.column_stack([tq, Xi]), columns=cols).to_csv(folder / fname, index=False)
        n_files += 1
        if n_files % 10 == 0:
            print(f"  {n_files} files processed...")

    print(f"\nDone: {n_files} files -> {OUT_ROOT}")


if __name__ == "__main__":
    main()