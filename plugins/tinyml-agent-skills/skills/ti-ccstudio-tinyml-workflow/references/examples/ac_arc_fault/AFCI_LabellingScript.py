import argparse
import os
import re
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def numeric_key(name: str) -> int:
    m = re.search(r"_(\d+)\.txt$", name)
    return int(m.group(1)) if m else -1


def read_csv_numeric_matrix(file_path: Path) -> np.ndarray:
    df = pd.read_csv(file_path, header=None, low_memory=False)
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy()
    return arr[~np.all(np.isnan(arr), axis=1)]


def process_arc_csv(csv_file, FRAME_SIZE, WINDOW_FRAMES, V_MIN, V_MAX, FRAME_ARC_THRESHOLD):
    df = pd.read_csv(csv_file)

    voltage = df["y"].values.astype(float) - (2**11)
    current = (df["x"].values.astype(int) - 2048) << 4
    current = np.clip(current, -32768, 32767).astype(np.int16)

    num_frames = len(voltage) // FRAME_SIZE
    num_windows = num_frames - WINDOW_FRAMES + 1

    arc_windows, arc_normals = [], []
    base = os.path.splitext(os.path.basename(csv_file))[0]

    for w in range(num_windows):
        start = w * FRAME_SIZE
        end = start + FRAME_SIZE * WINDOW_FRAMES
        if end > len(voltage):
            break

        arc_frame_count = 0
        v_max = None

        for f in range(WINDOW_FRAMES):
            fs, fe = start + f * FRAME_SIZE, start + (f + 1) * FRAME_SIZE
            v_frame = voltage[fs:fe]
            v_rms = np.sqrt(np.mean(v_frame.astype(np.float32) ** 2))
            v_max = np.max(np.abs(v_frame))
            if (v_rms <= V_MAX) and (V_MIN <= v_max):
                arc_frame_count += 1

        if arc_frame_count >= FRAME_ARC_THRESHOLD:
            arc_windows.append((f"{base}_arc_window_{w}.txt", current[start:end]))
        else:
            if (v_max is not None) and (v_max <= V_MIN / 2):
                arc_normals.append((f"{base}_normal_window_{w}.txt", current[start:end]))

    return arc_windows, arc_normals


def process_normal_csv(csv_file, FRAME_SIZE, WINDOW_FRAMES):
    df = pd.read_csv(csv_file)

    current = (df["x"].values.astype(int) - 2048) << 4
    current = np.clip(current, -32768, 32767).astype(np.int16)

    num_frames = len(current) // FRAME_SIZE
    num_windows = num_frames - WINDOW_FRAMES + 1

    pure_normals = []
    base = os.path.splitext(os.path.basename(csv_file))[0]

    for w in range(num_windows):
        start = w * FRAME_SIZE
        end = start + FRAME_SIZE * WINDOW_FRAMES
        if end > len(current):
            break
        pure_normals.append((f"{base}_normal_window_{w}.txt", current[start:end]))

    return pure_normals


def flatten_window_outputs(classes_dir: Path):
    arc_out = classes_dir / "arc"
    normal_out = classes_dir / "normal"
    arc_out.mkdir(parents=True, exist_ok=True)
    normal_out.mkdir(parents=True, exist_ok=True)

    mapping = {
        "arc_dir": arc_out,
        "normal_dir": normal_out,
    }

    for dir_type, out_dir in mapping.items():
        source = classes_dir / dir_type
        if not source.exists():
            continue

        app_dirs = sorted([p for p in source.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

        for app_dir in app_dirs:
            files = sorted([p for p in app_dir.iterdir() if p.is_file() and p.suffix == ".txt"],
                           key=lambda p: p.name.lower())

            for file in files:
                dst_file = out_dir / f"{app_dir.name}_{file.name}"
                shutil.copy(file, dst_file)

            shutil.rmtree(app_dir)

        shutil.rmtree(source)


def run_window_mode(args):
    dataset_path = Path(args.dataset_path).resolve()
    classes_dir = dataset_path / "classes"
    ARC_DIR = classes_dir / "arc_dir"
    NORMAL_DIR = classes_dir / "normal_dir"
    classes_dir.mkdir(parents=True, exist_ok=True)
    ARC_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    FRAME_SIZE = args.frame_size
    WINDOW_FRAMES = args.window_frames
    V_MIN = (2**11) * 0.12
    V_MAX = (2**11) * (0.34) / 0.707
    T = args.frame_thresh

    csv_root = Path(args.csv_folder).resolve()
    arc_root = csv_root / "arc"
    normal_root = csv_root / "normal"

    arc_csvs_at_root = sorted(list(arc_root.glob("*.csv")), key=lambda p: p.name.lower())
    arc_subdirs = sorted([p for p in arc_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    if arc_subdirs:
        app_folders = arc_subdirs
        root_mode = False
    elif arc_csvs_at_root:
        app_folders = [arc_root]
        root_mode = True
    else:
        raise FileNotFoundError(f"No CSVs found under {arc_root} (neither subfolders nor root CSVs).")

    print(f"[DEBUG] arc_root = {arc_root}")
    print(f"[DEBUG] normal_root = {normal_root}")
    print(f"[DEBUG] app_folders = {[p.name for p in app_folders]}")
    for app_folder in app_folders:
        app = "root" if (root_mode and app_folder == arc_root) else app_folder.name

        arc_csvs = (
            arc_csvs_at_root
            if (root_mode and app_folder == arc_root)
            else sorted(list(app_folder.glob("*.csv")), key=lambda p: p.name.lower())
        )

        normal_app = normal_root if root_mode else (normal_root / app_folder.name)
        normal_csvs = sorted(list(normal_app.glob("*.csv")), key=lambda p: p.name.lower()) if normal_app.exists() else []

        arc_outdir = ARC_DIR / app
        normal_outdir = NORMAL_DIR / app
        arc_outdir.mkdir(parents=True, exist_ok=True)
        normal_outdir.mkdir(parents=True, exist_ok=True)

        arc_windows, arc_normals, pure_normals = [], [], []

        for csv in arc_csvs:
            aw, an = process_arc_csv(csv, FRAME_SIZE, WINDOW_FRAMES, V_MIN, V_MAX, T)
            arc_windows.extend(aw)
            arc_normals.extend(an)

        for csv in normal_csvs:
            pure_normals.extend(process_normal_csv(csv, FRAME_SIZE, WINDOW_FRAMES))

        if args.balanced:
            N = len(arc_windows)
            sel_an = min(len(arc_normals), N)
            sel_pn = min(len(pure_normals), N)
            selected_arc_normals = random.sample(arc_normals, sel_an) if sel_an > 0 else []
            selected_pure_normals = random.sample(pure_normals, sel_pn) if sel_pn > 0 else []
        else:
            selected_arc_normals = arc_normals
            selected_pure_normals = pure_normals

        if arc_windows:
            ordered = sorted(arc_windows, key=lambda x: numeric_key(x[0]))
            out = arc_outdir / f"{app}_arc.txt"
            with open(out, "w") as f:
                for _, data in ordered:
                    f.writelines(f"{v}\n" for v in data)

        if selected_arc_normals:
            ordered = sorted(selected_arc_normals, key=lambda x: numeric_key(x[0]))
            out = normal_outdir / f"{app}_arc_normal.txt"
            with open(out, "w") as f:
                for _, data in ordered:
                    f.writelines(f"{v}\n" for v in data)

        if selected_pure_normals:
            ordered = sorted(selected_pure_normals, key=lambda x: numeric_key(x[0]))
            out = normal_outdir / f"{app}_pure_normal.txt"
            with open(out, "w") as f:
                for _, data in ordered:
                    f.writelines(f"{v}\n" for v in data)

    flatten_window_outputs(classes_dir)

def run_frame_mode(args):
    dataset_path = Path(args.dataset_path).resolve()
    classes_dir = dataset_path / "classes"
    arc_dir = classes_dir / "arc_dir"
    normal_dir = classes_dir / "normal_dir"
    classes_dir.mkdir(parents=True, exist_ok=True)
    arc_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(Path(args.csv_folder).rglob("*.csv"), key=lambda p: str(p).lower())
    uid = 0
    save_col = args.frame_save_col  # 1-based

    FRAME_SIZE = args.frame_size
    FRAME_SMOOTH_FRAMES = args.frame_smooth_frames
    FRAME_THRESH = args.frame_thresh

    MIN_V = 150.0
    MAX_V = (2**11) * (0.24) / 0.707
    NORM_V = 110.0

    for file_path in csvs:
        pname = file_path.stem.replace("_", "")
        arc_ivl = read_csv_numeric_matrix(file_path)
        if arc_ivl.shape[0] == 0:
            continue
        arc_ivl[:, 1:3] = arc_ivl[:, 1:3] - (2**11) + 60 #The bandpass and voltage are both centered at 1.6 V , hence -2**11
        arc_ivl[:, 3] = arc_ivl[:, 3] * 8 #multiply raw log300 value by 8
    
        new_size = (arc_ivl.shape[0] // FRAME_SIZE) * FRAME_SIZE
        if new_size == 0:
            continue

        mcu_iv = np.column_stack([
            arc_ivl[:new_size, 0],
            arc_ivl[:new_size, save_col - 1],
            arc_ivl[:new_size, 2],
        ])

        n_frames = new_size // FRAME_SIZE
        frames = np.reshape(mcu_iv.T, (3, FRAME_SIZE, n_frames), order="F")

        labels = np.zeros(n_frames, dtype=np.int32)
        norms = np.zeros(n_frames, dtype=np.int32)
        combined = np.zeros(n_frames, dtype=np.int32)

        for j in range(n_frames):
            v = frames[2, :, j]
            v_max = np.max(np.abs(v))
            v_rms = np.sqrt(np.mean(v**2))
            if (v_max >= MIN_V) and (v_rms <= MAX_V):
                labels[j] = 1
            if v_max <= NORM_V:
                norms[j] = 1

        for j in range(0, n_frames - FRAME_SMOOTH_FRAMES):
            s = int(np.sum(labels[j:j + FRAME_SMOOTH_FRAMES + 1]))
            if s >= FRAME_THRESH:
                combined[j:j + FRAME_SMOOTH_FRAMES + 1] = FRAME_THRESH + 1

        norms[combined >= FRAME_THRESH] = 0

        abs_index = 1
        file_len = 1
        label_arc = False
        label_norm = False
        condition = "none"

        for j in range(1, n_frames + 1):
            prev = condition

            if norms[j - 1] >= 1:
                condition = "normal"
            elif combined[j - 1] >= FRAME_THRESH:
                condition = "arc"
            else:
                condition = "none"

            start = abs_index - 1
            end = (j - 1) * FRAME_SIZE

            if prev != condition:
                if label_arc:
                    if j != 1:
                        if file_len > 8:
                            uid += 1
                            np.savetxt(
                                arc_dir / f"arc_{pname}{uid}_{file_len}.txt",
                                mcu_iv[start:end, 1],
                                fmt="%.18g",
                            )
                        file_len = 1
                    label_arc = False
                elif label_norm:
                    if j != 1:
                        if file_len > 8:
                            uid += 1
                            np.savetxt(
                                normal_dir / f"normal_{pname}{uid}_{file_len}.txt",
                                mcu_iv[start:end, 1],
                                fmt="%.18g",
                            )
                        file_len = 1
                    label_norm = False

            if condition == "arc":
                if label_arc:
                    file_len += 1
                else:
                    file_len = 1
                    abs_index = (j - 1) * FRAME_SIZE + 1
                    label_arc = True
            elif condition == "normal":
                if label_norm:
                    file_len += 1
                else:
                    file_len = 1
                    abs_index = (j - 1) * FRAME_SIZE + 1
                    label_norm = True

        if condition == "arc":
            uid += 1
            np.savetxt(
                arc_dir / f"arc_{pname}{uid}_{file_len}.txt",
                mcu_iv[abs_index - 1:, 1],
                fmt="%.18g",
            )
        elif condition == "normal":
            uid += 1
            np.savetxt(
                normal_dir / f"normal_{pname}{uid}_{file_len}.txt",
                mcu_iv[abs_index - 1:, 1],
                fmt="%.18g",
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["frame", "window"], required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--csv_folder", required=True)

    p.add_argument("--frame_size", type=int, required=True)
    p.add_argument("--frame_thresh", type=int, required=True)
    p.add_argument(
    "--frame_save_col",
    type=int,
    default=4,
    help="1-based CSV column to save as output waveform in frame mode. "
         "Use 4 for LOG300 (4-col CSV), 2 for current (3-col LPF). (This method assumes that column 1 is time and column 3 is Voltage)"
)
    p.add_argument("--frame_smooth_frames", type=int, default=6)
    p.add_argument("--window_frames", type=int, default=8)
    p.add_argument("--balanced", action="store_true")

    args = p.parse_args()

    if args.mode == "frame":
        run_frame_mode(args)
    else:
        run_window_mode(args)


if __name__ == "__main__":
    main()