"""
annotate_radar_dataset.py

Converts mmWave Data Recorder capture folder ("binData" layout) into a
Tensorlab-ready BYOD dataset ZIP containing classes/<class_name>/<session>.csv,
with columns matching the Tensorlab standards :
(session_id, track_id, track_x/y/z, track_velocity_x/y/z, track_acc_x/y/z,
point_x{i}/point_y{i}/point_z{i}/point_doppler{i}/point_snr{i}).

Important Modification needed in binData Folder from User's end :
To develop 'class label' named folders (eg. falling, walking etc.) and 
classify the session folders in binData accordingly.

EXPECTED INPUT 
--------------------------------------------------------------------------
<input>/
  <class_name>/                      # e.g. falling, sitting, standing, walking, laying
    <session_name>/                  # one recording session
      replay_<timestamp>.csv         # per-frame dict dump (frameData_pointCloud, frameData_trackData, ...)
      replay_<timestamp>.json        # same recording, JSON form 
      pHistBytes_<timestamp>_.bin    # raw point-cloud history bytes (not used here)
    ...

i.e. one subfolder per class, each holding one subfolder per recording session
(a session = one replay_*.csv/.json pair). A loose replay_*.csv/.json sitting
directly in a class folder (no session subfolder) also works - it becomes its
own session named after the file. `<input>` itself can be the class-folders
level, or its parent if it contains a `classes/` subfolder.

Only one recording source is read per session (CSV preferred over JSON, since
both encode the same frames) so a session's frames are never double-counted.

OUTPUT
------
A single ZIP file containing:
  classes/<class_name>/<session_name>.csv   (one file per recording session)

USAGE
-----
  python annotate_radar_dataset.py --input /path/to/binData --output /path/to/my_radar_dataset.zip

  my_radar_dataset.zip is then ready to be used directly as
  dataset.input_data_path for this example's config.yaml.
"""

import argparse
import csv
import io
import json
import os
import zipfile
from ast import literal_eval
from collections import Counter

# Track vector layout coming out of the mmWave tracker TLV (see parseTrackTLV):
# [tid, x, y, z, vx, vy, vz, ax, ay, az, G, confidence, ec0..ec3]. Only the
# first 10 fields are meaningful for Tensorlab's radar_track_data_selector groups.
TRACK_FIELDS = ["track_x", "track_y", "track_z",
                 "track_velocity_x", "track_velocity_y", "track_velocity_z",
                 "track_acc_x", "track_acc_y", "track_acc_z"]

# Point tuple layout coming out of the point-cloud TLV: [x, y, z, doppler, snr, noise, trackIdx].
# Only the first 5 fields map to Tensorlab's point_* columns.
POINT_FIELDS = ["point_x", "point_y", "point_z", "point_doppler", "point_snr"]

BASE_COLUMNS = ["session_id", "track_id"] + TRACK_FIELDS


def safe_eval(value):
    """Parse a stringified Python list ('[]', '[[0.0, 1.2, ...]]') or pass through an existing list. None/NaN/empty -> []."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float):
        # NaN cells read back from CSV by csv.DictReader never happen (it yields strings),
        # but guard anyway if this is ever called with a pre-parsed pandas cell.
        return []
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def discover_class_root(input_dir):
    """If <input>/classes exists, that's the folder of class subfolders; otherwise <input> itself is."""
    candidate = os.path.join(input_dir, "classes")
    return candidate if os.path.isdir(candidate) else input_dir


def discover_sessions(class_dir):
    """
    Walk a class folder and group replay_*.csv/.json files into sessions.

    A session subfolder (containing one or more replay_<ts>.csv/.json pairs) becomes one
    session per distinct stem, named after the subfolder when it holds exactly one
    recording (the common case), or "<subfolder>__<stem>" when it holds more than one.
    A loose raw file directly in the class folder becomes its own session, named after
    its stem. Returns {session_name: {"csv": path_or_None, "json": path_or_None}}.
    """
    sessions = {}
    for root, _dirs, files in os.walk(class_dir):
        file_map = {}
        for filename in files:
            lower = filename.lower()
            if lower.endswith(".csv"):
                file_map.setdefault(filename[:-4], {})["csv"] = os.path.join(root, filename)
            elif lower.endswith(".json"):
                file_map.setdefault(filename[:-5], {})["json"] = os.path.join(root, filename)
        if not file_map:
            continue

        if root == class_dir:
            for stem, paths in file_map.items():
                sessions[stem] = paths
        else:
            folder_name = os.path.basename(root)
            if len(file_map) == 1:
                ((_stem, paths),) = file_map.items()
                sessions[folder_name] = paths
            else:
                for stem, paths in file_map.items():
                    sessions[f"{folder_name}_{stem}"] = paths

    return sessions


def load_frames(session_paths):
    """
    Load one session's frames as a list of dicts with keys 'pointCloud', 'trackData',
    'trackIndexes'. Prefers the CSV source; falls back to JSON only if no CSV exists,
    so a session with both never has its frames counted twice.
    """
    if session_paths.get("csv"):
        return load_frames_from_csv(session_paths["csv"]), "csv"
    return load_frames_from_json(session_paths["json"]), "json"


def load_frames_from_csv(csv_path):
    frames = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frames.append({
                "pointCloud": safe_eval(row.get("frameData_pointCloud")),
                "trackData": safe_eval(row.get("frameData_trackData")),
                "trackIndexes": safe_eval(row.get("frameData_trackIndexes")),
            })
    return frames


def load_frames_from_json(json_path):
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    frames = []
    for frame in payload.get("data", []):
        frame_data = frame.get("frameData", {}) or {}
        frames.append({
            "pointCloud": frame_data.get("pointCloud") or [],
            "trackData": frame_data.get("trackData") or [],
            "trackIndexes": frame_data.get("trackIndexes") or [],
        })
    return frames


def determine_primary_track_id(frames, override):
    """
    Pick the track ID to extract for this session: the user's --track-id override if
    given, otherwise the most frequently observed track ID across all frames (ties
    broken by the lowest ID). Locking one ID per session - rather than following
    whichever track happens to be present in each frame - keeps a single recording
    from silently mixing frames between two different physical targets.
    """
    if override is not None:
        return override
    counts = Counter()
    for frame in frames:
        for track in frame["trackData"]:
            if track:
                counts[int(track[0])] += 1
    if not counts:
        return 0
    max_count = max(counts.values())
    return min(tid for tid, count in counts.items() if count == max_count)


def format_value(value):
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def extract_rows(frames, session_id, primary_tid):
    """
    Build one output row per frame that contains the primary track, keeping only the
    point-cloud points whose trackIndexes entry matches that track (mirrors the
    mmWave Data Recorder tool's own track-filtered CSV output). Frames without the
    primary track are dropped (nothing valid to report for this session's target).
    Returns (rows, frames_dropped_no_track).
    """
    rows = []
    dropped_no_track = 0

    for frame in frames:
        track = None
        for candidate in frame["trackData"]:
            if candidate and int(candidate[0]) == primary_tid:
                track = candidate
                break
        if track is None:
            dropped_no_track += 1
            continue

        row = {
            "session_id": session_id,
            "track_id": format_value(track[0]),
        }
        for name, value in zip(TRACK_FIELDS, track[1:10]):
            row[name] = format_value(value)

        point_cloud = frame["pointCloud"]
        track_indexes = frame["trackIndexes"]
        points = []
        for idx, point in enumerate(point_cloud):
            point_track = track_indexes[idx] if idx < len(track_indexes) else None
            if point_track is None or int(point_track) != primary_tid:
                continue
            points.append(point[:5])
        row["_points"] = points
        rows.append(row)

    return rows, dropped_no_track


def render_session_csv(rows):
    """Render one session's rows to CSV text. Returns (csv_text, max_points)."""
    max_points = max((len(row["_points"]) for row in rows), default=0)

    fieldnames = list(BASE_COLUMNS)
    for i in range(1, max_points + 1):
        fieldnames.extend(f"{field}{i}" for field in POINT_FIELDS)

    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(fieldnames)
    for row in rows:
        line = [row[col] for col in BASE_COLUMNS]
        for point in row["_points"]:
            line.extend(format_value(v) for v in point)
        line.extend([""] * (max_points - len(row["_points"])) * len(POINT_FIELDS))
        writer.writerow(line)

    return buf.getvalue(), max_points


def process_class(class_name, class_dir, zf):
    sessions = discover_sessions(class_dir)
    if not sessions:
        print(f"  [{class_name}] no replay_*.csv/.json sessions found - skipping")
        return 0

    print(f"  [{class_name}] {len(sessions)} session(s)")
    written = 0
    for session_name, session_paths in sorted(sessions.items()):
        frames, source = load_frames(session_paths)
        if not frames:
            print(f"    - {session_name}: 0 frames (empty/unreadable {source}) - skipped")
            continue

        primary_tid = determine_primary_track_id(frames, override=args.track_id)
        rows, dropped = extract_rows(frames, session_name, primary_tid)
        if not rows:
            print(f"    - {session_name}: 0/{len(frames)} frames kept (no track id {primary_tid} found) - skipped")
            continue

        csv_text, max_points = render_session_csv(rows)
        zf.writestr(f"classes/{class_name}/{session_name}.csv", csv_text)
        written += 1
        note = " (json ignored, csv present)" if source == "csv" and session_paths.get("json") else ""
        print(f"    - {session_name}: {len(rows)}/{len(frames)} frames kept "
              f"(track {primary_tid}, {dropped} dropped, max {max_points} pts/frame, source {source}{note})")

    return written


def main():
    class_root = discover_class_root(args.input)
    class_dirs = sorted(
        name for name in os.listdir(class_root)
        if os.path.isdir(os.path.join(class_root, name))
    )
    if not class_dirs:
        raise SystemExit(f"No class subfolders found under {class_root}")

    output_path = args.output
    if not output_path.lower().endswith(".zip"):
        output_path += ".zip"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Input class folder: {class_root}")
    print(f"Classes found: {class_dirs}")
    print(f"Output dataset zip: {output_path}")
    print()

    total_written = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for class_name in class_dirs:
            total_written += process_class(class_name, os.path.join(class_root, class_name), zf)

    if total_written == 0:
        os.remove(output_path)
        raise SystemExit("No sessions were converted; no ZIP was written.")

    print()
    print(f"Done. {total_written} session CSV(s) written. "
          f"Point config.yaml's dataset.input_data_path at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a raw mmWave Data Recorder 'binData' capture folder into a "
                    "Tensorlab-ready ZIP (classes/<class_name>/<session>.csv) radar dataset."
    )
    parser.add_argument("--input", required=True,
                         help="Path to the binData folder (or its parent, if it contains a 'classes' subfolder).")
    parser.add_argument("--output", required=True,
                         help="Output ZIP path (classes/<class_name>/<session>.csv written inside it). "
                              "'.zip' is appended if missing.")
    parser.add_argument("--track-id", type=int, default=None,
                         help="Force a specific track ID instead of auto-detecting the dominant one per session.")
    args = parser.parse_args()
    main()
