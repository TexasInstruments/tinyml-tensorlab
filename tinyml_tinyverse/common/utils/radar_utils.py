#################################################################################
# Copyright (c) 2023-2026, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

"""
Radar-specific utility functions for mmWave radar data processing.

All pipeline stage functions (Range, Point Cloud, Track, Scaling, Simple Window)
and helper functions live here as standalone, stateless functions.
"""

import os
from glob import glob
import re
from ast import literal_eval
from os.path import splitext as ops

import numpy as np
import pandas as pd


# ==================== Constants / Column Mappings ====================

# Track feature groups - The Unified Data Recorder Tool will make sure mapping of prefixes is matched
TRACK_COLUMN_PREFIXES = {
    'velocity': ['track_velocity_x', 'track_velocity_y', 'track_velocity_z'],
    'x': ['track_x'],
    'y': ['track_y'],
    'z': ['track_z'],
    'acceleration': ['track_acc_x', 'track_acc_y', 'track_acc_z'],
}

# Point cloud feature mapping
POINT_CLOUD_COLUMN_PREFIXES = {
    'x': 'point_x',
    'y': 'point_y',
    'z': 'point_z',
    'doppler': 'point_doppler',
    'snr': 'point_snr',
}

# Any of these prefixes existing for a point index means the point exists
POINT_INDEX_DISCOVERY_PREFIXES = ('point_x', 'point_y', 'point_z')

# Pipeline stage identifiers
TRANSFORM_RANGE_DATA = 'Radar Range Data'
TRANSFORM_POINT_CLOUD_DATA = 'Radar Point Cloud Data'
TRANSFORM_TRACK_DATA = 'Radar Track Data'
TRANSFORM_SCALING = 'Scaling'
TRANSFORM_SIMPLE_WINDOW = 'Simple Window'


# ==================== Kwarg Parsing Helpers ====================

def parse_list_kwarg(obj, name: str):
    """
    Normalize a kwarg on an object to a list. Handles:
    - Actual list (from parsed YAML)
    - Stringified list: "['a','b']" or "[1, 2]"
    - Comma-separated string: "a,b,c"
    - Single value: "a" -> ['a']
    - None/empty -> None (disables that filter)

    Args:
        obj: Object that has the attribute (e.g., self)
        name: Name of the attribute to normalize
    """
    if not hasattr(obj, name):
        return
    value = getattr(obj, name)
    if value is None or value in ('None', ''):
        setattr(obj, name, None)
        return
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('['):
            try:
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid {name} format: {value}")
        else:
            value = [v.strip() for v in value.split(',') if v.strip()]
    if not isinstance(value, (list, tuple)):
        value = [value]
    setattr(obj, name, list(value))


def transform_active(data_proc_transforms: list | None, token: str) -> bool:
    """
    Check if a pipeline stage is enabled via data_proc_transforms.

    Args:
        data_proc_transforms: List of active transform names
        token: Transform name to check

    Returns:
        True if token is in data_proc_transforms
    """
    return token in (data_proc_transforms or [])


def get_bounds_pairs(value: list | None, n_pairs: int) -> list[tuple[float | None, float | None]]:
    """
    Read consolidated bounds list [min1, max1, min2, max2, ...] as (min, max) tuples.
    Unset attribute -> all pairs default to (None, None) i.e. no filtering.
    String 'None' values are treated as no bound (None).

    Args:
        value: List of bound values
        n_pairs: Number of (min, max) pairs expected

    Returns:
        List of (min, max) tuples
    """
    if value is None:
        return [(None, None)] * n_pairs

    # Convert values: string 'None' -> None, others -> float
    converted = []
    for v in value:
        if v is None or (isinstance(v, str) and v.strip().lower() == 'none'):
            converted.append(None)
        else:
            try:
                converted.append(float(v))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid bound value: {v!r} (expected number or 'None')")

    if len(converted) != 2 * n_pairs:
        raise ValueError(
            f"Bounds must have {2 * n_pairs} values (min/max per field), "
            f"got {len(converted)}: {converted}"
        )
    return [(converted[2 * i], converted[2 * i + 1]) for i in range(n_pairs)]


def bounds_mask(df: pd.DataFrame, column_bounds: list[tuple[str, float | None, float | None]]) -> pd.Series:
    """
    Build boolean mask: True where all (column, min, max) constraints satisfied.
    NaN values pass through (treated as valid). Missing columns are skipped.
    None bounds mean no filtering on that side.

    Args:
        df: DataFrame to filter
        column_bounds: List of (column_name, min_value, max_value) tuples

    Returns:
        Boolean Series mask
    """
    mask = pd.Series(True, index=df.index)
    for col, min_v, max_v in column_bounds:
        if col not in df.columns:
            continue
        col_mask = pd.Series(True, index=df.index)
        if min_v is not None:
            col_mask &= (df[col] >= min_v)
        if max_v is not None:
            col_mask &= (df[col] <= max_v)
        # NaN values pass through (treated as valid)
        col_mask |= df[col].isna()
        mask &= col_mask
    return mask


def column_prefix_map(columns, prefix: str) -> list[str]:
    """
    Return columns starting with `prefix` + numeric suffix, sorted by suffix.

    Args:
        columns: Iterable of column names
        prefix: Prefix to match (e.g., 'bin', 'point_x')

    Returns:
        Sorted list of matching column names
    """
    matches = []
    for col in columns:
        col_str = str(col)
        if col_str == prefix:
            matches.append((col, -1))
            continue
        m = re.match(rf'^{re.escape(prefix)}(\d+)$', col_str)
        if m:
            matches.append((col, int(m.group(1))))
    matches.sort(key=lambda x: x[1])
    return [col for col, _ in matches]


def is_numeric(val) -> bool:
    """Check if value can be converted to float."""
    try:
        float(str(val))
        return True
    except (ValueError, TypeError):
        return False


def promote_header_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """
    If first row is non-numeric, promote to column headers.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with proper headers
    """
    if len(df) and not all(is_numeric(v) for v in df.iloc[0]):
        df.columns = df.iloc[0].astype(str)
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [str(c) for c in df.columns]
    return df


# ==================== Point Cloud Helpers ====================

def get_point_indices(df: pd.DataFrame) -> list[int]:
    """
    Discover point indices (0..N) present in this file.
    A point index exists if ANY of point_x*/point_y*/point_z* exists for it.

    Args:
        df: DataFrame with point cloud columns

    Returns:
        Sorted list of point indices
    """
    indices = set()
    for prefix in POINT_CLOUD_COLUMN_PREFIXES.values():
        for col in df.columns:
            m = re.match(rf'^{re.escape(prefix)}(\d+)$', str(col))
            if m:
                indices.add(int(m.group(1)))
    return sorted(indices)


def select_frame_points(
    frame_points: pd.DataFrame,
    max_points: int = 8,
    strategy: str = 'highest and lowest',
    sort_by: str = 'z'
) -> tuple[pd.DataFrame, int]:
    """
    Select up to max_points_per_frame points for a single frame.
    Strategy: 'highest only', 'lowest only', or 'highest and lowest' (interleaved).
    Sorted by sort_by column (default 'z' = point_z).
    Zero-pads if fewer points available.

    Args:
        frame_points: DataFrame with point features for one frame
        max_points: Maximum number of points to select
        strategy: Selection strategy
        sort_by: Column to sort by

    Returns:
        (selected_points_df, real_point_count)
    """
    strategy = str(strategy).lower()
    feature_cols = [c for c in frame_points.columns if c != 'frame_idx']

    can_sort = sort_by in frame_points.columns
    if not can_sort:
        # Can't sort, use file order
        pass

    # Sort descending (highest first)
    if can_sort:
        frame_points = frame_points.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    else:
        frame_points = frame_points.reset_index(drop=True)

    n = len(frame_points)

    if strategy == 'highest only':
        selected = frame_points.head(max_points)
    elif strategy == 'lowest only':
        if can_sort:
            selected = frame_points.sort_values(by=sort_by, ascending=True).reset_index(drop=True).head(max_points)
        else:
            selected = frame_points.head(max_points)
    elif strategy == 'highest and lowest':
        num_high = -(-max_points // 2)  # ceil
        num_low = max_points // 2
        if num_high + num_low > n:
            num_high = min(num_high, n)
            num_low = max(0, n - num_high)
        high_idx = list(range(num_high))
        low_idx = list(range(n - num_low, n))[::-1]
        interleaved = []
        for i in range(max(len(high_idx), len(low_idx))):
            if i < len(high_idx):
                interleaved.append(high_idx[i])
            if i < len(low_idx) and low_idx[i] not in interleaved:
                interleaved.append(low_idx[i])
        selected = frame_points.iloc[interleaved].reset_index(drop=True)
    else:
        raise ValueError(
            f"strategy must be 'highest only', 'lowest only', or 'highest and lowest', got {strategy!r}"
        )

    real_count = len(selected)

    # Zero-pad to max_points
    if len(selected) < max_points:
        pad = pd.DataFrame(np.zeros((max_points - len(selected), len(feature_cols))), columns=feature_cols)
        selected = pd.concat([selected[feature_cols], pad], ignore_index=True)
    else:
        selected = selected[feature_cols].iloc[:max_points].reset_index(drop=True)

    return selected, real_count


# ==================== Scaling Helpers ====================

def scale_columns(df: pd.DataFrame, prefix: str, scale_factor: float | None) -> pd.DataFrame:
    """
    Multiply all columns starting with `prefix` by `scale_factor`. No-op if None.

    Args:
        df: DataFrame to modify
        prefix: Column prefix to match
        scale_factor: Scaling factor (None to skip)

    Returns:
        Modified DataFrame
    """
    if scale_factor is None:
        return df
    for col in df.columns:
        if str(col).startswith(prefix):
            df[col] = df[col] * float(scale_factor)
    return df


# ==================== Frame Quality & Windowing ====================

def build_frame_table(
    range_df: pd.DataFrame,
    track_df: pd.DataFrame,
    points_df: pd.DataFrame | None,
    valid_mask: pd.Series,
    max_points_per_frame: int = 8,
    min_points_per_frame: int = 0,
    point_cloud_sort_by: str = 'z',
    point_cloud_structure_strategy: str = 'highest and lowest',
    radar_point_cloud_data_selector: list | None = None,
    quality_mode: str = '',
) -> tuple[pd.DataFrame, list[int] | None, list[int] | None]:
    """
    Combine range/track/point features into per-frame feature table.
    Returns (frame_table, frame_point_counts, frame_indices) where:
    - frame_table: DataFrame with one row per valid frame, columns = all features
    - frame_point_counts: real point count per frame (for per_window quality), or None
    - frame_indices: original frame indices (for consecutive check), or None

    Quality filtering:
    - per_radar_frame: drop frames with < min_points_per_frame real points
    - per_window: keep all frames, filter at window level later
    """
    quality_mode = str(quality_mode).lower()
    use_per_frame_quality = (quality_mode == 'per_radar_frame')
    use_per_window_quality = (quality_mode == 'per_window')
    # per_window quality is applied later at the window level (in transform_simple_window),
    # but it still needs frame_point_counts/frame_indices collected here to do that filtering.
    track_counts = use_per_frame_quality or use_per_window_quality

    frame_index = range_df.index if len(range_df.columns) else track_df.index
    rows = []
    frame_point_counts = [] if track_counts else None
    frame_indices = [] if track_counts else None
    dropped_per_frame = 0

    for frame_idx in frame_index:
        if not valid_mask.get(frame_idx, True):
            continue

        feature_row = {}
        # Range features
        for col in range_df.columns:
            feature_row[col] = range_df.at[frame_idx, col]
        # Track features
        for col in track_df.columns:
            feature_row[col] = track_df.at[frame_idx, col]

        real_point_count = 0
        if points_df is not None:
            frame_pts = points_df[points_df['frame_idx'] == frame_idx]
            selected_pts, real_point_count = select_frame_points(
                frame_pts,
                max_points=max_points_per_frame,
                strategy=point_cloud_structure_strategy,
                sort_by=point_cloud_sort_by
            )

            if use_per_frame_quality and real_point_count < min_points_per_frame:
                dropped_per_frame += 1
                continue

            # Flatten point features: point_{feat}_{point_num}
            pt_feature_cols = [c for c in selected_pts.columns]
            flat = selected_pts.values.flatten()
            for i, val in enumerate(flat):
                feat_name = pt_feature_cols[i % len(pt_feature_cols)]
                pt_num = i // len(pt_feature_cols)
                feature_row[f'point_{feat_name}_{pt_num}'] = val

        rows.append(feature_row)
        if frame_point_counts is not None:
            frame_point_counts.append(real_point_count)
        if frame_indices is not None:
            frame_indices.append(frame_idx)

    if not rows:
        empty = pd.DataFrame()
        return empty, frame_point_counts, frame_indices

    frame_table = pd.DataFrame(rows).fillna(0.0)
    return frame_table, frame_point_counts, frame_indices


def transform_simple_window(
    df: pd.DataFrame,
    frame_size: int,
    stride_size: float = 1.0,
    radar_min_points_per_window: int = 0,
    radar_frames_quality_index: str = '',
    frame_point_counts: list[int] | None = None,
    frame_indices: list[int] | None = None,
    radar_feature_ordering: str = 'frame_major',
    logger=None
) -> np.ndarray:
    """
    Create sliding windows of size `frame_size` with optional `stride_size`.
    Quality checks (only if 'Simple Window' active and per_window mode):
    - Min points per window
    - Consecutive frame indices (no gaps from dropped ghost tracks)
    Returns: (N_windows, frame_size * n_features) float32 array
    """
    if logger is None:
        from logging import getLogger
        logger = getLogger("root.radar_utils")

    df = df.reset_index(drop=True)
    window_size = int(frame_size)
    stride_size = stride_size if stride_size is not None else 1.0
    stride_size = float(stride_size)
    stride_window = int(window_size * stride_size)
    if stride_window == 0:
        logger.warning("Stride Window size calculated is 0. Defaulting the value to Sample Window i.e. no overlap.")
        stride_window = window_size

    quality_mode = str(radar_frames_quality_index).lower()
    use_per_window_quality = (quality_mode == 'per_window')

    # Ensure no NaN in frame table before windowing
    if df.isna().any().any():
        logger.warning(f"Frame table contains NaN values before windowing. Filling with 0.")
        df = df.fillna(0.0)

    # Read feature ordering: 'frame_major' (default) or 'feature_major'
    feature_ordering = str(radar_feature_ordering).lower()
    if feature_ordering not in ('frame_major', 'feature_major'):
        logger.warning(f"Unknown radar_feature_ordering '{feature_ordering}', defaulting to 'frame_major'")
        feature_ordering = 'frame_major'

    logger.debug(f"Windowing: input {df.shape}, window_size={window_size}, stride_size={stride_size}, stride_window={stride_window}, feature_ordering={feature_ordering}")
    windows = []
    dropped_consecutive = 0
    dropped_quality = 0

    for i in range(0, len(df) - window_size + 1, stride_window):
        # Per-window quality check
        if use_per_window_quality and frame_point_counts is not None:
            window_pts = sum(frame_point_counts[i:i + window_size])
            if window_pts < radar_min_points_per_window:
                dropped_quality += 1
                continue

        # Consecutive frame check (prevent windows spanning gaps)
        if frame_indices is not None:
            win_indices = frame_indices[i:i + window_size]
            if win_indices[-1] != win_indices[0] + window_size - 1:
                dropped_consecutive += 1
                continue

        window = df.iloc[i:i + window_size]

        # Apply feature ordering when flattening
        if feature_ordering == 'frame_major':
            # [frame0_feat0, frame0_feat1, ..., frame1_feat0, frame1_feat1, ...]
            flat = window.values.flatten()
        elif feature_ordering == 'feature_major':
            # [frame0_feat0, frame1_feat0, ..., frame0_feat1, frame1_feat1, ...]
            # Transpose: features x frames, then flatten
            flat = window.values.T.flatten()
        else:
            flat = window.values.flatten()

        windows.append(flat)

    if dropped_consecutive:
        logger.debug(f"Dropped {dropped_consecutive} windows (non-consecutive frames)")
    if dropped_quality:
        logger.debug(f"Dropped {dropped_quality} windows (insufficient points)")

    if not windows:
        logger.warning("No windows passed quality filters")
        return np.empty((0, window_size * df.shape[1]), dtype=np.float32)

    result = np.array(windows, dtype=np.float32)
    logger.debug(f"Windowing output: {result.shape}")
    return result


# ==================== Stage 1: Range Bin Data ====================

def process_range_bins(
    df: pd.DataFrame,
    transform_active_func,
    radar_range_bins_bounds: list | None = None,
    get_bounds_pairs_func=None,
    logger=None
) -> pd.DataFrame:
    """
    Stage 1: Select range bin columns (bin*). Optional index range filter.

    Args:
        df: Input DataFrame
        transform_active_func: Callable(token) -> bool to check if this stage is active
        radar_range_bins_bounds: [min_bin, max_bin] bounds, or None for all bins
        get_bounds_pairs_func: Function to parse bounds (handles None values)
        logger: Optional logger

    Returns:
        DataFrame with selected bin columns (empty DataFrame if stage inactive)
    """
    if not transform_active_func(TRANSFORM_RANGE_DATA):
        return pd.DataFrame(index=df.index)

    bin_cols = column_prefix_map(df.columns, 'bin')
    if not bin_cols:
        if logger:
            logger.warning(f"{TRANSFORM_RANGE_DATA!r} active but no 'bin*' columns found")
        return pd.DataFrame(index=df.index)

    bounds = radar_range_bins_bounds
    if bounds is None:
        selected = bin_cols
    else:
        # Use get_bounds_pairs to properly handle None values
        if get_bounds_pairs_func is None:
            from .radar_utils import get_bounds_pairs as get_bounds_pairs_func
        (min_bin, max_bin) = get_bounds_pairs_func(bounds, 1)[0]
        # min_bin or max_bin can be None (no bound)
        selected = [
            c for c in bin_cols
            if (min_bin is None or min_bin <= int(re.match(r'^bin(\d+)$', str(c)).group(1))) and
               (max_bin is None or int(re.match(r'^bin(\d+)$', str(c)).group(1)) <= max_bin)
        ]

    return df[selected].astype(float)


# ==================== Stage 2: Point Cloud Data ====================

def process_point_cloud(
    df: pd.DataFrame,
    track_df: pd.DataFrame,
    transform_active_func,
    radar_point_cloud_data_selector: list | None = None,
    radar_point_cloud_coordinate_system: str = 'absolute',
    radar_point_cloud_relative_coordinate_filter: list | None = None,
    radar_point_cloud_spatial_filter_bounds: list | None = None,
    radar_point_cloud_non_spatial_filter_bounds: list | None = None,
    get_bounds_pairs_func=None,
    bounds_mask_func=None,
    logger=None
) -> pd.DataFrame | None:
    """
    Stage 2: Process point cloud data. Returns long-format DataFrame with columns:
    frame_idx, x, y, z, doppler, snr (only selected features kept).
    One row per (frame, point).

    Args:
        df: Main DataFrame with point_* columns
        track_df: Track DataFrame (for relative coordinate transform)
        transform_active_func: Callable(token) -> bool
        radar_point_cloud_data_selector: List of feature names to keep
        radar_point_cloud_coordinate_system: 'absolute' or 'relative'
        radar_point_cloud_relative_coordinate_filter: Which coords to make relative
        radar_point_cloud_spatial_filter_bounds: [min_x, max_x, min_y, max_y, min_z, max_z]
        radar_point_cloud_non_spatial_filter_bounds: [min_dop, max_dop, min_snr, max_snr]
        get_bounds_pairs_func: get_bounds_pairs function
        bounds_mask_func: bounds_mask function
        logger: Optional logger

    Returns:
        DataFrame with point cloud features or None if inactive
    """
    if not transform_active_func(TRANSFORM_POINT_CLOUD_DATA):
        return None

    point_indices = get_point_indices(df)
    if not point_indices:
        if logger:
            logger.warning(f"{TRANSFORM_POINT_CLOUD_DATA!r} active but no point columns found")
        return None

    selector = radar_point_cloud_data_selector
    # If selector is None, use all available point cloud features
    if selector is None:
        selector = list(POINT_CLOUD_COLUMN_PREFIXES.keys())

    # Build records: one per (frame, point)
    records = []
    for frame_idx in df.index:
        row = df.loc[frame_idx]
        for pt in point_indices:
            point = {'frame_idx': frame_idx}
            for feat, prefix in POINT_CLOUD_COLUMN_PREFIXES.items():
                col = f'{prefix}{pt}'
                if col in df.columns:
                    point[feat] = float(row[col])
                else:
                    point[feat] = np.nan
            # Point exists only if spatial coords are present
            if all(np.isnan(point[k]) for k in ('x', 'y', 'z')):
                continue
            records.append(point)

    if not records:
        return pd.DataFrame(columns=['frame_idx'] + list(POINT_CLOUD_COLUMN_PREFIXES.keys()))

    points_df = pd.DataFrame(records)

    # Coordinate transform: relative subtracts track position
    coord_system = str(radar_point_cloud_coordinate_system).lower()
    rel_filter = radar_point_cloud_relative_coordinate_filter or []

    if coord_system == 'relative':
        if 'x' in rel_filter and 'track_x' in track_df.columns:
            track_x = track_df['track_x'].reindex(points_df['frame_idx']).values
            points_df['x'] = points_df['x'].values - track_x
        if 'y' in rel_filter and 'track_y' in track_df.columns:
            track_y = track_df['track_y'].reindex(points_df['frame_idx']).values
            points_df['y'] = points_df['y'].values - track_y
        if 'z' in rel_filter and 'track_z' in track_df.columns:
            track_z = track_df['track_z'].reindex(points_df['frame_idx']).values
            points_df['z'] = points_df['z'].values - track_z
    elif coord_system != 'absolute':
        raise ValueError(f"radar_point_cloud_coordinate_system must be 'absolute' or 'relative', got {coord_system!r}")

    # Spatial filter (x, y, z) - no-op if bounds unset
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bounds_pairs_func(
        radar_point_cloud_spatial_filter_bounds, 3)
    mask = bounds_mask_func(points_df, [
        ('x', min_x, max_x), ('y', min_y, max_y), ('z', min_z, max_z)
    ])
    points_df = points_df[mask]

    # Non-spatial filter (doppler, snr) - no-op if bounds unset
    (min_dop, max_dop), (min_snr, max_snr) = get_bounds_pairs_func(
        radar_point_cloud_non_spatial_filter_bounds, 2)
    mask = bounds_mask_func(points_df, [
        ('doppler', min_dop, max_dop), ('snr', min_snr, max_snr)
    ])
    points_df = points_df[mask]

    # Keep only selected features
    keep = [c for c in ('x', 'y', 'z', 'doppler', 'snr') if c in selector]
    points_df = points_df[['frame_idx'] + keep].reset_index(drop=True)

    return points_df


# ==================== Stage 3: Track Data ====================

def process_track_data(
    df: pd.DataFrame,
    transform_active_func,
    radar_track_data_selector: list | None = None,
    radar_track_velocity_bounds: list | None = None,
    radar_track_acceleration_bounds: list | None = None,
    radar_track_spatial_filter_bounds: list | None = None,
    get_bounds_pairs_func=None,
    bounds_mask_func=None,
    logger=None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Stage 3: Process track data. Returns (track_df, valid_mask).
    - valid_mask: True for frames with valid track_id (ghost tracks filtered out)
    - track_df: Selected track features per frame

    Args:
        df: Input DataFrame
        transform_active_func: Callable(token) -> bool
        radar_track_data_selector: List of feature group names to select
        radar_track_velocity_bounds: [min_v, max_v]
        radar_track_acceleration_bounds: [min_a, max_a]
        radar_track_spatial_filter_bounds: [min_x, max_x, min_y, max_y, min_z, max_z]
        get_bounds_pairs_func: get_bounds_pairs function
        bounds_mask_func: bounds_mask function
        logger: Optional logger

    Returns:
        (track_df, valid_mask)
    """
    if not transform_active_func(TRANSFORM_TRACK_DATA):
        return pd.DataFrame(index=df.index), pd.Series(True, index=df.index)

    # Ensure required track columns exist (add missing with 0.0)
    required_cols = {
        'track_x', 'track_y', 'track_z',
        'track_velocity_x', 'track_velocity_y', 'track_velocity_z',
        'track_acc_x', 'track_acc_y', 'track_acc_z',
    }
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Ghost track validation: track_id must be integer (ends with .0)
    valid_mask = pd.Series(True, index=df.index)
    track_id_col = next((c for c in df.columns if str(c).lower() in ('track_id', 'tid')), None)
    if track_id_col is not None:
        def is_valid(v):
            try:
                return str(float(v)).endswith('.0')
            except (TypeError, ValueError):
                return False
        valid_mask = df[track_id_col].apply(is_valid)
    else:
        if logger:
            logger.warning(f"{TRANSFORM_TRACK_DATA!r} active but no 'track_id'/'tid' column; skipping ghost validation")

    # Select requested track features
    selector = radar_track_data_selector
    # If selector is None, use all available track features
    if selector is None:
        selector = list(TRACK_COLUMN_PREFIXES.keys())
    selected_cols = []
    for feat in selector:
        selected_cols.extend(TRACK_COLUMN_PREFIXES.get(feat, []))
    selected_cols = [c for c in selected_cols if c in df.columns]

    track_df = df[selected_cols].astype(float) if selected_cols else pd.DataFrame(index=df.index)

    # Velocity bounds filter
    (min_v, max_v) = get_bounds_pairs_func(radar_track_velocity_bounds, 1)[0]
    vel_cols = [c for c in track_df.columns if c.startswith('track_velocity')]
    valid_mask &= bounds_mask_func(track_df, [(c, min_v, max_v) for c in vel_cols])

    # Acceleration bounds filter
    (min_a, max_a) = get_bounds_pairs_func(radar_track_acceleration_bounds, 1)[0]
    acc_cols = [c for c in track_df.columns if c.startswith('track_acc')]
    valid_mask &= bounds_mask_func(track_df, [(c, min_a, max_a) for c in acc_cols])

    # Spatial filter (track position) - uses x, y, z
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bounds_pairs_func(
        radar_track_spatial_filter_bounds, 3)
    valid_mask &= bounds_mask_func(track_df, [
        ('track_x', min_x, max_x),
        ('track_y', min_y, max_y),
        ('track_z', min_z, max_z),
    ])

    return track_df, valid_mask


# ==================== Stage 4: Scaling ====================

def apply_scaling(
    track_df: pd.DataFrame,
    transform_active_func,
    radar_data_scaling_track_velocity: float | None = None,
    scale_columns_func=None
) -> pd.DataFrame:
    """
    Stage 4: Scale track velocity columns if 'Scaling' transform is active.

    Args:
        track_df: Track DataFrame
        transform_active_func: Callable(token) -> bool
        radar_data_scaling_track_velocity: Scale factor
        scale_columns_func: scale_columns function

    Returns:
        Modified track DataFrame
    """
    if not transform_active_func(TRANSFORM_SCALING):
        return track_df
    scale = radar_data_scaling_track_velocity
    return scale_columns_func(track_df, 'track_velocity', scale)