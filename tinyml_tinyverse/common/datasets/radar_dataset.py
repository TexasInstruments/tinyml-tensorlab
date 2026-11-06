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

import os
from glob import glob
from logging import getLogger

from os.path import basename as opb
from os.path import dirname as opd
from os.path import splitext as ops

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.misc_utils import str2int_or_float
from ..utils.radar_utils import (
    # Constants
    TRANSFORM_RANGE_DATA,
    TRANSFORM_POINT_CLOUD_DATA,
    TRANSFORM_TRACK_DATA,
    TRANSFORM_SCALING,
    TRANSFORM_SIMPLE_WINDOW,
    # Helpers
    parse_list_kwarg,
    transform_active,
    get_bounds_pairs,
    bounds_mask,
    column_prefix_map,
    is_numeric,
    promote_header_if_present,
    select_frame_points,
    scale_columns,
    build_frame_table,
    transform_simple_window,
    # Stage functions
    process_range_bins,
    process_point_cloud,
    process_track_data,
    apply_scaling,
)


class GenericRadarDataset(Dataset):
    """
    GenericRadarDataset: PyTorch Dataset for mmWave Radar sensors Applications
    Supports Range, Point Cloud Only & Point Cloud + Tracks Classification
    Future Scope: Range-Doppler Heatmap, Range-Azimuth Heat Map and other Radar Specific Inputs

    Processing Pipeline (stages supported by data_proc_transforms):
    1. "Radar Range Data"       -> Range bin selection (bin* columns)
    2. "Radar Point Cloud Data" -> Point cloud filtering/transform (point_* columns)
    3. "Radar Track Data"       -> Track validation & feature selection (track_* columns)
    4. "Scaling"                -> Velocity scaling (independent of stage 3)
    5. "Simple Window"          -> Frame quality index + sliding window

    Each CSV file = one recording session (Recommended)

    Returns:
    (input_feature_tensor_raw, input_feature_tensor_processed, label_id)
    """

    # nn_for_feature_extraction is disabled for Radar Today - In future
    # input_feature_tensor_raw will be made in use

    # Make a note: Column Names are kept Standardized in order to make sure
    # the link between Radar mmWave Data Recorder/Visualizer and Tensorlab
    # is modular.

    # Pipeline stage identifiers (imported from radar_utils for consistency)
    TRANSFORM_RANGE_DATA      = TRANSFORM_RANGE_DATA
    TRANSFORM_POINT_CLOUD_DATA = TRANSFORM_POINT_CLOUD_DATA
    TRANSFORM_TRACK_DATA      = TRANSFORM_TRACK_DATA
    TRANSFORM_SCALING         = TRANSFORM_SCALING
    TRANSFORM_SIMPLE_WINDOW   = TRANSFORM_SIMPLE_WINDOW

    _logger_name = "root.GenericRadarDataset"

    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        self.logger = getLogger(self._logger_name)
        self._path = dataset_dir
        self.classes = []
        self.label_map = {}
        self.feature_extraction_params = {}
        self.preprocessing_flags = []
        self.X = []
        self.Y = []
        self.file_names = []

        # Store all kwargs as attributes (with type conversion)
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

        list_kwargs = [
            'radar_point_cloud_data_selector',
            'radar_point_cloud_relative_coordinate_filter',
            'radar_track_data_selector',
            'data_proc_transforms',
            'radar_range_bins_bounds',
            'radar_point_cloud_spatial_filter_bounds',
            'radar_point_cloud_non_spatial_filter_bounds',
            'radar_track_velocity_bounds',
            'radar_track_acceleration_bounds',
            'radar_track_spatial_filter_bounds',
        ]
        for kw in list_kwargs:
            parse_list_kwarg(self, kw)

        if subset is not None:
            # Load file list based on subset
            self._walker = self._load_file_list(subset, kwargs)

            # Get classes from the dataset
            self.classes = self._get_classes()

    # ==================== Radar Data Processing Helper Functions ====================

    def _transform_active(self, token: str) -> bool:
        """Check if a pipeline stage is enabled via data_proc_transforms."""
        return transform_active(getattr(self, 'data_proc_transforms', []), token)

    def _get_bounds_pairs(self, attr_name: str, n_pairs: int) -> list[tuple[float | None, float | None]]:
        """Read consolidated bounds list [min1, max1, min2, max2, ...] as (min, max) tuples."""
        value = getattr(self, attr_name, None)
        return get_bounds_pairs(value, n_pairs)

    @staticmethod
    def _bounds_mask(df: pd.DataFrame, column_bounds: list[tuple[str, float | None, float | None]]) -> pd.Series:
        """Build boolean mask using radar_utils.bounds_mask."""
        return bounds_mask(df, column_bounds)

    @staticmethod
    def _column_prefix_map(columns, prefix: str) -> list[str]:
        """Return columns starting with `prefix` + numeric suffix, sorted by suffix."""
        return column_prefix_map(columns, prefix)

    @staticmethod
    def _is_numeric(val) -> bool:
        return is_numeric(val)

    def _promote_header_if_present(self, df: pd.DataFrame) -> pd.DataFrame:
        """If first row is non-numeric, promote to column headers."""
        return promote_header_if_present(df)

    @staticmethod
    def _select_frame_points(
        frame_points: pd.DataFrame,
        max_points: int = 8,
        strategy: str = 'highest and lowest',
        sort_by: str = 'z'
    ) -> tuple[pd.DataFrame, int]:
        """Select up to max_points_per_frame points for a single frame."""
        return select_frame_points(frame_points, max_points, strategy, sort_by)

    @staticmethod
    def _scale_columns(df: pd.DataFrame, prefix: str, scale_factor: float | None) -> pd.DataFrame:
        """Multiply all columns starting with `prefix` by `scale_factor`. No-op if None."""
        return scale_columns(df, prefix, scale_factor)

    # ==================== Data Loading ====================

    def _load_datafile(self, datafile: str) -> pd.DataFrame:
        """Load a single radar capture file as DataFrame.
        Supports .npy, .pkl, .csv, .txt.
        """
        ext = ops(datafile)[-1].lower()

        if ext == '.npy':
            df = pd.DataFrame(np.load(datafile).astype(float))
            self.logger.debug(f"Loaded NPY {datafile}: shape {df.shape}")
            return df

        if ext == '.pkl':
            df = pd.read_pickle(datafile)
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f".pkl file {datafile} does not contain a DataFrame")
            df.columns = [str(c) for c in df.columns]
            self.logger.debug(f"Loaded PKL {datafile}: shape {df.shape}")
            return df

        if ext in ('.csv', '.txt'):
            df = pd.read_csv(datafile, encoding='utf-8', engine='python', header=None)
            df = promote_header_if_present(df)
            self.logger.debug(f"Loaded {ext.upper()[1:]} {datafile}: shape {df.shape}")
            return df

        raise ValueError(f"Unsupported file extension {ext!r} for {datafile}")

    def _load_file_list(self, subset, kwargs):
        """Load the list of data files based on subset (train/test/val)."""
        def load_list(kwargs_list, file_name):
            joined_path = os.path.join(os.path.dirname(self._path), 'annotations', file_name)
            list_to_load = glob(joined_path)[0]
            if kwargs.get(kwargs_list):
                list_to_load = kwargs.get(kwargs_list)
            loadList = []
            with open(list_to_load) as fileobj:
                for line in fileobj:
                    line = line.strip()
                    if line:
                        loadList.append(os.path.join(self._path, line))
            return loadList

        if subset in ["training", "train"]:
            kwargs_list_key = 'training_list'
            list_filename = '*train*_list.txt'
        elif subset in ["testing", "test"]:
            kwargs_list_key = 'testing_list'
            try:
                list_filename = '*test*_list.txt'
                return load_list(kwargs_list_key, list_filename)
            except IndexError:
                list_filename = '*file*_list.txt'
                return load_list(kwargs_list_key, list_filename)
        elif subset in ["validation", "val"]:
            kwargs_list_key = 'validation_list'
            list_filename = '*val*_list.txt'

        return load_list(kwargs_list_key, list_filename)

    def _get_classes(self):
        """Extract classes from the dataset. Override in subclasses if needed."""
        return sorted(set([opb(opd(datafile)) for datafile in self._walker]))

    # ==================== Radar Data Processing Functions ====================

    def _process_range_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 1: Select range bin columns (bin*). Optional index range filter."""
        return process_range_bins(
            df,
            self._transform_active,
            getattr(self, 'radar_range_bins_bounds', None),
            get_bounds_pairs_func=get_bounds_pairs,
            logger=self.logger
        )

    def _process_point_cloud(self, df: pd.DataFrame, track_df: pd.DataFrame) -> pd.DataFrame | None:
        """Stage 2: Process point cloud data."""
        return process_point_cloud(
            df, track_df, self._transform_active,
            radar_point_cloud_data_selector=getattr(self, 'radar_point_cloud_data_selector', None),
            radar_point_cloud_coordinate_system=str(getattr(self, 'radar_point_cloud_coordinate_system', 'absolute')).lower(),
            radar_point_cloud_relative_coordinate_filter=getattr(self, 'radar_point_cloud_relative_coordinate_filter', []),
            radar_point_cloud_spatial_filter_bounds=getattr(self, 'radar_point_cloud_spatial_filter_bounds', None),
            radar_point_cloud_non_spatial_filter_bounds=getattr(self, 'radar_point_cloud_non_spatial_filter_bounds', None),
            get_bounds_pairs_func=get_bounds_pairs,
            bounds_mask_func=bounds_mask,
            logger=self.logger
        )

    def _process_track_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Stage 3: Process track data. Returns (track_df, valid_mask)."""
        return process_track_data(
            df, self._transform_active,
            radar_track_data_selector=getattr(self, 'radar_track_data_selector', None),
            radar_track_velocity_bounds=getattr(self, 'radar_track_velocity_bounds', None),
            radar_track_acceleration_bounds=getattr(self, 'radar_track_acceleration_bounds', None),
            radar_track_spatial_filter_bounds=getattr(self, 'radar_track_spatial_filter_bounds', None),
            get_bounds_pairs_func=get_bounds_pairs,
            bounds_mask_func=bounds_mask,
            logger=self.logger
        )

    def _apply_scaling(self, track_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 4: Scale track velocity columns if 'Scaling' transform is active."""
        return apply_scaling(
            track_df, self._transform_active,
            radar_data_scaling_track_velocity=getattr(self, 'radar_data_scaling_track_velocity', None),
            scale_columns_func=scale_columns
        )

    def _build_frame_table(
        self,
        range_df: pd.DataFrame,
        track_df: pd.DataFrame,
        points_df: pd.DataFrame | None,
        valid_mask: pd.Series
    ) -> tuple[pd.DataFrame, list[int] | None, list[int] | None]:
        """
        Combine range/track/point features into per-frame feature table.
        Returns (frame_table, frame_point_counts, frame_indices).
        Note: windowing is applied later, per-class, in prepare() -- not here.
        """
        return build_frame_table(
            range_df, track_df, points_df, valid_mask,
            max_points_per_frame=int(getattr(self, 'radar_max_points_per_frame', 8)),
            min_points_per_frame=int(getattr(self, 'radar_min_points_per_frame', 0)),
            point_cloud_sort_by=str(getattr(self, 'radar_point_cloud_sort_by', 'z')),
            point_cloud_structure_strategy=str(getattr(self, 'radar_point_cloud_structure_strategy', 'highest and lowest')),
            radar_point_cloud_data_selector=getattr(self, 'radar_point_cloud_data_selector', None),
            quality_mode=str(getattr(self, 'radar_frames_quality_index', '')),
        )

    # ==================== Per-File Processing ====================

    def _process_datafile(self, datafile: str) -> tuple[pd.DataFrame, list[int] | None, list[int] | None, int]:
        """
        Returns: (frame_table, frame_point_counts, frame_indices, frames_dropped_per_frame_quality)
        """
        df = self._load_datafile(datafile)
        if df.empty:
            return pd.DataFrame(), None, None, 0

        # Remove session_id column if present (not used in pipeline)
        if 'session_id' in df.columns:
            df = df.drop(columns=['session_id'])

        range_df = self._process_range_bins(df)
        track_df, valid_mask = self._process_track_data(df)
        track_df = self._apply_scaling(track_df)
        points_df = self._process_point_cloud(df, track_df)

        # Check if any processing stage produced features
        has_radar_features = (
            len(range_df.columns) > 0 or
            len(track_df.columns) > 0 or
            points_df is not None
        )

        # Support for pre-refined datasets: if only Simple Window is active
        # and no radar features, assume the input data IS already the frame table
        simple_window_only = (
            self._transform_active(self.TRANSFORM_SIMPLE_WINDOW) and
            not has_radar_features
        )

        # Pass-through mode: if no transforms are active, just use raw data frame by frame
        passthrough_mode = (
            not self._transform_active(self.TRANSFORM_RANGE_DATA) and
            not self._transform_active(self.TRANSFORM_POINT_CLOUD_DATA) and
            not self._transform_active(self.TRANSFORM_TRACK_DATA) and
            not self._transform_active(self.TRANSFORM_SCALING) and
            not has_radar_features
        )

        if not has_radar_features and not simple_window_only and not passthrough_mode:
            self.logger.warning(
                f"No features extracted from {datafile}. "
                f"Enable one of: {self.TRANSFORM_RANGE_DATA!r}, {self.TRANSFORM_POINT_CLOUD_DATA!r}, {self.TRANSFORM_TRACK_DATA!r}"
            )
            return pd.DataFrame(), None, None, 0

        if simple_window_only or passthrough_mode:
            mode_name = "simple_window_only" if simple_window_only else "passthrough"
            frame_table = df.astype(float).reset_index(drop=True)
            if frame_table.isna().any().any():
                self.logger.warning(f"Input data contains NaN values in {mode_name} mode. Filling with 0.")
                frame_table = frame_table.fillna(0.0)
            valid_mask = pd.Series(True, index=df.index)

            # Apply per_radar_frame quality filtering even in pass-through/simple_window_only mode
            quality_mode = str(getattr(self, 'radar_frames_quality_index', '')).lower()
            use_per_frame_quality = (quality_mode == 'per_radar_frame')

            if use_per_frame_quality:
                # For pre-cleaned data, we don't have point cloud info to count real points
                # User must provide point count via a column or we skip quality check
                # points_per_frame can be added later
                min_points = int(getattr(self, 'radar_min_points_per_frame', 0))
                if min_points > 0:
                    self.logger.warning(
                        f"per_radar_frame quality requested (min_points={min_points}) but "
                        f"{mode_name} mode has no point cloud data to count real points. "
                        f"Quality filtering SKIPPED. Provide point data or use full pipeline."
                    )
            frame_point_counts = None
            frame_indices = list(df.index)
            dropped_per_frame = 0
        else:
            frame_table, frame_point_counts, frame_indices = self._build_frame_table(
                range_df, track_df, points_df, valid_mask
            )

        # Count frames dropped by per-frame quality (those not in frame_table)
        raw_valid = int(valid_mask.sum())
        kept = len(frame_table)
        dropped_per_frame = raw_valid - kept

        return frame_table, frame_point_counts, frame_indices, dropped_per_frame

    # ==================== Preparation (Main Entry) ====================

    def prepare(self, **kwargs):
        """Process all files, build windows, create final tensors."""
        stats = {
            'total_files': len(self._walker),
            'loaded_files': 0,
            'skipped_files': 0,
            'total_raw_frames': 0,
            'frames_after_filters': 0,
            'frames_dropped_per_frame_quality': 0,
            'windows_dropped_per_window_quality': 0,
            'total_windows_before_quality': 0,
            'per_class_raw_frames': {},
            'per_class_windows': {},
        }

        # Group per-frame DataFrames by class
        class_dfs = {cls: [] for cls in self.classes}
        class_point_counts = {cls: [] for cls in self.classes}
        class_frame_indices = {cls: [] for cls in self.classes}

        for datafile in tqdm(self._walker, desc="Processing files"):
            try:
                label = opb(opd(datafile))
                frame_table, pt_counts, f_indices, dropped = self._process_datafile(datafile)

                if len(frame_table) > 0:
                    class_dfs[label].append(frame_table)
                    if pt_counts is not None:
                        class_point_counts[label].extend(pt_counts)
                    if f_indices is not None:
                        class_frame_indices[label].extend(f_indices)
                    stats['loaded_files'] += 1
                else:
                    stats['skipped_files'] += 1

                stats['frames_dropped_per_frame_quality'] += dropped
                self.file_names.append(datafile)

            except Exception as e:
                self.logger.warning(f"Skipping {datafile}: {e}")
                stats['skipped_files'] += 1

        # Concatenate per class
        class_data = {}
        for cls, dfs in class_dfs.items():
            if dfs:
                class_data[cls] = pd.concat(dfs, ignore_index=True)
                stats['per_class_raw_frames'][cls] = len(class_data[cls])
                stats['total_raw_frames'] += len(class_data[cls])

        if not class_data:
            raise RuntimeError(
                f"No data loaded from {stats['total_files']} files. "
                f"Check dataset_dir, annotation lists, and data_proc_transforms config."
            )

        # Normalize feature width across classes (truncate to minimum)
        feature_widths = [df.shape[1] for df in class_data.values()]
        if len(set(feature_widths)) > 1:
            min_cols = min(feature_widths)
            self.logger.warning(f"Feature width varies {feature_widths}; truncating to {min_cols}")
            class_data = {cls: df.iloc[:, :min_cols] for cls, df in class_data.items()}
            # Also truncate auxiliary arrays
            for cls in class_data:
                n = len(class_data[cls])
                class_point_counts[cls] = class_point_counts[cls][:n]
                class_frame_indices[cls] = class_frame_indices[cls][:n]
        else:
            min_cols = feature_widths[0]

        # Build label maps
        self.label_map = {cls: i for i, cls in enumerate(self.classes)}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

        quality_mode = str(getattr(self, 'radar_frames_quality_index', '')).lower()

        use_per_window_quality = (
            self._transform_active(self.TRANSFORM_SIMPLE_WINDOW)
            and quality_mode == 'per_window'
        )

        # Check if windowing should be applied (Simple Window transform active)
        apply_windowing = self._transform_active(self.TRANSFORM_SIMPLE_WINDOW)
        window_size = getattr(self, 'frame_size', None)

        X_list, y_list = [], []

        for cls, frame_table in class_data.items():
            pt_counts = class_point_counts[cls] if class_point_counts[cls] else None
            f_indices = class_frame_indices[cls] if class_frame_indices[cls] else None

            # Track frames after all filtering (before windowing)
            stats['frames_after_filters'] += len(frame_table)

            if apply_windowing:
                # Apply windowing (sliding windows)
                if use_per_window_quality and pt_counts is not None:
                    possible = max(0, len(frame_table) - window_size + 1)
                    stats['total_windows_before_quality'] += possible
                    windows = transform_simple_window(
                        frame_table,
                        frame_size=window_size,
                        stride_size=getattr(self, 'stride_size', 1.0),
                        radar_min_points_per_window=int(getattr(self, 'radar_min_points_per_window', 0)),
                        radar_frames_quality_index=quality_mode,
                        frame_point_counts=pt_counts,
                        frame_indices=f_indices,
                        radar_feature_ordering=str(getattr(self, 'radar_feature_ordering', 'frame_major')),
                        logger=self.logger
                    )
                    stats['windows_dropped_per_window_quality'] += (possible - len(windows))
                else:
                    windows = transform_simple_window(
                        frame_table,
                        frame_size=window_size,
                        stride_size=getattr(self, 'stride_size', 1.0),
                        radar_min_points_per_window=int(getattr(self, 'radar_min_points_per_window', 0)),
                        radar_frames_quality_index=quality_mode,
                        frame_point_counts=pt_counts,
                        frame_indices=f_indices,
                        radar_feature_ordering=str(getattr(self, 'radar_feature_ordering', 'frame_major')),
                        logger=self.logger
                    )
                if len(windows) == 0:
                    self.logger.warning(f"No windows for class '{cls}' (window_size={window_size})")
                    continue
            else:
                # Pass-through mode: each frame is a sample (no windowing)
                windows = frame_table.values.astype(np.float32)
                # Note: per_radar_frame quality already applied in _build_frame_table
                # if radar_frames_quality_index == 'per_radar_frame'

            X_list.append(windows)
            y_list.extend([self.label_map[cls]] * len(windows))
            stats['per_class_windows'][cls] = len(windows)

        if not X_list:
            raise RuntimeError("No windows produced. Reduce window_size or add more data.")

        X = np.concatenate(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)

        # Final tensors: X = [N, window_size * features_per_frame], Y = [N]
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(y).long()

        self.logger.debug(f"Final tensors: X={self.X.shape}, Y={self.Y.shape}")

        # Pipeline summary
        apply_windowing = self._transform_active(self.TRANSFORM_SIMPLE_WINDOW)
        self._log_pipeline_summary(stats, window_size, min_cols, quality_mode, apply_windowing)
        return self

    def _log_pipeline_summary(self, stats, window_size, features_per_frame, quality_mode, apply_windowing=True):
        """Log comprehensive pipeline summary."""
        self.logger.info("=" * 70)
        self.logger.info("RADAR DATA PROCESSING PIPELINE SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Files: {stats['loaded_files']} loaded, {stats['skipped_files']} skipped (of {stats['total_files']})")

        self.logger.info("-" * 70)
        self.logger.info("PER-CLASS RAW FRAMES (after all filtering, before windowing):")
        for cls in self.classes:
            self.logger.info(f"  {cls:12s}: {stats['per_class_raw_frames'].get(cls, 0):6d}")

        self.logger.info("-" * 70)
        if apply_windowing:
            self.logger.info("WINDOWING:")
            self.logger.info(f"  Window size (frame_size): {window_size}")
            self.logger.info(f"  Features per frame:       {features_per_frame}")
            self.logger.info(f"  Input per window:         {window_size * features_per_frame}")

            if quality_mode == 'per_radar_frame':
                self.logger.info(f"  Quality: per_radar_frame (min_points/frame={getattr(self, 'radar_min_points_per_frame', 0)})")
            elif quality_mode == 'per_window':
                self.logger.info(f"  Quality: per_window (min_points/window={getattr(self, 'radar_min_points_per_window', 0)})")
            else:
                self.logger.info("  Quality: none")
        else:
            self.logger.info("PASS-THROUGH MODE (no windowing):")
            self.logger.info(f"  Features per frame:       {features_per_frame}")
            self.logger.info(f"  Each frame = 1 sample")
            if quality_mode == 'per_radar_frame':
                self.logger.info(f"  Quality: per_radar_frame (min_points/frame={getattr(self, 'radar_min_points_per_frame', 0)})")
            elif quality_mode == 'per_window':
                self.logger.info(f"  Quality: per_window (requires Simple Window transform)")
            else:
                self.logger.info("  Quality: none")

        self.logger.info("-" * 70)
        self.logger.info("PER-CLASS WINDOWS:")
        total_w = sum(stats['per_class_windows'].values())
        for cls in self.classes:
            cnt = stats['per_class_windows'].get(cls, 0)
            pct = (cnt / total_w * 100) if total_w else 0
            self.logger.info(f"  {cls:12s}: {cnt:6d} ({pct:.1f}%)")

        self.logger.info("-" * 70)
        self.logger.info("FINAL TENSORS:")
        if apply_windowing:
            self.logger.info(f"  X: {tuple(self.X.shape)}  [N, {window_size * features_per_frame}]")
        else:
            self.logger.info(f"  X: {tuple(self.X.shape)}  [N, {features_per_frame}]")
        self.logger.info(f"  Y: {tuple(self.Y.shape)}  [N]")
        self.logger.info(f"  Total samples: {len(self.X)}")

        if hasattr(self, 'split_factor') and self.split_factor:
            tr, va, te = self.split_factor
            self.logger.info("-" * 70)
            self.logger.info("DATA SPLIT (approx from annotation lists):")
            self.logger.info(f"  Train: {tr:.0%}  Val: {va:.0%}  Test: {te:.0%}")
            self.logger.info(f"  Expected: Train~{int(total_w*tr)}  Val~{int(total_w*va)}  Test~{int(total_w*te)}")

        self.logger.info("=" * 70)
        self.pipeline_stats = stats

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (raw_sequence, processed_sequence, label)
        # For now raw == processed (will be different when nn for feature extraction for radar will be enabled)
        return self.X[index], self.X[index], self.Y[index]