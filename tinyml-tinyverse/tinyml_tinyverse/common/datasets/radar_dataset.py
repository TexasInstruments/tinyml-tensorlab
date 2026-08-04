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


class GenericRadarDataset(Dataset):

    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        self.logger = getLogger("root.GenericRadarDataset")
        
        self._path = dataset_dir
        self.classes = list()
        self.label_map = dict()
        self.inverse_label_map = dict()
        self.feature_extraction_params = dict()
        self.X = []
        self.Y = []
        self.file_names = []

        # Store all the kwargs in self
        for key, value in kwargs.items():
            value = str2int_or_float(value)
            setattr(self, key, value)

        if not hasattr(self, 'frame_size'):
            self.frame_size = 8

        self._walker = self._load_file_list(subset, kwargs)
        self.classes = self._get_classes()

    # ==================== Data Loading ====================

    def _load_file_list(self, subset, kwargs):
        """Load the list of data files based on subset (train/test/val)."""
        data_dir_name = os.path.basename(self._path)  # usually 'classes'

        def load_list(kwargs_list, file_name_pattern):
            ann_dir = os.path.join(os.path.dirname(self._path), 'annotations')
            matches = glob(os.path.join(ann_dir, file_name_pattern))
            # Also handle double-extension files (e.g. instances_train_list.txt.txt)
            if not matches:
                matches = glob(os.path.join(ann_dir, file_name_pattern + '.txt'))
            if not matches:
                return None  # signal to fall back to directory scan

            list_to_load = matches[0]
            if kwargs.get(kwargs_list):
                list_to_load = kwargs.get(kwargs_list)

            loadList = []
            with open(list_to_load) as fileObj:
                for line in fileObj:
                    line = line.strip().replace('\\', '/')
                    if not line:
                        continue
                    # Annotation paths may be relative to dataset root (e.g. classes/falling/f.csv)
                    # or relative to dataset_dir (e.g. falling/f.csv). Strip the data_dir prefix
                    # if present so we always join from self._path correctly.
                    if line.startswith(data_dir_name + '/'):
                        line = line[len(data_dir_name) + 1:]
                    loadList.append(os.path.join(self._path, line))
            return loadList

        def load_or_fallback(kwargs_list, file_name_pattern):
            result = load_list(kwargs_list, file_name_pattern)
            if result is not None:
                return result
            self.logger.warning(
                f"No annotation file matching '{file_name_pattern}' found in "
                f"{os.path.join(os.path.dirname(self._path), 'annotations')}. "
                f"Falling back to scanning {self._path} directly."
            )
            return self._scan_class_files(subset)

        if subset in ["training", "train"]:
            return load_or_fallback('training_list', '*train*_list.txt')
        elif subset in ["testing", "test"]:
            result = load_list('testing_list', '*test*_list.txt')
            if result is None:
                result = load_list('testing_list', '*file*_list.txt')
            if result is None:
                result = self._scan_class_files(subset)
            return result
        elif subset in ["validation", "val"]:
            return load_or_fallback('validation_list', '*val*_list.txt')

    def _scan_class_files(self, subset):
        """
        Fallback when annotation files are absent: scan self._path for class subdirs
        and split files deterministically by subset.
        """
        if not os.path.isdir(self._path):
            self.logger.error(f"dataset_dir does not exist: {self._path}")
            return []

        all_files = []
        for cls_dir in sorted(os.listdir(self._path)):
            cls_path = os.path.join(self._path, cls_dir)
            if not os.path.isdir(cls_path):
                continue
            for fname in sorted(os.listdir(cls_path)):
                fpath = os.path.join(cls_path, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in ('.csv', '.npy'):
                    all_files.append(fpath)

        if not all_files:
            self.logger.error(f"No CSV/npy files found under {self._path}")
            return []

        n = len(all_files)
        train_end = max(1, int(n * 0.8))
        val_end = max(train_end + 1, int(n * 0.9))

        if subset in ("training", "train"):
            return all_files[:train_end]
        elif subset in ("validation", "val"):
            return all_files[train_end:val_end]
        else:  # test
            return all_files[val_end:] if val_end < n else all_files[train_end:val_end]

    def _get_classes(self):
        """Extract classes from the dataset. Override in subclasses if needed."""
        return sorted(set([opb(opd(datafile)) for datafile in self._walker]))

    # ==================== Data Loading ====================

    def _load_datafile(self, datafile):
        """
        Load a single CSV/npy file and return a float DataFrame.
        Drops session/id/time/classification columns that are not features.
        """
        file_extension = ops(datafile)[-1]

        if file_extension == '.npy':
            return pd.DataFrame(np.load(datafile).astype(float))

        # Always read without consuming any row as a header. This avoids the bug
        # where files without a real header row have their first data row treated
        # as column labels — causing pd.concat to produce a NaN-padded union of
        # all unique first-row values instead of aligning on column position.
        df = pd.read_csv(datafile, encoding='utf-8', engine='python', header=None)

        def _is_numeric(val):
            try:
                float(str(val))
                return True
            except (ValueError, TypeError):
                return False

        # If the first row is non-numeric, it's a real text header. Promote it to
        # column names, drop the row from data, then remove non-feature columns.
        if not all(_is_numeric(v) for v in df.iloc[0]):
            df.columns = df.iloc[0].astype(str)
            df = df.iloc[1:].reset_index(drop=True)
            drop_cols = [c for c in df.columns
                         if any(tok in str(c).lower() for tok in
                                ['session', 'recording', 'time', 'track', 'tid', 'classification'])]
            df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return df.astype(float)

    # ==================== Windowing ====================

    def _create_windows(self, df):
        df = df.reset_index(drop=True)
        windows = []
        for i in range(len(df) - self.frame_size + 1):
            window = df.loc[i: i + self.frame_size - 1, :]
            x = window.unstack().to_frame().T.values.flatten().tolist()
            if getattr(self, 'window_sort', False):
                x.sort()
            windows.append(x)
        return np.array(windows, dtype=np.float32)

    # ==================== Preparation ====================

    def prepare(self, **kwargs):
        # Group DataFrames by class
        class_dfs = {cls: [] for cls in self.classes}
        for datafile in tqdm(self._walker):
            try:
                label = opb(opd(datafile))
                df = self._load_datafile(datafile)
                if len(df) > 0:
                    class_dfs[label].append(df)
                self.file_names.append(datafile)
            except Exception as e:
                self.logger.warning(f"Skipping {datafile}: {e}")

        # Concatenate per class
        class_data = {}
        for cls, dfs in class_dfs.items():
            if dfs:
                class_data[cls] = pd.concat(dfs, ignore_index=True)

        if not class_data:
            raise Exception("No data loaded. Check dataset_dir and annotation lists.")

        # Normalize feature width: different captures may have different numbers of
        # columns. Truncate all classes to the narrowest column count so
        # the windowed arrays can be concatenated into a single tensor.
        min_cols = min(df.shape[1] for df in class_data.values())
        if len(set(df.shape[1] for df in class_data.values())) > 1:
            self.logger.warning(
                f"Feature width varies across classes. Truncating all to {min_cols} columns."
            )
        class_data = {cls: df.iloc[:, :min_cols] for cls, df in class_data.items()}

        if getattr(self, 'equalise_data_lengths', False):
            min_len = min(len(df) for df in class_data.values())
            class_data = {cls: df.head(min_len) for cls, df in class_data.items()}

        # Build label map (class name -> integer index, matches notebook's class_data dict)
        self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

        X_list = []
        y_list = []
        for cls, df in class_data.items():
            windows = self._create_windows(df)
            if len(windows) == 0:
                self.logger.warning(f"No windows created for class '{cls}' — file may be shorter than window_size={self.window_size}")
                continue
            X_list.append(windows)
            y_list.extend([self.label_map[cls]] * len(windows))

        if not X_list:
            raise Exception("No windowed samples produced. Reduce window_size or add more data.")

        X = np.concatenate(X_list, axis=0)
        y = np.array(y_list, dtype=np.float32)

        # X: (N, WINDOW_SIZE * FEATURE_COUNT)  float
        # Y: (N,)                               long  (class indices for CrossEntropyLoss)
        self.X = torch.from_numpy(X).type(torch.float)
        self.Y = torch.from_numpy(y).type(torch.long)

        self.preprocessing_flags = []
        self.feature_extraction_params['FE_FRAME_SIZE'] = self.frame_size

        return self

    # ==================== Dataset Interface ====================

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Return (raw_sequence, processed_sequence, label)
        # For radar dataset, raw and processed are the same since we don't
        # maintain separate raw/processed streams like time series data
        return self.X[index], self.X[index], self.Y[index]


