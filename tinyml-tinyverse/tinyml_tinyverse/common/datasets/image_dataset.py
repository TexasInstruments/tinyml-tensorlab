import os
from glob import glob
from pathlib import Path
from logging import getLogger

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def _ensure_tuple_of_floats(val):
    # Handles string, float, int, list, tuple
    if isinstance(val, (float, int)):
        return (float(val),)
    if isinstance(val, str):
        return (float(val),)
    if isinstance(val, (list, tuple)):
        return tuple(float(x) for x in val)
    raise ValueError(f"Unsupported type for mean/std: {type(val)} -> {val}")


class GenericImageDataset(Dataset):
    """
    Expected layout:
        <dataset_dir>/0/*.png
                     /1/*.png
                     ...
                     /9/*.png
    Returns (raw_tensor, normalized_tensor, int_label).
    """

    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        super().__init__()
        self.logger = getLogger("root.GenericImageDataset")

        self.subset = subset or "training"
        self._path = dataset_dir

        # Absorb kwargs 
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.image_height = int(kwargs["image_height"])
        self.image_width =  int(kwargs["image_width"])
        self.image_num_channel = int(kwargs["image_num_channel"])
        self.mean = _ensure_tuple_of_floats(kwargs["image_mean"])
        self.std  = _ensure_tuple_of_floats(kwargs["image_scale"])
       
        # Buffers
        self.classes = []
        self.label_map = {}
        self.inverse_label_map = {}
        self.X_raw, self.X, self.Y = [], [], []

        # Transforms
        self._raw_tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=self.image_num_channel),
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor()
        ])
        self._tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=self.image_num_channel),
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    # ---------- helpers ----------
    def _discover_classes(self):
        """Look for subfolders directly under dataset_dir (0..9)."""
        cands = [d for d in Path(self._path).glob("*") if d.is_dir()]
        self.classes = sorted([d.name for d in cands])
        if not self.classes:
            raise FileNotFoundError(f"No class folders found under {self._path}")

        self.label_map = {c: i for i, c in enumerate(self.classes)}
        self.inverse_label_map = {i: c for c, i in self.label_map.items()}
        return self._path

    def _build_walker(self, root):
        exts = ("*.png", "*.jpg", "*.jpeg")
        all_paths = []
        for c in self.classes:
            for ext in exts:
                all_paths.extend(glob(os.path.join(root, c, ext)))
        return sorted(all_paths)

    # ---------- prepare ----------
    def prepare(self, **kwargs):
        # self.logger.info(f"prepare called on: {self._path}")
        root = self._discover_classes()
        walker = self._build_walker(root)
        self.logger.info(f"[GenericImageDataset] walker length: {len(walker)}")

        if not walker:
            self.logger.warning(f"No images found under {root}/<class>/*.png|jpg")
            raise FileNotFoundError(f"No images found under {root}/<class>/*.png|jpg")

        for path in walker:
            label = Path(path).parent.name
            img = Image.open(path)

            # Log transform calls
            # self.logger.debug(f"Applying raw transform to {path}")
            raw_tensor = self._raw_tfm(img)
            # self.logger.debug(f"Raw tensor shape: {raw_tensor.shape}, dtype: {raw_tensor.dtype}")

            # self.logger.debug(f"Applying normalized transform to {path}")
            norm_tensor = self._tfm(img)
            # self.logger.debug(f"Norm tensor shape: {norm_tensor.shape}, dtype: {norm_tensor.dtype}")

            self.X_raw.append(raw_tensor)
            self.X.append(norm_tensor)
            self.Y.append(self.label_map[label])

        self.X_raw = torch.stack(self.X_raw)
        self.X = torch.stack(self.X)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

        self.logger.info(f"Prepared dataset with {len(self.X)} samples across {len(self.label_map)} classes")

        return self

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_raw[idx], self.X[idx], self.Y[idx]