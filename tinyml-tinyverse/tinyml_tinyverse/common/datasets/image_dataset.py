import os
from glob import glob
from pathlib import Path
from logging import getLogger
from ast import literal_eval
import cv2
import yaml
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def _to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(val, (int, float)):
        return bool(val)
    return False


def _ensure_tuple_of_floats(val, n_channels=None):
    """
    Handles:
    - float/int: 0.5
    - numeric string: "0.5"
    - tuple/list: (0.485, 0.456, 0.406)
    - tuple/list string: "(0.485, 0.456, 0.406)" or "[0.485, 0.456, 0.406]"
    """

    if isinstance(val, (float, int)):
        out = (float(val),)

    elif isinstance(val, str):
        val = val.strip()

        # Handle tuple/list string
        if val.startswith("(") or val.startswith("["):
            parsed = literal_eval(val)
            if not isinstance(parsed, (list, tuple)):
                raise ValueError(f"Expected list/tuple string for mean/std, got: {val}")
            out = tuple(float(x) for x in parsed)
        else:
            # Handle normal numeric string like "0.5"
            out = (float(val),)

    elif isinstance(val, (list, tuple)):
        out = tuple(float(x) for x in val)

    else:
        raise ValueError(f"Unsupported type for mean/std: {type(val)} -> {val}")

    # Optional channel expansion/validation
    if n_channels is not None:
        if len(out) == 1 and n_channels > 1:
            out = out * n_channels
        elif len(out) != n_channels:
            raise ValueError(
                f"Expected mean/std of length 1 or {n_channels}, got {len(out)}: {out}"
            )

    return out


def _str2num_if_possible(value):
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


class ResizeWithPad:
    def __init__(self, size, fill=0):
        self.target_h, self.target_w = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == 0 or h == 0:
            raise ValueError(f"Invalid image size: {(w, h)}")

        scale = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        img = img.resize((new_w, new_h), resample)

        canvas = Image.new(img.mode, (self.target_w, self.target_h), color=self.fill)
        left = (self.target_w - new_w) // 2
        top = (self.target_h - new_h) // 2
        canvas.paste(img, (left, top))
        return canvas


class GenericImageDataset(Dataset):
    """
    Base class for all image datasets. 
    - config-driven init
    - file list loading
    - transform/preprocessing pipeline
    - prepare() loop
    - target processing

    Expected layout (default folder-based mode):
        <dataset_dir>/<class_name>/*.png|jpg|jpeg|bmp

    Returns:
        (raw_tensor, normalized_tensor, int_label)
    """

    _logger_name = "root.BaseGenericImageDataset"

    def __init__(self, subset: str = None, dataset_dir: str = None, **kwargs):
        super().__init__()

        self.logger = getLogger(self._logger_name)
        self.subset = subset or "training"
        self._path = dataset_dir

        self.classes = []
        self.label_map = {}
        self.inverse_label_map = {}
        self.image_preprocessing_params = {}
        self.preprocessing_flags = []
        self.X_raw = []
        self.X = []
        self.Y = []
        self.file_names = []

        # absorb kwargs
        for key, value in kwargs.items():
            setattr(self, key, _str2num_if_possible(value))

        # required image params
        self.image_height = int(getattr(self, "image_height"))
        self.image_width = int(getattr(self, "image_width"))
        self.image_num_channel = int(getattr(self, "image_num_channel"))
        self.mean = _ensure_tuple_of_floats(getattr(self, "image_mean"), self.image_num_channel)
        self.std = _ensure_tuple_of_floats(getattr(self, "image_scale"), self.image_num_channel)

        # optional params
        self.pad_value = int(getattr(self, "pad_value", 0))
        self.binary_threshold = int(getattr(self, "binary_threshold", 128))
        self.clahe_clip_limit = float(getattr(self, "clahe_clip_limit", 2.0))
        self.clahe_tile_grid_size = tuple(getattr(self, "clahe_tile_grid_size", (8, 8)))
        self.sobel_mode = getattr(self, "sobel_mode", "magnitude")
        self.sobel_ksize = int(getattr(self, "sobel_ksize", 3))
        self.laplacian_ksize = int(getattr(self, "laplacian_ksize", 3))
        self.horizontal_flip_prob = float(getattr(self, "horizontal_flip_prob", 0.5))
        self.vertical_flip_prob = float(getattr(self, "vertical_flip_prob", 0.5))
        self.random_rotation_deg = float(getattr(self, "random_rotation_deg", 15))

        self.color_jitter_brightness = float(getattr(self, "color_jitter_brightness", 0.10))
        self.color_jitter_contrast = float(getattr(self, "color_jitter_contrast", 0.10))
        self.color_jitter_saturation = float(getattr(self, "color_jitter_saturation", 0.05))
  
        self.feature_extraction_params = {}
        self.preprocessing_flags = []

        raw_feat_ext_transform = getattr(self, "feat_ext_transform", [])
        raw_augmentation_transform = getattr(self, "augmentation_transform", [])
        raw_transforms = getattr(self, "transforms", raw_feat_ext_transform)

        self.transforms = (raw_transforms)
        self.augmentation_transforms = (raw_augmentation_transform)
        self.enable_online_augmentation = False
     
        self.augment_pipeline = []

        if hasattr(self, "augment_config") and self.augment_config:
            if os.path.exists(self.augment_config):
                self.logger.info(f"Parsing {self.augment_config} to form augmentation pipeline")
                try:
                    with open(self.augment_config) as fp:
                        augment_config = yaml.load(fp, Loader=yaml.CLoader)
                    self.augment_pipeline = augment_config
                except yaml.YAMLError as exc:
                    self.logger.critical(f"{exc} error parsing {self.augment_config}")

        self._walker = self._load_file_list(self.subset, kwargs)
        self.classes = self._get_classes()

    def set_augmentation_enabled(self, enabled: bool):
        """
        Enable/disable online augmentation.

        Keep augmentation disabled by default. Training explicitly enables it only
        while iterating over the training dataloader. This keeps validation, export,
        test, and golden-vector generation deterministic.
        """
        self.enable_online_augmentation = (
            bool(enabled)
            and self._is_training_subset()
            and len(self.augmentation_transforms) > 0
        )
    # ==================== File Discovery ====================
    def _prepare_feature_extraction_variables(self):
        self.feature_extraction_params['FE_STACKING_CHANNELS'] = self.image_num_channel
        self.feature_extraction_params['FE_STACKING_FRAME_WIDTH'] = self.image_width
        self.feature_extraction_params['FE_HL'] = self.image_height
        self.feature_extraction_params['FE_NN_OUT_SIZE'] = len(self.classes)
        self.feature_extraction_params.update(self.image_preprocessing_params)

    def _load_file_list(self, subset, kwargs):
        """
        Prefer annotation list if available, otherwise discover files directly.
        Mirrors TS base behavior.
        """
        def load_list(kwargs_list_key, file_pattern):
            joined_path = os.path.join(os.path.dirname(self._path), "annotations", file_pattern)
            candidates = glob(joined_path)
            if kwargs.get(kwargs_list_key):
                list_to_load = kwargs.get(kwargs_list_key)
            elif candidates:
                list_to_load = candidates[0]
            else:
                return None

            walker = []
            with open(list_to_load) as fileobj:
                for line in fileobj:
                    line = line.strip()
                    if line:
                        walker.append(os.path.join(self._path, line))
            return walker

        walker = None
        if subset in ["training", "train"]:
            walker = load_list("training_list", "*train*_list.txt")
        elif subset in ["testing", "test"]:
            walker = load_list("testing_list", "*test*_list.txt") or load_list("testing_list", "*file*_list.txt")
        elif subset in ["validation", "val"]:
            walker = load_list("validation_list", "*val*_list.txt")

        if walker is not None:
            return walker

        # fallback: discover directly under class folders
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        all_paths = []
        for ext in exts:
            all_paths.extend(glob(os.path.join(self._path, "*", ext)))
        all_paths = sorted(all_paths)

        if not all_paths:
            raise FileNotFoundError(
                f"No images found under {self._path}/<class>/*.png|jpg|jpeg|bmp"
            )
        return all_paths

    def _get_classes(self):
        return sorted(set([Path(datafile).parent.name for datafile in self._walker]))
   
    # ==================== Image Prep Variables ====================

    def _prepare_image_variables(self):
 
        self.preprocessing_flags = []

        # Basic image input metadata
        self.image_preprocessing_params["IMG_HEIGHT"] = self.image_height
        self.image_preprocessing_params["IMG_WIDTH"] = self.image_width
        self.image_preprocessing_params["IMG_CHANNELS"] = self.image_num_channel
        self.image_preprocessing_params["IMG_NN_OUT_SIZE"] = len(self.classes)

        # Color mode
        if "RGB" in self.transforms:
            self.preprocessing_flags.append("IMG_RGB")

        if "GRAYSCALE" in self.transforms:
            self.preprocessing_flags.append("IMG_GRAY")

        # Resize / shape transforms
        if "RESIZE" in self.transforms:
            self.preprocessing_flags.append("IMG_RESIZE")

        if "RESIZE_PAD" in self.transforms:
            self.preprocessing_flags.append("IMG_RESIZE_PAD")
            self.image_preprocessing_params["IMG_PAD_VALUE"] = self.pad_value

        # Contrast / intensity transforms
        if "AUTOCONTRAST" in self.transforms:
            self.preprocessing_flags.append("IMG_AUTOCONTRAST")

        if "EQUALIZE" in self.transforms:
            self.preprocessing_flags.append("IMG_EQUALIZE")

        if "INVERT" in self.transforms:
            self.preprocessing_flags.append("IMG_INVERT")

        if "CLAHE" in self.transforms:
            self.preprocessing_flags.append("IMG_CLAHE")
            self.image_preprocessing_params["IMG_CLAHE_CLIP_LIMIT"] = self.clahe_clip_limit
            self.image_preprocessing_params["IMG_CLAHE_TILE_H"] = self.clahe_tile_grid_size[0]
            self.image_preprocessing_params["IMG_CLAHE_TILE_W"] = self.clahe_tile_grid_size[1]

        # Edge filters
        if "SOBEL" in self.transforms:
            self.preprocessing_flags.append("IMG_SOBEL")
            self.image_preprocessing_params["IMG_SOBEL_KSIZE"] = self.sobel_ksize

        if "LAPLACIAN" in self.transforms:
            self.preprocessing_flags.append("IMG_LAPLACIAN")
            self.image_preprocessing_params["IMG_LAPLACIAN_KSIZE"] = self.laplacian_ksize

        # Thresholding
        if "BINARIZE" in self.transforms:
            self.preprocessing_flags.append("IMG_BINARIZE")
            self.image_preprocessing_params["IMG_BINARY_THRESHOLD"] = self.binary_threshold

        # Normalization metadata
        self.preprocessing_flags.append("IMG_NORMALIZE")

        mean = self.mean
        std = self.std

        if len(mean) == 1 and self.image_num_channel > 1:
            mean = mean * self.image_num_channel

        if len(std) == 1 and self.image_num_channel > 1:
            std = std * self.image_num_channel

        for ch in range(self.image_num_channel):
            self.image_preprocessing_params[f"IMG_MEAN_{ch}"] = mean[ch]
            self.image_preprocessing_params[f"IMG_SCALE_{ch}"] = std[ch]
        # ==================== Transform Ops ====================
# ==================== Transform Ops ====================

    def _transform_grayscale(self, img):
        return img.convert("L")

    def _transform_rgb(self, img):
        return img.convert("RGB")

    def _transform_resize(self, img):
        mode = "L" if self.image_num_channel == 1 else "RGB"
        img = img.convert(mode)
        return img.resize((self.image_width, self.image_height))

    def _transform_resize_pad(self, img):
        mode = "L" if self.image_num_channel == 1 else "RGB"
        img = img.convert(mode)
        return ResizeWithPad((self.image_height, self.image_width), fill=self.pad_value)(img)

    def _transform_autocontrast(self, img):
        return ImageOps.autocontrast(img)

    def _transform_equalize(self, img):
        return ImageOps.equalize(img)

    def _transform_invert(self, img):
        if img.mode not in ("L", "RGB"):
            img = img.convert("L" if self.image_num_channel == 1 else "RGB")
        return ImageOps.invert(img)

    def _transform_clahe(self, img):
        img = img.convert("L")
        arr = np.array(img)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        out = clahe.apply(arr)
        return Image.fromarray(out)

    def _transform_sobel(self, img):
        img = img.convert("L")
        arr = np.array(img).astype(np.float32)
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)

        if self.sobel_mode == "x":
            out = np.abs(gx)
        elif self.sobel_mode == "y":
            out = np.abs(gy)
        else:
            out = np.sqrt(gx * gx + gy * gy)

        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def _transform_laplacian(self, img):
        img = img.convert("L")
        arr = np.array(img).astype(np.float32)
        lap = cv2.Laplacian(arr, cv2.CV_32F, ksize=self.laplacian_ksize)
        out = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def _transform_binarize(self, img):
        img = img.convert("L")
        return img.point(lambda p: 255 if p >= self.binary_threshold else 0)

    def _transform_random_horizontal_flip(self, img):
        if not self._is_training_subset():
            return img

        return transforms.RandomHorizontalFlip(
            p=self.horizontal_flip_prob
        )(img)


    def _transform_random_vertical_flip(self, img):
        if not self._is_training_subset():
            return img

        return transforms.RandomVerticalFlip(
            p=self.vertical_flip_prob
        )(img)


    def _transform_random_rotation(self, img):
        if not self._is_training_subset():
            return img

        return transforms.RandomRotation(
            degrees=self.random_rotation_deg,
            fill=0
        )(img)


    def _transform_color_jitter(self, img):
        if not self._is_training_subset():
            return img

        return transforms.ColorJitter(
            brightness=self.color_jitter_brightness,
            contrast=self.color_jitter_contrast,
            saturation=self.color_jitter_saturation,
            hue=self.color_jitter_hue
        )(img)

    def _apply_preprocess_transforms(self, img):
        """
        Applies preprocessing in the exact order listed in self.transforms.
        Returns one processed PIL image shared by both raw and normalized outputs.
        """
        for tfm in self.transforms:
            if tfm == "GRAYSCALE":
                img = self._transform_grayscale(img)
            elif tfm == "RGB":
                img = self._transform_rgb(img)
            elif tfm == "RESIZE":
                img = self._transform_resize(img)
            elif tfm == "RESIZE_PAD":
                img = self._transform_resize_pad(img)
            elif tfm == "AUTOCONTRAST":
                img = self._transform_autocontrast(img)
            elif tfm == "EQUALIZE":
                img = self._transform_equalize(img)
            elif tfm == "INVERT":
                img = self._transform_invert(img)
            elif tfm == "CLAHE":
                img = self._transform_clahe(img)
            elif tfm == "SOBEL":
                img = self._transform_sobel(img)
            elif tfm == "LAPLACIAN":
                img = self._transform_laplacian(img)
            elif tfm == "BINARIZE":
                img = self._transform_binarize(img)
            else:
               self.logger.warning(f"Unknown image transform: {tfm}")
    
        return img

    def _is_training_subset(self):
          return self.subset.lower() in ("training", "train")

    def _apply_augmentation_transforms(self, img):
        if img is None:
            raise ValueError("Input image to augmentation pipeline is None")

        if not self._is_training_subset():
            return img

        for tfm in self.augmentation_transforms:
            if tfm == "RANDOM_HORIZONTAL_FLIP":
                img = self._transform_random_horizontal_flip(img)

            elif tfm == "RANDOM_VERTICAL_FLIP":
                img = self._transform_random_vertical_flip(img)

            elif tfm == "RANDOM_ROTATION":
                img = self._transform_random_rotation(img)

            elif tfm == "COLOR_JITTER":
                img = self._transform_color_jitter(img)

            else:
                self.logger.warning(f"Unknown image augmentation transform: {tfm}")

            if img is None:
                raise ValueError(f"Augmentation transform returned None: {tfm}")

        return img
    # ==================== Loading / Storage ====================

    def _load_datafile(self, datafile):
        """
        Returns: (processed_img, label, raw_img_before_norm)
        """
        img = Image.open(datafile)
        label = Path(datafile).parent.name

        processed_img = self._apply_preprocess_transforms(img)

        # Ensure final mode if transforms did not explicitly set it
        if self.image_num_channel == 1 and processed_img.mode != "L":
            processed_img = processed_img.convert("L")
        elif self.image_num_channel == 3 and processed_img.mode != "RGB":
            processed_img = processed_img.convert("RGB")

        return processed_img, label, processed_img.copy()

    def _to_raw_tensor(self, img):
        return transforms.ToTensor()(img)

    def _to_norm_tensor(self, img):
        x = transforms.ToTensor()(img)
        x = transforms.Normalize(self.mean, self.std)(x)
        return x

    def _store_sample(self, raw_tensor, norm_tensor, label, datafile):
        self.X_raw.append(raw_tensor)
        self.X.append(norm_tensor)
        self.Y.append(label)
        self.file_names.append(datafile)

    def _process_targets(self):
        self.label_map = {k: v for v, k in enumerate(self.classes)}
        self.inverse_label_map = {k: v for v, k in self.label_map.items()}
        self.Y = [self.label_map[y] for y in self.Y]

        if not len(self.Y):
            self.logger.error("No image data could be loaded.")
            raise Exception("No image data could be loaded.")
# ==================== Main Preparation ====================

    def prepare(self, **kwargs):
        self._prepare_image_variables()

        for datafile in self._walker:
            try:
                self._process_datafile(datafile)
            except ValueError as v:
                self.logger.warning(f"Skipping file due to error: {datafile} : {v}")
            except KeyboardInterrupt:
                raise

        if not len(self.X):
            raise Exception("Aborting run as the dataset loaded is empty.")

        self.X_raw = torch.stack(self.X_raw)
        self.X = torch.stack(self.X)

        self._process_targets()
        self.Y = torch.tensor(self.Y, dtype=torch.long)
        self._prepare_feature_extraction_variables()
        return self

    def _process_datafile(self, datafile):
        processed_img, label, raw_img = self._load_datafile(datafile)

        raw_tensor = self._to_raw_tensor(raw_img)
        norm_tensor = self._to_norm_tensor(processed_img)

        self._store_sample(raw_tensor, norm_tensor, label, datafile)

    # ==================== Dataset API ====================

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.Y[index]

        if not self.enable_online_augmentation:
            return self.X_raw[index], self.X[index], label

        img = transforms.ToPILImage()(self.X_raw[index])
        img = self._apply_augmentation_transforms(img)

        raw_tensor = self._to_raw_tensor(img)
        norm_tensor = self._to_norm_tensor(img)

        return raw_tensor, norm_tensor, label