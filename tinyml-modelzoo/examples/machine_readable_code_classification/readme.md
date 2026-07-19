# Machine Readable Code 28×28 Dataset

## Overview

This dataset is a synthetic low-resolution image classification dataset created for tiny image classification experiments on low-memory and low-power edge devices.

The task is to classify an input image into one of three classes:
- `qr`
- `barcode`
- `other`

Each generated image has the following format:
- 28 x 28 resolution
- 1-channel grayscale
- binary black/white pixels
- .png file format

This makes the dataset similar in size and format to MNIST, but targeted toward visual code classification.

---

## Dataset Purpose

This dataset was created to test a simple image classification application:

> **Machine Readable Code Detection: QR vs Barcode vs Non-Code**

- The goal is **not** to decode QR codes or barcodes.
- The goal is **only** to classify the visual structure of the image.

At 28x28, many generated QR/barcode samples may not be reliably scannable by real decoders.
This is acceptable for this use case because the classifier only needs to learn visual patterns.

---

## Dataset Structure

After running the generation script, the dataset folder looks like this:

```
machine_readable_code/
├── qr/
│ ├── qr_00000.png
│ ├── qr_00001.png
│ └── ...
├── barcode/
│ ├── barcode_00000.png
│ ├── barcode_00001.png
│ └── ...
└── other/
├── other_00000.png
├── other_00001.png
└── ...
```


The default dataset size is:
- 3000 images per class
- 9000 images total

---

## Classes

### `qr`

The `qr` class contains synthetic QR code images.
QR codes are generated using the Python `qrcode` package.
The QR code generation settings are:

```python
version          = 1
error_correction = qrcode.constants.ERROR_CORRECT_L
box_size         = 4
border           = 1
```
A random alphanumeric payload is generated for every QR image.
Example payloads:

```
A7F92K
X91B0QZ
7KLD92A1
```

### `barcode`
The barcode class contains synthetic Code128 barcode images.
Barcodes are generated using the python-barcode package with Code128 and ImageWriter.
The text under the barcode is disabled:

```python
"write_text": False
```

This is intentional because the classifier should learn barcode-like line structure, not printed text.

### `other`
The other class contains non-code / garbage images.
This class intentionally mixes multiple simple image types:

- blank white images
- blank black images
- random binary noise
- random line patterns
- random block patterns

This prevents the other class from being just one trivial pattern.

## Image Conversion
All generated images are passed through the same conversion function:

```python
to_28x28_binary(img)
```
This function performs the following steps:

- Converts the image to grayscale
- Resizes the image to 28x28
- Uses nearest-neighbor resizing to preserve hard black/white edges
- Applies a threshold at 127
- Converts all pixels to either 0 or 255

Final pixel values:

```
0   = black
255 = white
```

## Reproducibility
The script uses a fixed random seed:

```python
SEED = 42
```

Both Python random and NumPy random are seeded:

```python
random.seed(SEED)
np.random.seed(SEED)
```
This makes the generated dataset reproducible across runs, assuming the same package versions are used.

## Dependencies
Install the required packages using:

```bash
python -m pip install "qrcode[pil]" "python-barcode" pillow numpy
```

## How to Generate the Dataset
Run the generation script:

``` bash
python generate_machine_readable_code_28x28.py
```
The script creates the following folder in the current working directory:


```
machine_readable_code/
```

## Recommended TensorLab / ModelMaker Preset
For the clean 28x28 dataset, use a simple preprocessing preset:

```python
VisualCode_28_Default = dict(
    data_processing_feature_extraction=dict(
        variables         = 1,
        image_height      = 28,
        image_width       = 28,
        image_num_channel = 1,
        image_mean        = 0.5,
        image_scale       = 0.5,
        feat_ext_transform=[
            "GRAYSCALE",
            "RESIZE",
        ],
        data_proc_transforms=[],
        augmentation_transform=[],
    ),
    common=dict(
        task_type=TASK_TYPE_IMAGE_CLASSIFICATION,
    ),
)
``` 
Since the generated images are already grayscale, binary, and 28x28, heavy preprocessing is not required.

## Notes and Limitations
This is a synthetic dataset.

This dataset is useful for:

- tiny image classification experiments
- low-memory CNN testing
- NPU / edge-AI preprocessing validation
- QR-vs-barcode visual structure classification

A later version can extend this dataset to 32x32 or 64x64 for better preservation of barcode and QR structure.
