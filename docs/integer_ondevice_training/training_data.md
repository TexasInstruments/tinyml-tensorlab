# Training Data

## Table of Contents

- [1. Overview](#1-overview)
- [2. train_data.h](#2-train_datah)
- [3. train_data.c: Data Format and Layout](#3-train_datac-data-format-and-layout)
- [4. How Training Data Is Selected](#4-how-training-data-is-selected)
- [5. Why Inputs Are float32](#5-why-inputs-are-float32)
- [6. Using Training Data in main.c](#6-using-training-data-in-mainc)
- [7. Flash and RAM Footprint](#7-flash-and-ram-footprint)
- [8. Limitations and Future Work](#8-limitations-and-future-work)

---

## 1. Overview

`train_data.h` and `train_data.c` contain the embedded dataset used for integer on-device training. These are the samples the model trains on, validates against, and is evaluated with, all stored in flash and available without any external data source.

ModelMaker selects a balanced subset of samples from the training, validation, and test splits of your dataset and exports them as C arrays. The number of samples per split per class is controlled by `export_samples_per_class` in the YAML.

---

## 2. train_data.h

```c
#ifndef TRAIN_DATA_H
#define TRAIN_DATA_H

#include <stdint.h>
#include "integer_model_config.h"   // needed for INPUT_SIZE define

// ── Sample counts ──────────────────────────────────────────────────────────
// Total samples = n_per_class × n_classes
// With export_samples_per_class='[10,5,5]' and 2 classes:
#define N_TRAIN_SAMPLES    20   // 10 per class × 2 = 20 training samples
#define N_VAL_SAMPLES      10   //  5 per class × 2 = 10 validation samples
#define N_TEST_SAMPLES     10   //  5 per class × 2 = 10 test samples

// ── Array declarations ─────────────────────────────────────────────────────
// TRAIN_INPUTS: raw float feature-extracted data, one row per sample
// Each row is INPUT_SIZE floats — the same data that goes into TVM as input
extern const float  TRAIN_INPUTS [N_TRAIN_SAMPLES][INPUT_SIZE];
extern const int8_t TRAIN_LABELS [N_TRAIN_SAMPLES];

// VAL_INPUTS/LABELS: validation set — available for evaluation but not used
extern const float  VAL_INPUTS   [N_VAL_SAMPLES  ][INPUT_SIZE];
extern const int8_t VAL_LABELS   [N_VAL_SAMPLES  ];

// TEST_INPUTS/LABELS: test set — available for post-training evaluation
extern const float  TEST_INPUTS  [N_TEST_SAMPLES ][INPUT_SIZE];
extern const int8_t TEST_LABELS  [N_TEST_SAMPLES ];

#endif // TRAIN_DATA_H
```

### N_TRAIN_SAMPLES, N_VAL_SAMPLES, N_TEST_SAMPLES

The total count across all classes. With `export_samples_per_class='[10,5,5]'` and a 2-class model:
- N_TRAIN_SAMPLES = 10 × 2 = 20
- N_VAL_SAMPLES = 5 × 2 = 10
- N_TEST_SAMPLES = 5 × 2 = 10

---

## 3. train_data.c: Data Format and Layout

### Sample Format

Each row is annotated with its index and label:

```c
const float TRAIN_INPUTS[N_TRAIN_SAMPLES][INPUT_SIZE] = {
    /* [   0] label= 0 */ {  0.01234567f, -0.87654321f,  0.45678901f, -0.23456789f,
                              0.12345678f,  0.34567890f, -0.56789012f,  0.78901234f,
                              ...  /* 128 floats total */ },
    /* [   1] label= 0 */ {  0.23456789f,  0.12345678f, -0.34567890f,  0.45678901f,
                              ...  },
    ...
    /* [  19] label= 1 */ { -0.98765432f,  0.76543210f,  0.65432198f, -0.43210987f,
                              ...  },
};
```

**Shuffled order:** TRAIN_INPUTS is randomly shuffled before writing. Class 0 and class 1 samples are interleaved

### Label Format

```c
const int8_t TRAIN_LABELS[N_TRAIN_SAMPLES] = {
    0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1
};
```

Labels are `int8_t` class indices (0, 1, 2, ...) matching `N_CLASSES`. For a 2-class model, values are only 0 and 1.

`TRAIN_LABELS[i]` is the class of `TRAIN_INPUTS[i]`.

### Validation and Test Sets

Same format as TRAIN_INPUTS/LABELS:
```c
const float  VAL_INPUTS  [N_VAL_SAMPLES ][INPUT_SIZE] = { ... };
const int8_t VAL_LABELS  [N_VAL_SAMPLES ]             = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };

const float  TEST_INPUTS [N_TEST_SAMPLES][INPUT_SIZE] = { ... };
const int8_t TEST_LABELS [N_TEST_SAMPLES]             = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 };
```

