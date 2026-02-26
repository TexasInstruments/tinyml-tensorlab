=================
Anomaly Detection
=================

Anomaly detection identifies patterns that deviate from "normal" behavior using
autoencoder-based models trained only on normal data.

Overview
--------

**What it does**: Learns what "normal" looks like, then flags anything different.

**Key advantage**: Only needs normal data for training - no need to collect examples
of all possible faults.

**Use cases**:

* Equipment health monitoring
* Predictive maintenance
* Fault detection when fault examples are scarce
* Detecting novel/unknown failures

**Real-world applications**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Domain
     - Normal Behavior
     - Anomalies
   * - Predictive Maintenance
     - Healthy motor vibration
     - Bearing wear, imbalance, misalignment
   * - Manufacturing
     - Defect-free products
     - Cracks, scratches, dimensional errors
   * - Medical Diagnosis
     - Healthy vital signs
     - Arrhythmia, abnormal glucose levels
   * - IoT Sensors
     - Expected sensor readings
     - Sensor drift, hardware failure

How It Works
------------

1. **Training**: Autoencoder learns to compress and reconstruct normal data
2. **Inference**: Input is compressed and reconstructed
3. **Decision**: High reconstruction error → Anomaly

.. code-block:: text

   Normal input → Low reconstruction error → "Normal"
   Anomaly input → High reconstruction error → "Anomaly"


Autoencoder Architecture
------------------------

What is an Autoencoder?
^^^^^^^^^^^^^^^^^^^^^^^

An **autoencoder** is a type of neural network designed to learn an efficient
compressed representation of input data. It consists of two main parts:

1. **Encoder**: Compresses the input into a lower-dimensional representation
   called the *latent space*.
2. **Decoder**: Reconstructs the original input from the compressed
   representation.

The training objective is to make the output as similar to the input as possible.
In other words, the autoencoder learns to capture the **essential patterns** in
the data while discarding noise and irrelevant details.

.. code-block:: text

   Input --> Encoder --> Latent Space --> Decoder --> Output
     X         f             z              g          X_hat

   Goal: Minimize ||X - X_hat||^2 (reconstruction error)

Architecture Diagram
^^^^^^^^^^^^^^^^^^^^

The autoencoder used in ModelMaker is composed of stacked **Linear (fully
connected) layers** with **ReLU activations**. The encoder progressively
compresses the input, and the decoder mirrors this structure to expand back
to the original dimensionality.

.. code-block:: text

   +-------------------------------------------------------+
   |                        INPUT                          |
   |          (e.g., 256 samples x 3 channels)             |
   |                    = 768 features                     |
   +-------------------------------------------------------+
                            |
                            v
   +-------------------------------------------------------+
   |                       ENCODER                         |
   |              (Dimensionality Reduction)               |
   |                                                       |
   |    Layer 1: 768 --> 512 features  (Linear + ReLU)     |
   |    Layer 2: 512 --> 256 features  (Linear + ReLU)     |
   |    Layer 3: 256 --> 128 features  (Linear + ReLU)     |
   |    Layer 4: 128 -->  64 features  (Linear + ReLU)     |
   +-------------------------------------------------------+
                            |
                            v
   +-------------------------------------------------------+
   |                    LATENT SPACE                        |
   |        (Compressed representation: 64 features)       |
   |                                                       |
   |      Contains essential patterns from input           |
   |      Compression ratio: 768 / 64 = 12x               |
   +-------------------------------------------------------+
                            |
                            v
   +-------------------------------------------------------+
   |                       DECODER                         |
   |              (Dimensionality Expansion)               |
   |                                                       |
   |    Layer 1:  64 --> 128 features  (Linear + ReLU)     |
   |    Layer 2: 128 --> 256 features  (Linear + ReLU)     |
   |    Layer 3: 256 --> 512 features  (Linear + ReLU)     |
   |    Layer 4: 512 --> 768 features  (Linear, no act.)   |
   +-------------------------------------------------------+
                            |
                            v
   +-------------------------------------------------------+
   |                 RECONSTRUCTED OUTPUT                   |
   |                    768 features                        |
   |          (reshaped to 256 samples x 3 channels)       |
   +-------------------------------------------------------+

Dimensionality Flow
^^^^^^^^^^^^^^^^^^^

The encoder progressively **compresses** the input:

.. code-block:: text

   Input:            768 features  (256 samples x 3 channels)
                      |
   Encoder Layer 1:  512 features
                      |
   Encoder Layer 2:  256 features
                      |
   Encoder Layer 3:  128 features
                      |
   Encoder Layer 4:   64 features
                      |
   LATENT SPACE:      64 features  (8.3% of original size)

The decoder then **expands** back to original dimensions:

.. code-block:: text

   LATENT SPACE:      64 features
                      |
   Decoder Layer 1:  128 features
                      |
   Decoder Layer 2:  256 features
                      |
   Decoder Layer 3:  512 features
                      |
   Decoder Layer 4:  768 features
                      |
   Output:           768 features  (same as input)

The bottleneck at the latent space (64 features) forces the network to learn
compressed representations. The network cannot simply memorize the input -- it
must extract the most important patterns.

.. note::

   The encoder and decoder have a **mirror structure**. Each encoder layer
   has a corresponding decoder layer to "undo" the compression. This symmetry
   ensures balanced architecture and smooth gradient flow during training.

Key Components
^^^^^^^^^^^^^^

**Linear (Fully Connected) Layers** transform features from one dimensional
space to another:

.. code-block:: text

   y = W * x + b

   Where:
     x = Input features
     W = Weight matrix (learned during training)
     b = Bias vector (learned during training)
     y = Output features

Each output feature is a weighted combination of all input features, allowing
the network to learn complex relationships.

**ReLU Activation** introduces non-linearity (enables learning of complex
patterns):

.. code-block:: text

   ReLU(x) = max(0, x)

   If x > 0: output = x
   If x <= 0: output = 0

ReLU is simple and fast to compute, and avoids the vanishing gradient problem.
Without activation functions, the network would be purely linear (just matrix
multiplications).

.. note::

   The final decoder layer has **no activation function**, allowing the output
   to take on negative values and match the full range of the input data.

   Convolutional layers may also be used in place of linear layers. Convolution
   layers capture the local structure of the input and learn to extract features
   across channels.

Training with MSE Loss
^^^^^^^^^^^^^^^^^^^^^^

The autoencoder is trained to minimize the **Mean Squared Error (MSE)** between
the input and the reconstructed output:

.. code-block:: text

   MSE = (1/N) * sum( (input[i] - output[i])^2 )

   Where:
     N         = Total number of elements in the input
     input[i]  = Original value at position i
     output[i] = Reconstructed value at position i

For a 256-sample, 3-channel input: ``N = 256 x 3 = 768 elements``.

Semi-Supervised Learning: Normal Data Only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The autoencoder is trained **exclusively on normal data**. This is the core
principle that enables anomaly detection:

- **During training**: The model learns to compress and reconstruct normal
  patterns with low error. It learns what "normal" looks like.
- **During inference on normal data**: The model has seen similar patterns
  before, so reconstruction error is **low**.
- **During inference on anomaly data**: The model has never seen these patterns,
  so reconstruction error is **high**. The decoder tries to "force fit" the
  unfamiliar latent representation using normal patterns it knows, resulting in
  a distorted output.

.. code-block:: text

   +-------------------------------------------------------+
   |              TRAINING (Normal Data Only)               |
   +-------------------------------------------------------+
                           |
                           v
   +-------------------------------------------------------+
   |  Model learns: "What do normal patterns look like?"   |
   |                                                       |
   |  Normal vibration --> Encode --> Decode --> ~ Input    |
   |  Reconstruction Error: LOW                            |
   +-------------------------------------------------------+
                           |
                           v
   +-------------------------------------------------------+
   |                INFERENCE (Test Data)                   |
   +-------------------------------------------------------+
                           |
               +-----------+-----------+
               |                       |
          Normal sample          Anomaly sample
               |                       |
               v                       v
        Encode --> Decode       Encode --> Decode
               |                       |
               v                       v
        Output ~ Input          Output != Input
               |                       |
               v                       v
        Error: LOW              Error: HIGH
               |                       |
               v                       v
      Prediction: Normal     Prediction: Anomaly


Advantages and Limitations
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Advantages**:

* No need to label anomaly types -- only need normal data
* Generalizes to unseen anomalies -- any deviation from normal triggers high error
* Learns complex, non-linear patterns across multi-dimensional sensor data
* Scalable: fast inference, deployable on edge devices (MCUs)

**Limitations**:

* Requires good normal data that covers the full range of normal operating
  conditions. If training data is narrow, valid conditions may be flagged as
  anomalies.
* Threshold selection is critical -- trade-off between false positives and false
  negatives.
* Cannot distinguish anomaly types -- only outputs "normal" vs "anomaly". For
  fault classification, use a supervised approach.
* Subtle anomalies very similar to normal patterns may be missed if reconstruction
  error does not exceed the threshold.


Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'my_anomaly_data'
     input_data_path: '/path/to/data'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'
     variables: 3

   training:
     model_name: 'AD_4k_NPU'
     training_epochs: 50

   testing: {}
   compilation: {}


Dataset Format for Anomaly Detection
-------------------------------------

Folder Structure
^^^^^^^^^^^^^^^^

For anomaly detection tasks, ModelMaker expects the dataset to be organized with
separate folders for **Normal** and **Anomaly** data:

.. code-block:: text

   dataset_name/
   |
   +-- classes/
   |   +-- Normal/
   |   |   +-- file1.csv
   |   |   +-- file2.csv
   |   |   +-- ...
   |   +-- Anomaly/
   |       +-- file1.csv
   |       +-- file2.csv
   |       +-- ...
   |
   +-- annotations/                          # Optional
       +-- file_list.txt                     # List of all files
       +-- instances_train_list.txt          # Training files (Normal only)
       +-- instances_val_list.txt            # Validation files (Normal only)
       +-- instances_test_list.txt           # Test files (Normal + Anomaly)

.. warning::

   **Training uses ONLY normal data.** The ``Anomaly/`` folder is excluded
   from training entirely. Anomaly data is used **only for testing** to
   evaluate detection performance.

Key points:

* **Normal/ folder**: Contains all samples representing normal operating
  conditions. All training and validation data comes from this folder.
* **Anomaly/ folder**: Contains samples representing anomalous behavior
  (faults, failures, defects). Used ONLY for testing, never for training.
* **annotations/ folder**: Optional. If not provided, ModelMaker automatically
  generates the annotation files based on your ``split_type`` and
  ``split_factor`` settings.
* Multiple anomaly types (e.g., imbalance, damage, bearing wear) all go in
  the same ``Anomaly/`` folder. The model treats them all as "not normal" and
  does not distinguish between different anomaly types.

Concrete Example
^^^^^^^^^^^^^^^^

.. code-block:: text

   fan_blade_ad_dataset.zip/
   |
   +-- classes/
   |   +-- Normal/
   |   |   +-- normal_001.csv
   |   |   +-- normal_002.csv
   |   |   +-- ...
   |   |   +-- normal_100.csv              # 100 normal samples
   |   |
   |   +-- Anomaly/
   |       +-- imbalance_001.csv
   |       +-- imbalance_002.csv
   |       +-- damage_001.csv
   |       +-- obstruction_001.csv
   |       +-- ...
   |       +-- obstruction_005.csv         # 20 anomaly samples (mixed types)
   |
   +-- annotations/                         # Optional - auto-generated if missing
       +-- file_list.txt
       +-- instances_train_list.txt
       +-- instances_val_list.txt
       +-- instances_test_list.txt

Data Splitting Strategy
^^^^^^^^^^^^^^^^^^^^^^^

Normal data is split into train, validation, and test sets. Anomaly data is
used **only** in the test set.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Split
     - Normal Data
     - Anomaly Data
     - Purpose
   * - **Training**
     - 60% of Normal files
     - **None**
     - Learn what "normal" looks like
   * - **Validation**
     - 10% of Normal files
     - **None**
     - Monitor overfitting on normal patterns
   * - **Test**
     - 30% of Normal files
     - **All Anomaly files**
     - Evaluate detection performance

Configure splitting in your YAML:

.. code-block:: yaml

   dataset:
     split_type: 'amongst_files'        # or 'within_files'
     split_factor: [0.6, 0.1, 0.3]      # [train, val, test]

Two split methods are available:

* **amongst_files** (default): Files are divided into train, validation, and
  test sets. For example, with 100 normal files and ``split_factor: [0.6, 0.1, 0.3]``,
  you get 60 files for training, 10 for validation, and 30 for testing.
* **within_files**: Each file is split internally. All files appear in all
  splits with different portions. For example, each file contributes its first
  60% of samples to training, next 10% to validation, and last 30% to testing.

.. note::

   If you do not provide an ``annotations/`` folder, ModelMaker automatically
   generates annotation files. For ``amongst_files``, it randomly splits
   normal files according to the split factor and adds all anomaly files to
   the test list. For ``within_files``, it creates new files for each split
   portion.

Datafile Format (CSV)
^^^^^^^^^^^^^^^^^^^^^

ModelMaker supports ``.csv``, ``.txt``, ``.npy``, and ``.pkl`` file formats.
Within CSV files, two layouts are accepted:

**Headerless format** (no header row, no index column):

.. code-block:: text

   2078
   2136
   2117
   2077
   2029
   1989
   2056

**Headered format** (with column names, optionally with a time/index column):

.. code-block:: text

   Time,Vibration_X,Vibration_Y,Vibration_Z
   0.0000,-2753,-558,64376
   0.0001,-2551,-468,63910
   0.0002,-424,-427,64032
   0.0003,1429,-763,64132

.. warning::

   Any column with the text ``time`` in its header (case-insensitive) is
   **automatically dropped** by ModelMaker. If you have a useful column that
   contains "time" in its name, rename it before using the dataset (e.g.,
   ``uptime_hours`` to ``operation_hours``).

For multi-channel data (e.g., 3-axis accelerometer), specify the number of
variables in your configuration:

.. code-block:: yaml

   data_processing_feature_extraction:
     variables: 3   # X, Y, Z axes

ModelMaker will automatically use the first ``variables`` columns (after
dropping time columns) as input.


Dataset Format
--------------

Same as classification but with specific requirements:

.. code-block:: text

   my_dataset/
   └── classes/
       ├── normal/              # Training data
       │   ├── file1.csv
       │   └── ...
       └── anomaly/             # Test-only data
           ├── file1.csv
           └── ...

**Important**:

* Training uses **only normal class** data
* Anomaly class is only used for testing/validation
* You can have multiple anomaly types in separate folders for testing

Available Models
----------------

**NPU-Optimized Models**:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model Name
     - Parameters
     - Description
   * - ``AD_500_NPU``
     - ~500
     - Small autoencoder
   * - ``AD_1k_NPU``
     - ~1,000
     - Medium
   * - ``AD_4k_NPU``
     - ~4,000
     - Large
   * - ``AD_20k_NPU``
     - ~20,000
     - Very large

**Specialized Models**:

* ``AD_Linear`` - Linear autoencoder
* ``Ondevice_Trainable_AD_Linear`` - On-device training capable


Training Workflow
-----------------

Overview
^^^^^^^^

The typical workflow for training an anomaly detection model consists of:

.. code-block:: text

   1. Prepare Dataset
          |
          v
   2. Configure YAML
          |
          v
   3. Run ModelMaker
          |
          v
   4. Review Results
          |
          v
   5. Deploy

Pipeline Steps in Detail
^^^^^^^^^^^^^^^^^^^^^^^^^

When you run ModelMaker, the following steps are executed in order:

1. **Load Configuration** -- Reads the ``config.yaml`` file specifying dataset
   paths, model selection, training hyperparameters, and target device.

2. **Load Dataset (Normal data only)** -- Downloads or extracts the dataset if
   needed. Creates train/val/test splits (if annotation files are not provided).
   Loads only Normal data for training and validation.

3. **Data Processing and Feature Extraction** -- Applies windowing
   (e.g., SimpleWindow). Optionally applies feature extraction transforms such
   as FFT or binning. Normalizes data.

4. **Train Autoencoder** -- Trains the autoencoder on normal training data using
   MSE loss. Monitors validation loss (on normal validation data). Selects the
   best epoch (lowest validation loss).

5. **Quantization** (if ``quantization: 1`` or ``2``) -- Fine-tunes with
   Quantization-Aware Training (QAT) or Post-Training Quantization (PTQ).
   Prepares model for MCU deployment.

6. **Threshold Calculation** -- Computes reconstruction errors on training data.
   Calculates ``mean_train`` and ``std_train``. Generates thresholds for
   k = 0 to 4.5.

7. **Testing** -- Evaluates model on test set (normal + anomaly). Calculates
   metrics (precision, recall, F1) for each threshold value. Generates
   ``threshold_performance.csv`` with results.

8. **Compilation** -- Exports model for target device (e.g., F28P55). Generates
   artifacts (``mod.a``, header files). Creates golden vectors for on-device
   validation.

Running the Pipeline
^^^^^^^^^^^^^^^^^^^^

Execute the training pipeline with:

.. code-block:: bash

   ./run_tinyml_modelmaker.sh path/to/your/config.yaml

Output Files
^^^^^^^^^^^^

After training completes, ModelMaker generates outputs in:

.. code-block:: text

   data/projects/{dataset_name}/run/{date-time}/{model_name}/

Key output files:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File/Folder
     - Description
   * - ``training/base/``
     - Float model training results
   * - ``training/quantization/``
     - Quantized model results
   * - ``training/quantization/post_training_analysis/``
     - Threshold performance analysis
   * - ``training/quantization/post_training_analysis/threshold_performance.csv``
     - Metrics for k=0 to 4.5
   * - ``training/quantization/post_training_analysis/reconstruction_error_histogram.png``
     - Error distribution plot
   * - ``training/quantization/golden_vectors/``
     - Test vectors and config for on-device validation
   * - ``compilation/artifacts/``
     - Compiled model (``mod.a``, ``tvmgen_default.h``)

Understanding Training Logs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A typical training log output looks like:

.. code-block:: text

   INFO: Best epoch: 39
   INFO: MSE: 1.773

   INFO: Reconstruction Error Statistics:
   INFO: Normal training data - Mean: 1.662490, Std: 1.968127
   INFO: Anomaly test data - Mean: 141.985321, Std: 112.756683
   INFO: Normal test data - Mean: 2.849831, Std: 1.343052

   INFO: Threshold for K = 4.5: 10.519060
   INFO: False positive rate: 0.00%
   INFO: Anomaly detection rate (recall): 100.00%
   INFO: Accuracy: 100.00%
   INFO: Precision: 100.00%
   INFO: F1 Score: 100.00%

   Confusion Matrix:
                        Predicted Normal    Predicted Anomaly
   Actual Normal                    899                     0
   Actual Anomaly                     0                 54900

**What to look for**:

* **Good training**: Validation loss decreases over epochs; training and
  validation loss converge.
* **Good separation**: A large gap between normal mean error and anomaly mean
  error (e.g., ``mean_anomaly >> mean_normal``). This indicates the model can
  distinguish anomalies effectively. Check the
  ``reconstruction_error_histogram.png`` for visual confirmation.
* **Warning signs**: Large gap between training and validation loss (overfitting);
  ``mean_test_normal`` much higher than ``mean_train`` (poor generalization);
  high false positive rate or low recall (threshold tuning needed).


Threshold Selection
-------------------

The threshold determines when reconstruction error indicates an anomaly.

Formula
^^^^^^^

.. code-block:: text

   threshold = mean_train + k * std_train

   Where:
     mean_train = Average reconstruction error on normal training data
     std_train  = Standard deviation of reconstruction errors on normal training data
     k          = Multiplier (ranges from 0 to 4.5)

The formula is based on the assumption that reconstruction errors on normal data
approximately follow a Gaussian distribution. The value of ``k`` represents how
many standard deviations away from the mean the threshold is set.

k-Value Impact
^^^^^^^^^^^^^^

The choice of ``k`` determines the **trade-off between false positives and
false negatives**:

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 25 25

   * - k
     - Threshold Level
     - Normal Coverage (Gaussian)
     - Sensitivity
     - Typical Use Case
   * - 1
     - mean + 1 * std
     - ~84% below threshold
     - **High sensitivity**, many false alarms
     - Safety-critical (aircraft, medical)
   * - 2
     - mean + 2 * std
     - ~97.5% below threshold
     - High sensitivity, some false alarms
     - High-risk industrial monitoring
   * - 3
     - mean + 3 * std
     - ~99.7% below threshold
     - **Balanced** detection and false alarms
     - General industrial monitoring (typical choice)
   * - 4
     - mean + 4 * std
     - ~99.99% below threshold
     - **High specificity**, may miss subtle anomalies
     - Cost-sensitive (avoid unnecessary maintenance)
   * - 4.5
     - mean + 4.5 * std
     - >99.99% below threshold
     - Very high specificity, rare false alarms
     - Extreme cost-of-false-alarm scenarios

**Lower k (e.g., k=1)**:

* High recall -- catches almost all anomalies
* Low precision -- many false positives (normal samples flagged)
* **When to use**: Safety-critical applications where missing an anomaly is
  very costly. Can tolerate false alarms.

**Balanced k (e.g., k=2.5 to 3.0)**:

* Balanced recall and precision
* Good F1 score
* **When to use**: Most general applications. Need to optimize both detection
  rate and false alarm rate.

**Higher k (e.g., k=4)**:

* High precision -- few false positives
* Lower recall -- may miss subtle anomalies
* **When to use**: Cost-sensitive applications where false alarms are expensive
  (e.g., unnecessary production shutdowns).

threshold_performance.csv
^^^^^^^^^^^^^^^^^^^^^^^^^

ModelMaker tests multiple ``k`` values and reports the results in a CSV file
called ``threshold_performance.csv``. This is the most important output for
selecting your deployment threshold.

Example contents:

.. code-block:: text

   k_value,threshold,accuracy,precision,recall,f1_score,false_positive_rate,true_positives,true_negatives,false_positives,false_negatives
   0.0,1.662,98.65,98.65,100.0,99.32,83.54,54900,148,751,0
   1.0,3.631,99.71,99.70,100.0,99.85,18.13,54900,736,163,0
   2.0,5.599,99.92,99.92,100.0,99.96,4.78,54900,856,43,0
   3.0,7.567,99.98,99.98,100.0,99.99,1.11,54900,889,10,0
   4.0,9.535,100.0,100.0,100.0,100.0,0.22,54900,897,2,0
   4.5,10.519,100.0,100.0,100.0,100.0,0.0,54900,899,0,0

**How to use this file**:

1. Identify your priority metric (recall, precision, or F1).
2. Find the row with acceptable performance for your application.
3. Note the ``threshold`` value from that row.
4. Use this threshold value in your deployed application code.

.. note::

   The Gaussian assumption is approximate. In practice, reconstruction error
   distributions may be skewed or have heavy tails. ModelMaker tests multiple
   ``k`` values so you can select based on actual metrics (precision/recall)
   rather than relying strictly on the Gaussian interpretation.

How to Choose k Based on Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Safety-critical systems** (aircraft engines, medical devices, nuclear
  sensors): Use a **lower k** (e.g., k=1 or k=2). Prioritize recall -- catch
  every anomaly, even at the cost of some false alarms.
* **Cost-sensitive systems** (predictive maintenance, manufacturing quality
  control): Use a **higher k** (e.g., k=3 or k=4). Prioritize precision --
  only flag true anomalies to avoid costly false alarms.
* **Balanced systems** (general industrial monitoring, IoT sensor networks):
  Use the k that gives the **best F1 score** (typically k=2.5 to 3.0).

.. warning::

   If training data contains **mislabeled anomalies** (anomalous samples
   incorrectly placed in the Normal/ folder), the computed mean and standard
   deviation will be inflated, resulting in a threshold that is too high. This
   can cause the model to miss real anomalies. Always ensure training data
   is clean and truly normal.


Evaluation Metrics
------------------

Evaluating anomaly detection models requires different considerations than
standard classification. Due to class imbalance (normal samples vastly
outnumber anomalies), accuracy alone can be highly misleading.

.. warning::

   **Accuracy is misleading for anomaly detection.** Consider a test set with
   950 normal samples and 50 anomaly samples. A naive model that always predicts
   "Normal" achieves 95% accuracy while detecting zero anomalies. Always use
   recall, precision, and F1 alongside accuracy.

Confusion Matrix
^^^^^^^^^^^^^^^^

The four possible outcomes for any prediction:

.. code-block:: text

                        Predicted Normal    Predicted Anomaly

   Actual Normal            TN                    FP
                       (True Negative)      (False Positive)

   Actual Anomaly           FN                    TP
                       (False Negative)     (True Positive)

* **True Positive (TP)**: Anomaly correctly detected as anomaly.
* **True Negative (TN)**: Normal sample correctly identified as normal.
* **False Positive (FP)**: Normal sample incorrectly flagged as anomaly
  (false alarm).
* **False Negative (FN)**: Anomaly missed -- not detected (the most
  dangerous outcome in safety-critical applications).

Metric Definitions
^^^^^^^^^^^^^^^^^^

**Precision** (Positive Predictive Value):

.. code-block:: text

   Precision = TP / (TP + FP)

Of all samples flagged as anomalies, what fraction were actually anomalous?
High precision means few false alarms.

**Recall** (Sensitivity / True Positive Rate / Detection Rate):

.. code-block:: text

   Recall = TP / (TP + FN)

Of all actual anomalies, what fraction did we detect? High recall means few
missed anomalies.

**F1-Score** (Harmonic Mean of Precision and Recall):

.. code-block:: text

   F1 = 2 * (Precision * Recall) / (Precision + Recall)

A balanced metric that accounts for both precision and recall. A high F1
requires both precision and recall to be high.

**False Positive Rate (FPR)**:

.. code-block:: text

   FPR = FP / (FP + TN)

What fraction of normal samples were incorrectly flagged as anomalies?

**Accuracy**:

.. code-block:: text

   Accuracy = (TP + TN) / (TP + TN + FP + FN)

Overall correctness across all samples. Use as a supplementary metric only,
not as the primary metric for anomaly detection.

Metric Summary
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 10 20

   * - Metric
     - Formula
     - What It Measures
     - Ideal
     - Priority For
   * - **Recall**
     - TP / (TP + FN)
     - Percentage of anomalies detected
     - High
     - Safety-critical (catch all faults)
   * - **Precision**
     - TP / (TP + FP)
     - Percentage of flagged anomalies that are real
     - High
     - Cost-sensitive (avoid false alarms)
   * - **F1 Score**
     - 2*(P*R)/(P+R)
     - Balance of precision and recall
     - High
     - Balanced applications
   * - **FPR**
     - FP / (FP + TN)
     - Percentage of normals incorrectly flagged
     - Low
     - Production deployment
   * - **Accuracy**
     - (TP+TN) / Total
     - Overall correctness
     - High
     - Supplementary only

.. note::

   **For anomaly detection, Recall is often more important than Precision.**
   In many industrial applications, the cost of missing a real anomaly (e.g.,
   an undetected bearing failure leading to equipment damage) far exceeds the
   cost of a false alarm (e.g., an unnecessary inspection). When in doubt,
   prioritize recall and choose a lower ``k`` value for the threshold.


What If You Don't Have Anomaly Data?
------------------------------------

It is common, especially early in a project, to have collected normal operating
data but no examples of anomalies. This section explains what you can and
cannot do in this scenario.

What You CAN Do
^^^^^^^^^^^^^^^^

* **Train the model**: Training only requires normal data. The autoencoder
  will learn to compress and reconstruct normal patterns with low error.
* **Evaluate on normal test data**: You can calculate the reconstruction error
  distribution on held-out normal data and measure the false positive rate
  (i.e., how many normal samples are incorrectly flagged as anomalies).
* **Set a conservative threshold**: Use the standard statistical threshold
  ``mean_train + 3 * std_train`` as a reasonable starting point. This covers
  approximately 99.7% of normal data under a Gaussian assumption.
* **Deploy and monitor**: Deploy the model in production and monitor the
  reconstruction error in real time. Samples with error exceeding the
  threshold can be flagged for human review or further investigation.

What You CANNOT Do
^^^^^^^^^^^^^^^^^^

* **Measure anomaly detection performance**: Without anomaly samples, you
  cannot calculate recall (there are no anomalies to detect), precision
  (no true positives), or F1 score.
* **Validate that the model actually detects faults**: You can only verify
  that it does not produce false alarms on normal data -- you cannot confirm
  it will catch real anomalies until anomaly data becomes available.

.. note::

   Even without anomaly data, the model is still useful. Once deployed, you
   can collect anomaly samples as they naturally occur in production. These
   can then be used to retroactively evaluate and fine-tune the threshold.
   As anomaly data becomes available, re-run the testing step in ModelMaker
   to get proper metrics and adjust ``k`` accordingly.


Semi-Supervised vs Supervised
-----------------------------

**Use Anomaly Detection (Semi-Supervised) when**:

* You only have normal data available
* Faults are rare and hard to collect
* You want to detect unknown failure modes
* New failure modes may emerge over time
* You need fast time-to-deployment without waiting for faults to occur

**Use Classification (Supervised) when**:

* You have labeled examples of all fault types
* You need to identify specific fault types (not just "something is wrong")
* Faults are well-defined and documented
* You have abundant labeled anomaly data (100+ samples per fault type)

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Semi-Supervised (Anomaly Detection)
     - Supervised (Classification)
   * - Training data
     - Normal only
     - Normal + all anomaly types (labeled)
   * - Labeling effort
     - Low
     - High
   * - Detects unseen anomalies
     - Yes
     - No
   * - Identifies fault type
     - No (just "normal" vs "anomaly")
     - Yes
   * - Accuracy
     - High (if good normal data)
     - High (for known types)
   * - Use case
     - Unknown or evolving faults
     - Known, fixed fault types

Example: Motor Bearing Anomaly
------------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'motor_bearing_ad'
     input_data_path: '/path/to/bearing_data'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'
     variables: 3   # 3-axis vibration

   training:
     model_name: 'AD_4k_NPU'
     training_epochs: 50

Tips
----

* **Include diverse normal conditions** in training data (different loads, speeds)
* **Feature extraction** can help or hurt - try both raw and FFT-based
* **Start with k=3** for threshold, adjust based on false positive rate
* **Monitor reconstruction error distribution** during testing
* **Clean your training data** - mislabeled anomalies in the Normal/ folder
  will distort the threshold calculation
* **Check the histogram** - ``reconstruction_error_histogram.png`` should show
  clear separation between normal and anomaly distributions
* **Use the threshold_performance.csv** to select the optimal threshold for
  your specific application requirements

See Also
--------

* :doc:`timeseries_classification` - Alternative supervised approach
* :doc:`/features/post_training_analysis` - Understanding results
