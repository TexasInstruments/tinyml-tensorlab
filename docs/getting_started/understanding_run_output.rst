===============================
Understanding the Run Output
===============================

Every run writes one folder. This page explains what is in it and which files you take to
the device. The examples are from a real run of ``examples/dc_arc_fault/config_dsk.yaml``
(``CLS_1k_NPU``, F28P55).

.. tip::

   A slide version of this page is available:
   `Understanding the Output of a ModelZoo Run <../slides/understanding_run_output.html>`_.

What a Run Produces
-------------------

.. code-block:: text

   data/projects/<dataset_name>/run/<date-time>/<model_name>/
   ├── run.yaml, run.json            # fully resolved parameters of this run
   ├── status.yaml, status.json      # paths and status of every stage
   ├── training/
   │   ├── base/                     # FLOAT model
   │   │   ├── model.onnx, checkpoint.pth
   │   │   ├── run.log               # training log of the whole run
   │   │   ├── golden_vectors/       # test_vector.c, user_input_config.h
   │   │   ├── model_aux.h           # class index -> class name
   │   │   ├── pca_on_feature_extracted_{train,validation}_data.png
   │   │   ├── file_level_classification_summary.log
   │   │   ├── memory_preflight/     # early memory estimate
   │   │   └── <date-time>_<model_name>.zip
   │   └── quantization/             # QUANTIZED model (the one you deploy)
   │       ├── model.onnx, checkpoint.pth
   │       ├── golden_vectors/, model_aux.h, memory_preflight/
   │       └── post_training_analysis/   # ROC, histograms, CSVs
   └── compilation/                  # DEVICE library
       ├── artifacts/mod.a, artifacts/tvmgen_default.h
       ├── run.log                   # compiler log + memory usage
       └── <date-time>_<model_name>_<device>.zip

The location is relative to the folder you ran from, and can be changed with
``common.projects_path`` (see :doc:`running_examples`). The tree differs a little by
task: the PCA plots and the file-level summary are for classification, and the analysis
plots depend on the task.

Float vs. Quantized Model
-------------------------

Training produces a float model first and then a quantized model derived from it.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - File
     - ``training/base/`` (float)
     - ``training/quantization/`` (quantized)
   * - ``model.onnx``
     - Float32 ONNX model (7 KB in this run)
     - Quantized ONNX model (13.7 KB in this run). This is the model that gets compiled.
   * - ``checkpoint.pth``
     - PyTorch weights of the float model
     - PyTorch weights of the quantized model
   * - ``golden_vectors/``
     - Expected outputs as ``float``
     - Expected outputs as ``int8_t``; the config adds ``SKIP_NORMALIZE`` and ``OUTPUT_INT``
   * - ``model_aux.h``
     - Number of classes and class-index-to-name table
     - Same
   * - ``memory_preflight/``
     - An early compile of the model to estimate its memory use on the target before the full run
     - Same
   * - ``run.log``
     - Log of the whole training: float, quantized and test evaluation
     - Not present (it is in ``base``)
   * - PCA plots, ``file_level_classification_summary.log``
     - Feature PCA (train, validation) and per-file predictions
     - Not present
   * - ``post_training_analysis/``
     - Not present
     - Plots and CSVs of the quantized model on the test set
   * - ``<date-time>_<model>.zip``
     - Packages the model, golden vectors and log
     - Not present (the packaged bundle is the compilation zip)

**Which one do I use?** Deploy the quantized model. Compare its accuracy with the float
model in ``training/base/run.log``: a large drop means quantization is hurting the model.

Golden Vectors: user_input_config.h
-----------------------------------

``golden_vectors/user_input_config.h`` holds the feature extraction settings for the
device, generated from your config so that the device reproduces the training-time
preprocessing. Excerpt from ``training/quantization/golden_vectors/`` of the run:

.. code-block:: c

   #define SKIP_NORMALIZE      // added by compilation
   #define OUTPUT_INT          // added by compilation
   #define FE_WIN
   #define FE_FFT
   #define FE_NORMALIZE
   #define FE_BIN
   #define FE_LOG
   #define FE_CONCAT
   #define FE_VARIABLES 1
   #define FE_FRAME_SIZE 1024
   #define FE_FEATURE_SIZE_PER_FRAME 256
   #define FE_NN_OUT_SIZE 2
   #define FE_FFT_STAGES 10
   #define FE_NUM_FRAME_CONCAT 1

* **Which steps run**: ``FE_WIN``, ``FE_FFT``, ``FE_BIN``, ``FE_LOG``, ``FE_CONCAT`` and so on
  mirror the transform list of your feature extraction preset.
* **Sizes and parameters**: frame size 1024 (``FE_FFT_STAGES 10`` = 2\ :sup:`10`), 256
  features per frame, 1 frame, 1 variable, 2 model outputs, plus the binning and log parameters.
* **Quantized vs. float**: the only difference between the two files is the two lines the
  compile step adds. ``SKIP_NORMALIZE`` means the device feature-extraction library
  normalizes and quantizes the features to ``int8_t`` itself (using the bias/scale/shift in
  ``tvmgen_default.h``), and ``OUTPUT_INT`` means integer outputs.

The content depends on the task. A regression run with a simple window defines only the
sizes (such as ``FE_FRAME_SIZE`` and ``FE_NN_OUT_SIZE``).

Golden Vectors: test_vector.c
-----------------------------

``test_vector.c`` contains real samples with the expected result at each stage, to check
your device implementation against the PC.

.. code-block:: c

   // SET 0
   //Class: class_0_normal (Index: 472): ADC Data
   float raw_input_test[1024] = { 2106.0, 2082.0, 1946.0, ... };
   //Class: class_0_normal (Index: 472): Extracted Features
   float model_test_input[256] = { 62.40304, -17.06461, ... };
   //Class: class_0_normal (Index: 472): Expected Model Output
   int8_t golden_output[2] = { 61, -73, };

and ``model_aux.h``:

.. code-block:: c

   const NUMBER_OF_CLASSES = 2;
   const char *classIdToName[NUMBER_OF_CLASSES] = {"class_0_normal", "class_1_arc"};

* ``raw_input_test``: one window of raw sensor (ADC) data.
* ``model_test_input``: the features extracted from it on the PC.
* ``golden_output``: what the model should output for it (``int8_t`` for the quantized
  model, ``float`` for the float model).
* Sets are picked from the training data and cover every class. Leave exactly one set
  uncommented (SET 0 by default).

On the device: run feature extraction on ``raw_input_test`` and compare with
``model_test_input``; run the model and compare with ``golden_output``; use
``classIdToName`` to print the class. The full procedure is in
:doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

PCA Graphs: Are Your Features Separable?
----------------------------------------

``training/base/pca_on_feature_extracted_train_data.png`` and ``..._validation_data.png``
(classification tasks) show every sample after feature extraction, projected onto its 3
principal components and coloured by class. They are computed before the model is trained.

* **Well-separated clusters**: the features already tell the classes apart, so even a
  small model can learn them.
* **Heavily overlapping clusters**: the features may not separate the classes. Try another
  feature extraction preset or frame size before reaching for a larger model.

Only 3 components are shown, so some overlap can still be separable in the full feature
space. Treat it as a quick sanity check.

Post-Training Analysis: Classification
--------------------------------------

``training/quantization/post_training_analysis/`` evaluates the quantized model on the test set.

``One_vs_Rest_MultiClass_ROC_test.png``
   One ROC curve per class, with its AUC. Curves hugging the top-left corner (AUC near 1.0)
   mean the model ranks that class well.

``Histogram_Class_Score_differences_test.png``
   For each pair of classes, the difference between their output scores. Two groups far
   from 0 mean confident decisions; values near 0 are samples the model is unsure about.

``fpr_tpr_thresholds.csv``
   The numbers behind the ROC curve: false positive rate and true positive rate at each
   threshold on the model output. Use it to pick a decision threshold with the false-alarm
   rate your application can accept.

``training/base/file_level_classification_summary.log`` shows which recordings are misclassified.

Post-Training Analysis: Other Tasks
-----------------------------------

Each task type writes the plots that suit it.

Regression
   ``actual_vs_predicted_test.png`` (with R\ :sup:`2`), ``residual_error_test.png`` and
   ``results_on_test_set.csv``.

Anomaly detection
   ``reconstruction_error.png`` (and a ``_log_scale`` version): errors of normal vs. anomaly
   samples and the chosen threshold. ``threshold_performance.csv`` lists accuracy, precision,
   recall, F1 and false positive rate for each threshold.

Forecasting
   Prediction plots and CSVs per target variable in ``training/base/best_epoch_<N>_results/``,
   ``training/quantization/best_epoch_<N>_results/`` and ``test_results/``
   (``prediction_plots/*.png``, ``predictions_csv/*.csv``): one scatter plot per forecast
   step, each with R\ :sup:`2` and SMAPE. Always compare the quantized ``test_results`` with
   the float best epoch.

Compilation Output: What Goes on the Device
-------------------------------------------

``compilation/`` holds the quantized model compiled by TI's Neural Network Compiler for the
``target_device``.

* ``artifacts/mod.a``: the compiled model library. Link it into your project.
* ``artifacts/tvmgen_default.h``: the API of the library. Call ``tvmgen_default_run()``.
  On NPU devices, initialize the NPU first (SysConfig in the C2000 examples; the F29H85
  examples call ``TI_NPU_init()``), then wait for ``tvmgen_default_finished``.
* ``<date-time>_<model>_<device>.zip``: the hand-off bundle with ``artifacts/``,
  ``golden_vectors/``, ``post_training_analysis/`` and ``model_aux.h`` (and a license file).
* ``run.log``: the compiler log. Look for ``TI Model Library Memory Usage (mod.a)``, which
  gives the code, RO data, RW data and total size on the device, and for the
  feature-extraction library memory usage. In the example run the model library total was
  8088 bytes.
* ``*.bin_sub*``: intermediate files the compiler can leave next to the artifacts (seen on
  NPU targets). They are not part of the bundle and you do not need them.

Excerpt of ``tvmgen_default.h``:

.. code-block:: c

   /* Symbol defined when running model on TI NPU hardware accelerator */
   #define TVMGEN_DEFAULT_TI_NPU
   /* Input feature normalization parameters */
   extern const float   tvmgen_default_bias_data[]  = {20.517256};
   extern const float   tvmgen_default_scale_data[] = {2931.};
   extern const int32_t tvmgen_default_shift_data[] = {11};
   extern void TI_NPU_init();
   extern volatile int32_t tvmgen_default_finished;
   int32_t tvmgen_default_run(struct tvmgen_default_inputs* inputs,
                              struct tvmgen_default_outputs* outputs);

Next Step
---------

Copy the bundle into your device project and use ``test_vector.c`` to confirm the device
matches the PC results: :doc:`/DEPLOYING_TO_SDK_EXAMPLE`. Slides:
`Deploying to an SDK Example <../slides/deploying_to_sdk_example.html>`_.
