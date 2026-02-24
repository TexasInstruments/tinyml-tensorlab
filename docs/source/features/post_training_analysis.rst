======================
Post-Training Analysis
======================

After training, Tiny ML Tensorlab provides comprehensive analysis tools
to evaluate model performance and understand its behavior.

Overview
--------

Post-training analysis helps you:

* Evaluate model accuracy and error patterns
* Understand which classes are confused
* Select optimal operating thresholds
* Verify quantization impact
* Generate reports for stakeholders

Enabling Analysis
-----------------

Analysis is enabled through the testing section:

.. code-block:: yaml

   testing:
     enable: True
     analysis:
       confusion_matrix: True
       roc_curve: True
       class_histograms: True
       error_analysis: True

Output Files
------------

After testing, you'll find analysis outputs:

.. code-block:: text

   .../testing/
   ├── confusion_matrix_test.png        # Confusion matrix
   ├── One_vs_Rest_MultiClass_ROC_test.png  # ROC curves
   ├── Histogram_Class_Score_differences_test.png  # Score distributions
   ├── fpr_tpr_thresholds.csv           # Threshold analysis
   ├── classification_report.txt        # Per-class metrics
   ├── error_samples/                   # Misclassified examples
   │   ├── error_001.csv
   │   └── ...
   └── test_results.json                # Summary statistics

Confusion Matrix
----------------

Shows classification results in matrix form:

.. code-block:: text

                    Predicted
                    A    B    C
   Actual    A    95    3    2
             B     1   97    2
             C     2    1   97

**Interpreting:**

* Diagonal = correct predictions
* Off-diagonal = misclassifications
* Rows sum to actual class counts
* Columns show predicted distribution

**Good matrix:**

* Strong diagonal (high values)
* Weak off-diagonal (low values)

**Problem indicators:**

* High off-diagonal values = specific class confusion
* Asymmetric confusion = direction-specific errors

ROC Curves
----------

Receiver Operating Characteristic shows trade-off between:

* True Positive Rate (sensitivity)
* False Positive Rate (1 - specificity)

**Example ROC Curves:**

.. figure:: /_static/img/post_training/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC Curves

   One-vs-Rest Multi-class ROC curves for arc fault detection

The ROC curve shows the trade-off between sensitivity and specificity at different thresholds.

.. code-block:: text

   TPR (Sensitivity)
   1.0 |        ******
       |      **
       |    **
   0.5 |  **
       | **
   0.0 +*-------------- FPR
       0.0    0.5    1.0

**Key Metrics:**

* **AUC (Area Under Curve)**: 1.0 = perfect, 0.5 = random
* **Operating Point**: Where you set the threshold

**Multi-Class ROC:**

For multi-class problems, one-vs-rest ROC shows each class:

.. code-block:: text

   Class A: AUC = 0.98
   Class B: AUC = 0.95
   Class C: AUC = 0.99

Class Score Histograms
----------------------

Shows distribution of model confidence for each class:

**Example Class Score Histogram:**

.. figure:: /_static/img/post_training/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class Score Histogram

   Distribution of class score differences showing model confidence

.. code-block:: text

   Correct predictions: [=====|=====] centered at high score
   Wrong predictions:   [==|==] centered at low score

**Interpretation:**

* **Well-separated histograms**: Model is confident and correct
* **Overlapping histograms**: Model is uncertain
* **Wrong predictions at high scores**: Confident mistakes (investigate)

FPR/TPR Thresholds
------------------

CSV file for threshold selection:

.. code-block:: text

   threshold,tpr,fpr,precision,recall,f1
   0.1,0.99,0.15,0.87,0.99,0.93
   0.3,0.97,0.08,0.92,0.97,0.94
   0.5,0.95,0.03,0.97,0.95,0.96
   0.7,0.90,0.01,0.99,0.90,0.94
   0.9,0.80,0.00,1.00,0.80,0.89

**Using this data:**

1. Choose your priority (minimize FPR or maximize TPR)
2. Find the threshold that meets your requirement
3. Use that threshold in deployment code

Classification Report
---------------------

Per-class performance metrics:

.. code-block:: text

   Class     Precision  Recall  F1-Score  Support
   Normal    0.98       0.96    0.97      500
   Fault_A   0.95       0.97    0.96      480
   Fault_B   0.97       0.95    0.96      520

   Accuracy: 0.96
   Macro Avg: 0.97    0.96    0.96      1500
   Weighted Avg: 0.96 0.96    0.96      1500

**Metrics explained:**

* **Precision**: Of predicted positives, how many are correct?
* **Recall**: Of actual positives, how many were detected?
* **F1-Score**: Harmonic mean of precision and recall
* **Support**: Number of samples per class

Error Analysis
--------------

Detailed examination of misclassified samples:

.. code-block:: yaml

   testing:
     error_analysis:
       save_errors: True
       max_errors_per_class: 20

**Error sample files:**

Each saved error includes:

* Original input data
* True label
* Predicted label
* Model confidence scores

**Using error analysis:**

1. Identify patterns in errors
2. Check for labeling mistakes
3. Find data collection issues
4. Improve feature extraction

Quantized vs Float Comparison
-----------------------------

Compare quantized model to float baseline:

.. code-block:: yaml

   testing:
     enable: True
     test_float: True
     test_quantized: True
     compare_results: True

**Output:**

.. code-block:: text

   Float32 Model:
   Accuracy: 99.2%
   F1-Score: 0.992

   INT8 Quantized Model:
   Accuracy: 98.8%
   F1-Score: 0.988

   Degradation: 0.4%

File-Level Classification Summary
----------------------------------

The File-Level Classification Summary provides an overview of how samples from
each input file are classified into different classes. It helps users quickly
identify if any particular file contains misclassified samples.

While the confusion matrix shows overall counts of correct and incorrect
classifications, it does not reveal which specific files contain those
misclassified samples. For example, even if the total misclassification count
is small, it might come entirely from one problematic file. This feature helps
pinpoint such cases instantly.

**Output Location:**

The summary is written to ``file_level_classification_summary.log`` inside the
``training/base/`` directory of your project run:

.. code-block:: text

   .../data/projects/{dataset_name}/run/{date-time}/{model_name}/training/base/
   └── file_level_classification_summary.log

The log file contains tables for float train, quantized train, and test data,
depending on which stages are enabled in the configuration. Each table shows
each file, its true class, and the count of samples from that file classified
into each class.

**Example: Fan Blade Fault Classification**

Consider a fan blade fault classification dataset with four classes: **Normal**,
**BladeDamage**, **BladeImbalance**, and **BladeObstruction**. The confusion
matrix for float train best epoch might look like this:

.. list-table:: Confusion Matrix (Float Train)
   :header-rows: 1
   :widths: 25 18 18 20 18

   * - Ground Truth
     - BladeDamage
     - BladeImbalance
     - BladeObstruction
     - Normal
   * - **BladeDamage**
     - 1159
     - 339
     - 0
     - 0
   * - **BladeImbalance**
     - 0
     - 1301
     - 0
     - 0
   * - **BladeObstruction**
     - 0
     - 0
     - 962
     - 0
   * - **Normal**
     - 0
     - 0
     - 0
     - 2114

From this confusion matrix, we can see that while all classes other than
BladeDamage are correctly classified, some BladeDamage samples are incorrectly
classified as BladeImbalance. However, from the confusion matrix alone, we
cannot determine which specific files contain these misclassified samples.

When we inspect the File-Level Classification Summary of FloatTrain, we discover
that in file numbers 0, 1, 2, 20, and 21, **all** samples were classified as
BladeImbalance even though their true class is BladeDamage. Similarly, in the
test data, file numbers 7 and 8 have all samples misclassified.

.. tip::

   A higher count of samples in the wrong class column for a specific file
   indicates potential data or labeling issues in that file.

**Use Cases:**

* **Identifying labeling issues**: Files where all samples are misclassified
  may have been assigned the wrong label during data collection.
* **Data quality assessment**: Pinpoint specific recordings or data files that
  contain noisy, corrupted, or otherwise problematic data.
* **Targeted investigation**: Rather than reviewing the entire dataset, focus
  review efforts on the specific files flagged by this summary.

Regression Analysis
-------------------

For regression tasks, different metrics apply:

.. code-block:: yaml

   testing:
     enable: True
     regression_metrics:
       mse: True
       mae: True
       r2: True
       scatter_plot: True

**Output:**

.. code-block:: text

   Mean Squared Error (MSE): 0.023
   Mean Absolute Error (MAE): 0.12
   R² Score: 0.95
   Max Error: 0.45

Anomaly Detection Analysis
--------------------------

For anomaly detection:

.. code-block:: yaml

   testing:
     enable: True
     anomaly_metrics:
       reconstruction_error: True
       threshold_analysis: True

**Output:**

.. code-block:: text

   Normal Data:
   Mean reconstruction error: 0.05
   Std reconstruction error: 0.02

   Anomaly Data:
   Mean reconstruction error: 0.35
   Std reconstruction error: 0.15

   Recommended threshold: 0.15
   At threshold 0.15:
   TPR: 0.92
   FPR: 0.05

Custom Analysis Scripts
-----------------------

For advanced analysis, use the saved model and data:

.. code-block:: python

   import torch
   import numpy as np

   # Load model
   model = torch.load('path/to/best_model.pt')
   model.eval()

   # Load test data
   test_data = np.load('path/to/test_data.npy')
   test_labels = np.load('path/to/test_labels.npy')

   # Run inference
   with torch.no_grad():
       outputs = model(torch.tensor(test_data))
       predictions = outputs.argmax(dim=1)

   # Custom analysis
   # ... your analysis code

Generating Reports
------------------

For documentation or stakeholder communication:

.. code-block:: yaml

   testing:
     enable: True
     generate_report: True
     report_format: 'pdf'  # or 'html', 'markdown'

**Report includes:**

* Model summary (architecture, parameters)
* Training curves
* Test metrics
* Confusion matrix
* ROC curves
* Recommendations

Example: Complete Analysis Configuration
----------------------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'CLS_4k_NPU'
     training_epochs: 30
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

   testing:
     enable: True
     test_float: True
     test_quantized: True
     analysis:
       confusion_matrix: True
       roc_curve: True
       class_histograms: True
       error_analysis: True
       save_errors: True
       max_errors_per_class: 10
     compare_results: True

Best Practices
--------------

1. **Always review confusion matrix**: Understand error patterns
2. **Check ROC curves**: Ensure good class separation
3. **Analyze errors**: Learn from misclassifications
4. **Compare quantized**: Verify acceptable accuracy drop
5. **Document findings**: Record analysis for future reference

Troubleshooting Low Accuracy
----------------------------

**If overall accuracy is low:**

* Check GoF test results (dataset quality)
* Try larger model
* Increase training epochs
* Improve feature extraction

**If specific classes have low accuracy:**

* Check class balance
* Investigate error samples
* May need more data for those classes
* Classes might be inherently similar

**If quantized accuracy drops significantly:**

* Try QAT instead of PTQ
* Use more calibration data
* Keep sensitive layers at higher precision
* Use larger model (more robust to quantization)

Next Steps
----------

* Deploy model: :doc:`/deployment/ccs_integration`
* Optimize further: :doc:`neural_architecture_search`
* Review :doc:`/troubleshooting/common_errors` if issues arise
