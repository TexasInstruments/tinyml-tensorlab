======================
Post-Training Analysis
======================

After training, the testing stage evaluates the exported ONNX model on the test data and writes plots and
tables that show how the model behaves. Use them to understand which classes are confused, to choose an
operating threshold and to check what quantization did to the model.

Enabling analysis
=================

The analysis is generated when testing is enabled (it is by default):

.. code-block:: yaml

   testing:
     enable: True

Where the files are
===================

The testing script writes a ``post_training_analysis`` folder next to the model it tested. For a quantized run
that is the ``quantization`` stage:

.. code-block:: text

   .../<model_name>/training/quantization/post_training_analysis/
   |-- One_vs_Rest_MultiClass_ROC_test.png
   |-- Histogram_Class_Score_differences_test.png
   `-- fpr_tpr_thresholds.csv

The files depend on the task type:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task type
     - Files in ``post_training_analysis``
   * - Classification
     - ``One_vs_Rest_MultiClass_ROC_test.png``, ``Histogram_Class_Score_differences_test.png``, ``fpr_tpr_thresholds.csv``
   * - Regression
     - ``results_on_test_set.csv``, ``actual_vs_predicted_test.png``, ``residual_error_test.png``
   * - Anomaly detection
     - ``threshold_performance.csv``, ``reconstruction_error.png``, ``reconstruction_error_log_scale.png``

The accuracy, AUC ROC score and the confusion matrix are printed to the console and written to the run log
(``run.log``). For classification the ``file_level_classification_summary.log`` in the ``training/base`` folder
lists, for each test file, how many samples were predicted as each class.

Confusion matrix
================

The confusion matrix is in the log, with one row per ground truth class and one column per predicted class:

.. code-block:: text

                       Predicted as A   Predicted as B   Predicted as C
   Ground Truth: A          95                3                2
   Ground Truth: B           1               97                2
   Ground Truth: C           2                1               97

* The diagonal holds the correct predictions.
* Large off-diagonal values show which classes are confused, and in which direction.

ROC curves
==========

``One_vs_Rest_MultiClass_ROC_test.png`` shows the trade-off between the true positive rate (sensitivity) and
the false positive rate for each class against all others.

.. figure:: /_static/img/post_training/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC curves

   One-vs-Rest multi-class ROC curves (arc fault example)

The area under the curve (AUC) is 1.0 for a perfect classifier and 0.5 for random guessing.

Class score histograms
======================

``Histogram_Class_Score_differences_test.png`` shows the distribution of the differences between class scores.
It tells you how confident the model is.

.. figure:: /_static/img/post_training/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class score histogram

   Distribution of class score differences

* Well-separated histograms: the model is confident and correct.
* Overlapping histograms: the model is uncertain.
* Wrong predictions at high scores: confident mistakes, which are worth investigating.

FPR/TPR thresholds
==================

``fpr_tpr_thresholds.csv`` lists the ROC points with the score threshold of each (columns: index, ``fpr``, ``tpr``,
``thresholds``). To choose an operating point:

1. Decide what matters more: a low false positive rate or a high true positive rate.
2. Find the row that meets the requirement.
3. Use its threshold in the application code on the device.

Regression outputs
==================

For regression the folder has the predictions for the test set (``results_on_test_set.csv``), a plot of actual
against predicted values and a plot of the residual errors. A good model has points close to the diagonal in
the first and residuals centered on zero in the second. The R2, MSE and SMAPE values are in the log.

Anomaly detection outputs
=========================

For anomaly detection the threshold on the reconstruction error is ``normal_train_mean + k x normal_train_std``.
``threshold_performance.csv`` tries ``k`` from 0 to 4.5 in steps of 0.5 and reports for each threshold the
accuracy, precision, recall, F1 score, false positive rate and the counts of true/false positives and negatives.
The best F1 score selects the threshold that is reported in the log. The reconstruction error plots (linear and
log scale) show the error distributions of the normal and anomalous test data and the chosen threshold.

Float versus quantized
======================

The log reports the results of the float model (``FloatTrain``) and of the quantized model (``QuantTrain``) and of the test
data (``TestData``), so you can see the effect of quantization directly. See :doc:`/features/quantization`.

Next steps
==========

* :doc:`/features/goodness_of_fit`
* :doc:`/features/quantization`
* :doc:`/getting_started/understanding_run_output`
