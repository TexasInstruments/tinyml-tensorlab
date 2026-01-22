================
Goodness of Fit
================

The Goodness of Fit (GoF) test helps you analyze dataset quality and class
separability before investing time in model training.

Overview
--------

GoF testing answers:

* Are my classes separable in feature space?
* Is my feature extraction appropriate?
* Will a neural network be able to learn the patterns?
* Which classes might be confused?

Running GoF tests before training saves time by identifying data or
feature extraction problems early.

Enabling GoF Test
-----------------

Add the GoF section to your configuration:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'your_dataset'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     gof_test: True
     frame_size: 256

   training:
     enable: True  # Can set to False for GoF-only analysis

Running the Test
----------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/your_example/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\\your_example\\config.yaml

Output Files
------------

GoF test generates analysis files:

.. code-block:: text

   .../gof_test/
   ├── gof_pca_2d.png           # PCA visualization
   ├── gof_tsne_2d.png          # t-SNE visualization
   ├── gof_lda_2d.png           # LDA visualization
   ├── class_separability.csv   # Quantitative metrics
   ├── confusion_potential.csv  # Likely confusion pairs
   └── feature_importance.csv   # Important features

Understanding the Visualizations
--------------------------------

**PCA Plot (gof_pca_2d.png)**

Principal Component Analysis projection:

**Example GoF Plots:**

.. figure:: /_static/img/goodness_of_fit/arc_fault_dsi_GoF_frame_size_256.png
   :width: 600px
   :align: center
   :alt: GoF Plot - Arc Fault 256

   GoF analysis for arc fault detection with 256 frame size

.. figure:: /_static/img/goodness_of_fit/arc_fault_dsi_GoF_frame_size_1024.png
   :width: 600px
   :align: center
   :alt: GoF Plot - Arc Fault 1024

   GoF analysis for arc fault detection with 1024 frame size

.. figure:: /_static/img/goodness_of_fit/motor_dsk_GoF_frame_size_256.png
   :width: 600px
   :align: center
   :alt: GoF Plot - Motor Bearing Fault

   GoF analysis for motor bearing fault detection

.. code-block:: text

   PC2
    ^
    |    * * *        Class A
    |  * * * *
    |
    |            + + +     Class B
    |          + + + +
    +-------------------> PC1

* **Well-separated clusters** = Good separability
* **Overlapping clusters** = Potential confusion
* **Scattered points** = High variance, harder to classify

**t-SNE Plot (gof_tsne_2d.png)**

Non-linear dimensionality reduction:

* Better at revealing complex cluster structures
* Preserves local neighborhoods
* May show separability that PCA misses

**LDA Plot (gof_lda_2d.png)**

Linear Discriminant Analysis:

* Maximizes class separation
* Shows best linear separation achievable
* Most relevant for linear-like classifiers

Interpreting Results
--------------------

**Class Separability Score:**

.. code-block:: text

   class_separability.csv:
   class_pair,separability_score,overlap_percentage
   A-B,0.95,2.3%
   A-C,0.82,8.5%
   B-C,0.99,0.1%

* Score > 0.9: Excellent separability
* Score 0.7-0.9: Good separability
* Score 0.5-0.7: Moderate (may need better features)
* Score < 0.5: Poor (investigate data or features)

**Confusion Potential:**

.. code-block:: text

   confusion_potential.csv:
   class_1,class_2,potential_confusion
   A,C,high
   B,D,low

Identifies which classes are most likely to be confused.

8-Plot Analysis
---------------

GoF generates 8 different visualizations using combinations of:

* **Transforms**: PCA, LDA, t-SNE
* **Scalings**: Standard, MinMax
* **Feature sets**: All features, top features

.. code-block:: text

   Plot 1: PCA + Standard scaling + All features
   Plot 2: PCA + MinMax scaling + All features
   Plot 3: LDA + Standard scaling + All features
   Plot 4: LDA + MinMax scaling + All features
   Plot 5: t-SNE + Standard scaling + All features
   Plot 6: t-SNE + MinMax scaling + All features
   Plot 7: PCA + Standard scaling + Top 50 features
   Plot 8: LDA + Standard scaling + Top 50 features

Examining all 8 helps identify the best analysis approach.

Common Patterns
---------------

**Good Dataset:**

.. code-block:: text

   - Tight, well-separated clusters
   - Consistent within-class variance
   - Clear boundaries between classes

**Problematic Dataset:**

.. code-block:: text

   - Overlapping clusters
   - Outliers far from clusters
   - One class scattered, others tight

**Feature Extraction Issue:**

.. code-block:: text

   - All classes overlap completely
   - No structure visible
   - Random-looking scatter

Actionable Insights
-------------------

**If classes overlap:**

1. Try different feature extraction:

   .. code-block:: yaml

      data_processing_feature_extraction:
        # Try FFT instead of raw
        feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'

2. Increase feature count:

   .. code-block:: yaml

      data_processing_feature_extraction:
        feature_extraction_name: 'Generic_512Input_RAW_512Feature_1Frame'

3. Review data labeling for errors

**If one class is scattered:**

1. Check for mislabeled samples
2. Consider splitting into sub-classes
3. Need more training data for that class

**If all classes overlap:**

1. Feature extraction may be inappropriate
2. Data might not contain discriminative information
3. Consider domain expertise for better features

Example: Motor Fault GoF Analysis
---------------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'motor_fault_classification_dsk'

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3
     gof_test: True
     frame_size: 256

   training:
     enable: False  # GoF only, skip training

**Expected Good Results:**

.. code-block:: text

   6 fault classes showing clear separation:
   - Normal: tight cluster, well separated
   - Contaminated: distinct from normal
   - Erosion: some overlap with flaking (similar faults)
   - Flaking: some overlap with erosion
   - No Lubrication: well separated
   - Localized Fault: distinct signature

GoF Without Training
--------------------

Run GoF analysis only (no model training):

.. code-block:: yaml

   data_processing_feature_extraction:
     gof_test: True

   training:
     enable: False

   testing:
     enable: False

   compilation:
     enable: False

This is useful for:

* Rapid dataset evaluation
* Feature extraction comparison
* Data quality assessment

Comparing Feature Extraction
----------------------------

Run GoF with different feature extraction to compare:

**Configuration 1:**

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     gof_test: True

**Configuration 2:**

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_512Input_RAW_512Feature_1Frame'
     gof_test: True

Compare the visualizations to see which gives better separability.

Best Practices
--------------

1. **Always run GoF first**: Before long training runs
2. **Compare multiple feature extractions**: Find the best approach
3. **Investigate overlapping classes**: May need more/different data
4. **Use domain knowledge**: Understand why classes separate (or don't)
5. **Document findings**: GoF results inform model expectations

Limitations
-----------

* GoF is a linear analysis; neural networks can learn non-linear boundaries
* Good GoF doesn't guarantee good model accuracy
* Poor GoF may still yield acceptable models with enough complexity
* 2D projections can hide separability in higher dimensions

Use GoF as a guide, not a definitive answer.

Next Steps
----------

* Learn about :doc:`feature_extraction` options
* See :doc:`post_training_analysis` for model evaluation
* Proceed to training if GoF looks good
