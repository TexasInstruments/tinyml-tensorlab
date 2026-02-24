=======================
Getting Started (GUI)
=======================

This guide walks you through training your first model using Edge AI Studio
Model Composer's graphical interface.

Prerequisites
-------------

Before starting:

1. TI account (create at https://www.ti.com)
2. Modern web browser (Chrome, Firefox, Edge recommended)
3. Dataset ready (or use built-in example)

Step 1: Access Model Composer
-----------------------------

1. Navigate to https://dev.ti.com/edgeaistudio/
2. Log in with your TI credentials
3. From the dashboard, select **Model Composer**
4. Click **Create New Project**

Step 2: Create a Project
------------------------

**Project Setup:**

1. Enter a project name (e.g., "Arc Fault Detection")
2. Select task type: **Time Series Classification**
3. Select target device: **F28P55** (or your target)
4. Click **Create**

You'll be taken to the project dashboard.

Step 3: Upload Dataset
----------------------

**Option A: Use Example Dataset**

1. Click **Datasets** in the sidebar
2. Click **Add Dataset**
3. Select **Use Example Dataset**
4. Choose "DC Arc Fault" from the list
5. Click **Import**

**Option B: Upload Your Own**

1. Click **Datasets** → **Add Dataset**
2. Select **Upload Custom Dataset**
3. Drag and drop your ZIP file or click to browse
4. Wait for upload and validation
5. Preview data to verify

**Dataset Format:**

Your ZIP should contain:

.. code-block:: text

   my_dataset.zip
   ├── annotations.yaml
   └── classes/
       ├── class_a/
       │   ├── sample1.csv
       │   └── sample2.csv
       └── class_b/
           └── ...

Step 4: Configure Feature Extraction
------------------------------------

1. Click **Feature Extraction** in sidebar
2. Select a preset or customize:

   * **Preset**: Choose from dropdown (e.g., "FFT 1024 Input, 256 Features")
   * **Custom**: Configure individual parameters

3. Key settings:

   * **Input Size**: Samples per inference window
   * **Transform**: FFT, Raw, Wavelet
   * **Features**: Number of output features
   * **Frames**: Temporal context

4. Click **Save Configuration**

**Tip:** Start with a preset; customize later if needed.

Step 5: Configure Training
--------------------------

1. Click **Training** in sidebar
2. Configure model settings:

   **Model Selection:**

   * Select model size (1k, 2k, 4k, etc.)
   * NPU devices show NPU-compatible options

   **Training Parameters:**

   * Epochs: 20-50 (start with 20)
   * Batch Size: 256 (default)
   * Learning Rate: 0.001 (default)

   **Quantization:**

   * Enable for NPU devices
   * Select INT8 (recommended)

3. Click **Save Configuration**

Step 6: Start Training
----------------------

1. Click **Train Model** button
2. Confirm your settings in the dialog
3. Click **Start Training**

**During Training:**

* Progress bar shows completion
* Loss graph updates in real-time
* Training log shows detailed status
* Estimated time remaining displayed

**Training Complete:**

* Success message appears
* Results summary shown
* Model ready for analysis

Step 7: Analyze Results
-----------------------

After training, review model performance:

**Accuracy Tab:**

* Overall accuracy percentage
* Per-class accuracy breakdown
* F1 scores

**Confusion Matrix Tab:**

* Visual confusion matrix
* Click cells for sample details
* Identify which classes are confused

**ROC Curves Tab:**

* One-vs-rest ROC for each class
* AUC scores
* Threshold selection helper

**Sample Viewer Tab:**

* View individual predictions
* See correct and incorrect samples
* Understand model behavior

Step 8: Evaluate on Test Set
----------------------------

1. Click **Testing** in sidebar
2. Select test dataset (holdout data)
3. Click **Run Evaluation**
4. Review test set metrics

Compare test metrics to training metrics:

* Similar = Good generalization
* Test much lower = Possible overfitting

Step 9: Export Model
--------------------

1. Click **Export** in sidebar
2. Choose export format:

   * **CCS Project**: Complete project template
   * **Artifacts Only**: Just model files
   * **Configuration**: YAML config for CLI

3. Click **Download**
4. Extract the ZIP file

See :doc:`exporting_models` for detailed export instructions.

Using Exported Model
--------------------

After export, you'll have files for CCS:

.. code-block:: text

   export/
   ├── mod.a                    # Model library
   ├── mod.h                    # Header file
   ├── feature_extraction.c    # Feature code
   └── example_project/        # CCS project (if selected)

See :doc:`/deployment/ccs_integration` for deployment steps.

GUI Tips and Tricks
-------------------

**Keyboard Shortcuts:**

* ``Ctrl+S``: Save current configuration
* ``Ctrl+Z``: Undo last change
* ``Ctrl+Enter``: Start training

**Dataset Preview:**

* Click any sample in dataset view
* View raw data and labels
* Check for data quality issues

**Training History:**

* All training runs are saved
* Compare runs in History tab
* Resume from any previous run

**Quick Actions:**

* Duplicate project: Copy with one click
* Reset configuration: Return to defaults
* Export configuration: Share settings

Troubleshooting
---------------

**Dataset Upload Fails:**

* Check file size limits
* Verify ZIP structure
* Ensure valid CSV format

**Training Doesn't Start:**

* Check browser console for errors
* Refresh page and retry
* Verify dataset is properly loaded

**Slow Training:**

* Reduce batch size
* Use smaller model
* Check internet connection (cloud)

**Export Download Fails:**

* Check browser download settings
* Try different browser
* Contact support if persists

What's Next?
------------

After completing your first model:

1. **Try your own data**: Upload custom dataset
2. **Experiment with settings**: Different models, features
3. **Compare results**: Use training history
4. **Deploy to device**: Follow CCS integration guide

Additional Resources
--------------------

* :doc:`overview` - Model Composer features
* :doc:`exporting_models` - Export options
* :doc:`/getting_started/first_example` - CLI equivalent
* :doc:`/deployment/ccs_integration` - Device deployment
