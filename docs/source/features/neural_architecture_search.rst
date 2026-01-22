===========================
Neural Architecture Search
===========================

Neural Architecture Search (NAS) automatically discovers optimal model
architectures for your specific task and device constraints.

Overview
--------

NAS eliminates manual architecture design by:

* Searching through possible layer configurations
* Evaluating accuracy vs size trade-offs
* Finding Pareto-optimal models
* Respecting device memory and latency constraints

This is especially valuable for MCUs where architecture choices
significantly impact whether a model fits in memory.

When to Use NAS
---------------

Use NAS when:

* You don't know the optimal model size for your task
* You need to balance accuracy vs inference speed
* You want to find the smallest model that meets accuracy requirements
* Manual architecture tuning is time-consuming

Don't use NAS when:

* You have tight time constraints
* A standard model already works well
* You need reproducible results quickly

Enabling NAS
------------

Add the NAS section to your configuration:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'your_dataset'

   nas:
     enable: True
     search_type: 'multi_trial'
     num_trials: 20
     param_range: [500, 5000]
     accuracy_target: 0.95

   training:
     model_name: 'auto'  # NAS will find the model
     training_epochs: 20

   compilation:
     enable: True

Configuration Options
---------------------

**Basic Options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``enable``
     - Set to ``True`` to enable NAS
   * - ``search_type``
     - ``'single_trial'`` or ``'multi_trial'``
   * - ``num_trials``
     - Number of architectures to evaluate
   * - ``param_range``
     - ``[min_params, max_params]`` constraint

**Accuracy Constraints:**

.. code-block:: yaml

   nas:
     accuracy_target: 0.95    # Minimum acceptable accuracy
     accuracy_weight: 1.0     # Weight for accuracy in optimization

**Size Constraints:**

.. code-block:: yaml

   nas:
     param_range: [500, 5000]  # Parameter count bounds
     size_weight: 0.5          # Weight for model size

**Latency Constraints:**

.. code-block:: yaml

   nas:
     latency_target: 1000     # Target inference time (µs)
     latency_weight: 0.3      # Weight for latency

Search Types
------------

**Single Trial Search:**

Evaluates one architecture at a time:

.. code-block:: yaml

   nas:
     search_type: 'single_trial'
     num_trials: 10

* Pros: Lower memory usage, simpler
* Cons: Slower, less exploration

**Multi-Trial Search:**

Evaluates multiple architectures in parallel:

.. code-block:: yaml

   nas:
     search_type: 'multi_trial'
     num_trials: 20
     parallel_trials: 4

* Pros: Faster, better exploration
* Cons: Higher memory usage

Search Space
------------

NAS searches over these architectural dimensions:

**Layer Types:**

* Convolutional layers
* Pooling layers (max, average)
* Fully connected layers
* Batch normalization
* Activation functions

**Hyperparameters:**

* Number of layers
* Channel counts (multiples of 4 for NPU)
* Kernel sizes
* Stride values

**NPU-Aware Search:**

For NPU devices, NAS automatically respects constraints:

.. code-block:: yaml

   common:
     target_device: 'F28P55'  # NPU device

   nas:
     enable: True
     npu_compatible: True     # Enforces NPU constraints

NAS ensures:

* Channels are multiples of 4
* Kernel heights ≤ 7
* Valid stride combinations

Running NAS
-----------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/your_example/config_nas.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\\your_example\\config_nas.yaml

NAS takes longer than single model training (proportional to num_trials).

Output Files
------------

After NAS completes, you'll find:

.. code-block:: text

   .../run/<timestamp>/NAS/
   ├── search_results.csv        # All evaluated architectures
   ├── pareto_frontier.csv       # Optimal trade-off models
   ├── best_model_spec.yaml      # Best model specification
   ├── pareto_plot.png           # Accuracy vs size plot
   └── best_model/
       └── best_model.pt         # Trained best model

**search_results.csv:**

.. code-block:: text

   trial,params,accuracy,latency,architecture
   0,1200,0.92,450,"[conv(4,8,5),conv(8,16,3),fc(16,3)]"
   1,2400,0.96,720,"[conv(4,16,5),conv(16,32,3),fc(32,3)]"
   ...

**pareto_frontier.csv:**

Contains only Pareto-optimal models (best accuracy at each size).

Interpreting Results
--------------------

**Pareto Frontier Plot:**

Shows trade-off between model size and accuracy:

.. code-block:: text

        Accuracy
        ^
   1.0  |          * * *
        |        *
   0.9  |      *
        |    *
   0.8  |  *
        +----------------> Model Size
          1k  2k  3k  4k

Points on the frontier are optimal choices.

**Selecting a Model:**

1. Find your accuracy requirement on the frontier
2. Choose the smallest model meeting that requirement
3. Use the architecture specification from ``best_model_spec.yaml``

Using NAS Results
-----------------

After NAS finds a good architecture, use it for final training:

.. code-block:: yaml

   # Final training with NAS-discovered architecture
   training:
     model_name: 'NAS_result'
     model_spec_path: 'path/to/best_model_spec.yaml'
     training_epochs: 50  # More epochs for final model

Or incorporate the discovered architecture into your model zoo.

Advanced Configuration
----------------------

**Custom Search Space:**

Define specific layers to search:

.. code-block:: yaml

   nas:
     enable: True
     search_space:
       conv_channels: [4, 8, 16, 32]
       conv_kernels: [3, 5, 7]
       num_conv_layers: [2, 3, 4]
       fc_features: [16, 32, 64]

**Early Stopping:**

Stop search when target is met:

.. code-block:: yaml

   nas:
     early_stopping: True
     accuracy_threshold: 0.98
     patience: 5  # Stop if no improvement for 5 trials

**Warm Starting:**

Start from a known good architecture:

.. code-block:: yaml

   nas:
     warm_start: True
     initial_architecture: 'CLS_2k_NPU'

Example: Finding Optimal Arc Fault Model
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

   nas:
     enable: True
     search_type: 'multi_trial'
     num_trials: 30
     param_range: [200, 2000]
     accuracy_target: 0.98
     npu_compatible: True

   training:
     model_name: 'auto'
     training_epochs: 15
     batch_size: 256

   compilation:
     enable: True

**Expected Results:**

NAS might find an architecture like:

.. code-block:: text

   Best Model: 650 parameters
   Accuracy: 98.5%
   Latency: 180 µs (F28P55 NPU)

   Architecture:
   - Conv: 1 → 4 channels, kernel 5
   - Conv: 4 → 8 channels, kernel 3
   - MaxPool: 2x2
   - FC: 64 → 2 (binary classification)

Computational Cost
------------------

NAS requires training multiple models:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Trials
     - Time (20 epochs each)
     - GPU Recommended
   * - 10
     - ~1-2 hours
     - Optional
   * - 30
     - ~3-6 hours
     - Yes
   * - 100
     - ~10-20 hours
     - Yes

**Reducing NAS Time:**

* Fewer training epochs per trial
* Smaller search space
* Early stopping
* Use GPU if available

Best Practices
--------------

1. **Start Simple**: Try standard models first, use NAS if needed
2. **Set Realistic Targets**: Don't aim for 100% accuracy
3. **Constrain Search Space**: Narrow bounds speed up search
4. **Use Appropriate Trials**: 20-50 trials usually sufficient
5. **Validate Results**: Test best model thoroughly before deployment

Next Steps
----------

* Learn about :doc:`quantization` for model compression
* Explore :doc:`feature_extraction` options
* Deploy your model: :doc:`/deployment/npu_device_deployment`
