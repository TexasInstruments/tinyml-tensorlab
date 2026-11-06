====================================
Machine Readable Code Classification
====================================

Classify a 28x28 image as a QR code, a barcode or something else, on the MSPM0G5187 with a LeNet-5 CNN.

Example folder: ``examples/machine_readable_code_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/machine_readable_code_classification>`_)

Overview
--------

* **Task**: image classification (``image_classification``, ``target_module: vision``), 3 classes
* **Application**: tiny visual pre-classification of machine readable codes on low-memory, low-power edge devices.
  The goal is not to decode the code, only to classify its visual structure. At 28x28 many generated codes would not
  be scannable by a real decoder, which is acceptable here.
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config in the folder)
* **Model**: ``Lenet5``

The Dataset
-----------

A synthetic dataset similar in size and format to MNIST: 28x28, one channel, binary black and white PNG images. The
default size is 3000 images per class, 9000 in total.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Class
     - How it is generated
   * - ``qr``
     - QR codes from the Python ``qrcode`` package (version 1, error correction L, box size 4, border 1) with a
       random alphanumeric payload
   * - ``barcode``
     - Code128 barcodes from the ``python-barcode`` package with the text under the barcode disabled, so the model
       learns the line structure and not printed text
   * - ``other``
     - Blank white, blank black, random binary noise, random line patterns and random block patterns, so the class
       is not a single trivial pattern

All images go through the same conversion: grayscale, nearest-neighbor resize to 28x28 (to keep hard edges),
threshold at 127, pixels set to 0 or 255. The script uses a fixed seed (42), so the dataset is reproducible with the
same package versions.

The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/machine_readable_code_classification.zip``. To regenerate the
data, install the dependencies and run the generator:

.. code-block:: bash

   python -m pip install "qrcode[pil]" "python-barcode" pillow numpy
   cd examples/machine_readable_code_classification
   python generate_machine_readable_code_28x28.py

The dataset layout is one folder per class (``qr``, ``barcode``, ``other``). Point ``dataset.input_data_path`` at
the generated zip or folder to use it.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/machine_readable_code_classification/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\machine_readable_code_classification\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: vision
     task_type: image_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: machine_readable_code_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/machine_readable_code_classification.zip
   data_processing_feature_extraction:
     feature_extraction_name: MachineReadable_Default
   training:
     model_name: Lenet5
     batch_size: 64
     training_epochs: 10
     num_gpus: 0
     quantization: 2
     learning_rate: 0.1
     weight_decay: 1e-5
   testing: {}
   compilation: {}

The preset ``MachineReadable_Default`` converts to grayscale, resizes to 28x28 and normalizes with mean 0.5 and
scale 0.5. The image models enabled in 1.5.0 are ``Lenet5``, ``MobileNetV1_28k_NPU`` and ``MobileNetV2_58k_NPU``.
For the image task type see :doc:`/task_types/image_classification`.

Results
-------

The readme does not publish accuracy numbers. In the automated 1.5.0 config test run (test set of 900 images, 300
per class) the quantized model reached 100.00% accuracy and an AUC ROC of 1.000. The classes are visually very
distinct and the data is synthetic, so expect lower accuracy on photographs of real codes.

Output Location
---------------

Results are written to ``./data/projects/machine_readable_code_classification/run/<date-time>/Lenet5/``. See
:doc:`/getting_started/understanding_run_output` and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` for deployment.

See Also
--------

* :doc:`mnist_image_classification`
* :doc:`coffee_bean_classification`
* :doc:`image_classification_example`
