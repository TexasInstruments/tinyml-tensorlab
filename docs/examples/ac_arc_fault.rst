=============
AC Arc Fault
=============

AC arc fault detection is an Edge AI solution for the `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_
microcontroller with an on-chip NPU. It detects series arc faults in residential and commercial
electrical systems, a leading cause of electrical fires, using ML inference on data captured through the
`TIDA-010971 <https://www.ti.com/lit/df/slvrbz4/slvrbz4.pdf>`_ analog front end with a PCB Rogowski coil.
The goal is Arc Fault Circuit Interrupter (AFCI) products that comply with NEC requirements.

Example folder: ``examples/ac_arc_fault``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/ac_arc_fault>`_).
For DC arc faults, see :doc:`arc_fault`.

Overview
--------

* **Task**: binary classification (``task_type: arc_fault``)
* **Device**: MSPM0G5187 (Arm Cortex-M0+ with NPU), config ``config_MSPM0.yaml``
* **Dataset**: ``ac_arc_fault_log300_example_dsk``
* **Model**: ``CLS_1k_NPU``

Why AC Arc Fault Detection?
---------------------------

Arc faults cause more than 30,000 home fires each year in the United States, and traditional circuit
breakers are not designed to detect low-level arcs. Edge AI enables complex pattern recognition,
multi-feature analysis and adaptive detection across load types. The targets of the reference solution are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Target
   * - Response time
     - less than 10 ms
   * - Detection accuracy
     - more than 95%
   * - Compliance
     - `UL 1699 <https://code-authorities.ul.com/wp-content/uploads/2014/05/Dini2.pdf>`_

System Components
-----------------

* Hardware: MSPM0G5187 and the TIDA-010971 analog front end (PCB Rogowski coil)
* Software: Code Composer Studio 12.x or later, `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_ 2.08.00 or later

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/ac_arc_fault/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\ac_arc_fault\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: arc_fault
     target_device: MSPM0G5187
   dataset:
     dataset_name: ac_arc_fault_log300_example_dsk
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/ac_arc_fault_log300.zip
   data_processing_feature_extraction:
     feature_extraction_name: ArcFault_512Input_FE_RFFT_32Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth
     store_feat_ext_data: false
   training:
     model_name: CLS_1k_NPU
     batch_size: 30
     training_epochs: 30
     num_gpus: 0
     quantization: 2
     learning_rate: 0.001
     weight_decay: 0.00001
   testing: {}
   compilation: {}

``quantization: 2`` enables quantization-aware training (INT8), which is required for the MSPM0G5187 NPU.

Dataset
-------

``ac_arc_fault_log300.zip`` contains 100+ captures per load type:

* **Normal operation**: inductive loads, LED lighting, switched-mode power supplies, mixed loads
* **Arc faults**: series and parallel arc faults
* **Masking loads**: dimmer, SMPS, drill, compressor, resistive loads, vacuum cleaner

To prepare and label your own AC arc fault captures use ``examples/ac_arc_fault/AFCI_LabellingScript.py``;
instructions are in ``examples/ac_arc_fault/readme_labelling.md``.

Feature Extraction
------------------

The preset ``ArcFault_512Input_FE_RFFT_32Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth`` mirrors the
on-device pipeline:

1. ADC capture: 512 samples at 107 kSps (about 4.76 ms per frame)
2. 512-point real FFT (ARM CMSIS-DSP on device)
3. Complex magnitude
4. DC removal
5. Binning of 8 adjacent bins, giving 32 features per frame
6. Normalization to the INT8 range
7. Concatenation of 8 frames, giving 256 features per input

Models
------

Any enabled classification model can be used. In 1.5.0 the arc fault specific models are enabled in the
ModelZoo and can be selected with ``training.model_name``, next to the generic NPU models.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Approx. parameters
     - Comment
   * - ``ArcFault_model_200_t``
     - 200
     - Smallest and fastest
   * - ``ArcFault_model_300_t``
     - 300
     -
   * - ``ArcFault_model_700_t``
     - 800
     - Balance of speed and memory
   * - ``ArcFault_model_1400_t``
     - 1,600
     - Most accurate
   * - ``CLS_500_NPU`` / ``CLS_1k_NPU`` / ``CLS_2k_NPU`` / ``CLS_4k_NPU``
     - 500 to 4,000
     - Generic NPU models (``CLS_1k_NPU`` is the default in the config)

Results
-------

The example ``readme.md`` reports these figures for the reference solution on MSPM0G5187 with NPU
(recommended model, 8-frame voting):

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Metric
     - Value
   * - End-to-end latency
     - less than 150 ms
   * - Detection accuracy
     - more than 99%
   * - False positive rate
     - 0.01%
   * - Precision / Recall / F1
     - 99.97% / 99.72% / 99.84%

In the 1.5.0 config test run, ``config_MSPM0.yaml`` with ``CLS_1k_NPU`` reached a test accuracy of 99.64%
with an AUC ROC of 1.000.

Training and Deployment Flow
----------------------------

Running the config performs feature extraction, training, INT8 quantization and compilation with the TI
Neural Network Compiler, producing the model library and interface headers for the MSPM0 SDK. See
:doc:`/DEPLOYING_TO_SDK_EXAMPLE` and :doc:`/deployment/npu_device_deployment`.

References
----------

* `UL 1699 Arc-Fault Circuit Interrupters <https://code-authorities.ul.com/wp-content/uploads/2014/05/Dini2.pdf>`_
* `TIDA-010971 reference design <https://www.ti.com/lit/df/slvrbz4/slvrbz4.pdf>`_
* `MSPM0G5187 product page <https://www.ti.com/product/MSPM0G5187>`_
* `TI Neural Network Compiler guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_

See also :doc:`arc_fault` and :doc:`/byod/classification_format`.
