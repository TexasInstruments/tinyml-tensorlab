==============
Early Stopping
==============

Early stopping ends training as soon as the validation metric stops improving, which
avoids overfitting and saves training time. New in 1.5.0, it is **on by default** for every
time series task type (classification, regression, anomaly detection, forecasting) and for
image classification.

Configuration
=============

The three keys live in the ``training`` section of the config YAML:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Meaning
   * - ``early_stopping``
     - ``True``
     - Enable or disable early stopping.
   * - ``early_stopping_patience``
     - ``5``
     - Number of consecutive epochs without improvement before training stops.
   * - ``early_stopping_min_delta``
     - ``0.0``
     - Minimum change in the monitored metric that counts as an improvement.

.. code-block:: yaml

   training:
     training_epochs: 100
     early_stopping: True
     early_stopping_patience: 5
     early_stopping_min_delta: 0.0

To always train for the full number of epochs, set ``early_stopping: False``.

How it works
============

After each validation pass the training script reports the monitored metric. An epoch counts as
an improvement only if the metric is better than the best value so far by more than
``early_stopping_min_delta``. Otherwise the counter of bad epochs increases, and when it reaches
``early_stopping_patience`` training stops and the log shows
``Early stopping at epoch N: no improvement for <patience> epochs.``
The best model found so far is the one that is kept, as in a normal run.

The monitored metric depends on the task type:

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Task type
     - Monitored metric
     - Better when
   * - Classification (time series, image)
     - Validation accuracy
     - higher
   * - Regression
     - Validation MSE
     - lower
   * - Anomaly detection
     - Validation MSE (reconstruction error)
     - lower
   * - Forecasting
     - Validation SMAPE
     - lower

.. tip::

   With a large ``training_epochs`` value, the default patience of 5 is usually enough. If your validation
   curve is noisy (small datasets), raise ``early_stopping_patience`` or set a small
   ``early_stopping_min_delta`` instead of disabling early stopping.

.. note::

   The audio and radar classification training scripts in 1.5.0 accept the same three keys in
   their parameter set but do not apply early stopping in their training loop.

See :doc:`/CONFIG_YAML_GUIDE` for the rest of the ``training`` keys.
