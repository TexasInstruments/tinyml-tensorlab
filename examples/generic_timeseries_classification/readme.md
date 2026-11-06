# Generic Timeseries Classification

  A simple timeseries classification example that classifies sine, square, and sawtooth waveforms.

  ## How to Run

  After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

  **Windows:**
  ```bash
  .\run_tinyml_modelzoo.bat examples\generic_timeseries_classification\<config_file>
  ```

  **Linux:**
  ```bash
  ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/<config_file>
  ```

 ## Available Default Configurations For Each Device Family

  - config.yaml - F28P55 (`generic_timeseries_classification` dataset preset, feature extraction `Generic_1024Input_FFTBIN_64Feature_8Frame`, model `CLS_1k_NPU`, batch size 256, 20 epochs, early stopping with patience 5)
  - config_MSPM0.yaml - MSPM0G5187 (model `CLS_13k_NPU`)
  - config_CC1352.yaml - CC1352 (model `CLS_1k_NPU`)
  - config_CC2755.yaml - CC2755 (model `CLS_1k_NPU`)
  - config_CC1354.yaml - CC1354 (model `CLS_1k_NPU`)
  - config_CC35X1.yaml - CC35X1 (model `CLS_1k_NPU`)

  The MSPM0 and CC* configs use the `hello_world_example_dsg` dataset (downloaded from `generic_timeseries_classification.zip`), feature extraction `FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth`, batch size 256 and 20 epochs.
