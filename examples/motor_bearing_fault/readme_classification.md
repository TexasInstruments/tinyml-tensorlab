# Motor Bearing Fault

  A simple timeseries classification example classify 5 bearing fault types plus normal operation from vibration data.

  ## How to Run

  After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

  **Windows:**
  ```bash
  .\run_tinyml_modelzoo.bat examples\motor_bearing_fault\<config_file>
  ```

  **Linux:**
  ```bash
  ./run_tinyml_modelzoo.sh examples/motor_bearing_fault/<config_file>
  ```

 ## Available Default Configurations For Each Device Family

  - config.yaml - F28P55 (dataset `motor_fault_example_dsk`, model `CLS_1k_NPU`, batch size 256, 20 epochs, early stopping with patience 5)
  - config_MSPM0.yaml - MSPM0G5187 (dataset `motor_fault_example_dsk`, feature extraction `MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1`, model `CLS_1k_NPU`, 20 epochs, QAT with 4-bit weights and 8-bit activations)

 Anomaly detection configurations (`config_anomaly_detection.yaml`, `config_anomaly_detection_MSPM0.yaml`) are described in [readme_anomaly_detection.md](readme_anomaly_detection.md).
