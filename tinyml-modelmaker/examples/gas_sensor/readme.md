# Gas Sensor:  Differences during inference between Native PyTorch quantization and TI-provided quantization Wrappers
### -Tushar Sharma, Adithya Thonse, Fasna S 
<hr>

## Overview

The Gas Sensor Array under Low Concentration dataset, available from the UCI Machine Learning Repository, provides time-series data recorded from a chemical sensor array exposed to low concentrations of six different gases. It is specifically designed to support research on gas identification and quantification at sub-part-per-million (ppb) levels, which is particularly relevant for applications such as environmental monitoring, air quality assessment, and medical diagnostics.

The dataset comprises recordings from 10 metal-oxide semiconductor (MOS) sensors. Each instance in the dataset contains a 30-minute time-series measurement sampled at 1 Hz, resulting in 9000 sensor readings per sample. These readings are split evenly across the 10 sensors, each contributing 900 data points. Each sample corresponds to a specific gas (ethanol, acetone, toluene, ethyl acetate, isopropanol, or n-hexane) at one of three concentration levels: 50, 100, or 200 ppb. In total, the dataset contains 90 samples, five per gas-concentration pair.This dataset is ideal for developing and benchmarking machine learning algorithms for classification (gas type recognition), regression (concentration estimation), and time-series analysis. Its challenges lie in the low signal-to-noise ratio and high dimensionality, making it an excellent testbed for advanced feature extraction.

The dataset used will have 10 measurable parameters/variables - **(Sensor_0, Sensor_1, Sensor_2, Sensor_3, Sensor_4, Sensor_5, Sensor_6, Sensor_7, Sensor_8, Sensor_9)** i.e the readings from gas sensors and there is 1 prediction variable i.e. the gas present and it's concentration.

## Downloading dataset

Prepare the zipped dataset by running the gas_sensor python file. The script will create zipped dataset as `gas_sensor_dataset.zip`. 
```bash
cd examples/gas_sensor
python gas_sensor.py
```
The path of this zipped dataset file is already mentioned in yaml [configuration](config.yaml) yaml, make sure it is same.

```yaml
dataset:
    input_data_path: 'examples/gas_sensor/gas_sensor_dataset.zip'
```

## Usage in TinyML ModelMaker

This zipped dataset is designed to work with TinyML ModelMaker. Run the modelmaker with the yaml [configuration](config.yaml) using the below code.

```bash
run_tinyml_modelmaker.sh examples/gas_sensor/config.yaml
```

1. `run_tinyml_modelmaker.sh` is the script to run modelmaker. It take two required arguments TARGET_SOC and CONFIG_FILE.
2. `F28P55` is the name of target soc present in config.yaml.
3. `examples/gas_sensor/config.yaml` path of the yaml configuration file to run

The users can configure the yaml [configuration](config.yaml) to change parameters related to data preprocessing feature extraction, **training**, testing, model and model compilation. In this example, we will configure the parameter of training, specifically *output_dequantize*. 

```yaml
training:
    output_dequantize: False #True
```

## Quantization Background

ModelMaker handles several functions, one of which is model quantization. After training the model with floating-point precision, it undergoes either Quantization Aware Training (QAT) or Post Training Quantization (PTQ) calibration to produce a quantized version. This quantized model is then transformed into a different format optimized for use with the hardware accelerators available on TI-MCUs. The converted model is subsequently used to validate accuracy. The illustration below demonstrates the inference process using the converted model, where the input provided is (0.91, 0.87, 0.13).

![Quantization & Dequantization](<assets/quantize_vs_dequantize.png>)

The expected output from the floating-point model is (0, 0.89, 0.12), which is essentially a scaled version of the quantized output (0, 13, 7). By default, ModelMaker trains the model to produce quantized outputs, as this approach is faster and operates in the integer domain. In contrast, generating dequantized (floating-point) outputs requires operations in the floating-point domain, which can slow down inference.

## Quantized/Dequantized Output

Users may notice differences during inference between Native PyTorch quantization and the TI-provided quantization wrappers. When using GenericTinyMLQATFxModule or GenericTinyMLPTQFxModule, which rely on Native PyTorch quantization, the output is dequantized by default. In contrast, quantizing the model with TINPUTinyMLQATFxModule or TINPUTinyMLPTQFxModule produces quantized outputs by default.

Our solution offers flexibility to obtain either quantized or dequantized outputs when using the TINYML wrappers (TINPUTinyMLQATFxModule or TINPUTinyMLPTQFxModule). This can be controlled through the **training** section of the YAML configuration by setting the *output_dequantize* parameter to either True or False depending on the desired output format.

- True: A multiply block is added to the last layer of the ONNX model to perform dequantization, resulting in dequantized floating-point outputs.
- False: No multiply block is added to the last layer, and the model outputs remain in quantized integer format.

```yaml
training:
    output_dequantize: True # False (default)
```

Furthermore, you can visualize the outputs realized from models in test_vector.c PATH=*data/projects/gas_sensor/run/<date>-<time>/TimeSeries_Generic_1k_t/training/quantization/golden_vectors/test_vector.c*

The **test_vector.c** file consists of many test cases to verify the outputs of converted model w.r.t qdq quantized model. Let's see the effect of output_dequantize on these test_vectors. For output_dequantize set to:

1. True
    - Floating Model Inference:  int8_t golden_output[3] = { 3, -2, -1, } ;
    - Quantized Model Inference: int8_t golden_output[3] = { 0, 0, 0, } ; # { 0.875, -0.4375, -0.40625}
2. False:
    - Floating Model Inference:  int8_t golden_output[3] = { 3, -2, -1, } ;
    - Quantized Model Inference: int8_t golden_output[3] = { 28, -14, -13, } ;

Since the dequantized output values here are of range (0, 1), we aren't able to see the mantissa of float values when type-casted to int. When output_dequantize is True, the scaling factor found from trained model is 0.03125.
Therefore
```python
    scaling_factor = 0.03125
    dequantized_output = { 0.875, -0.4375, -0.40625 }
    # quantized_output == dequantized_output / scaling_factor
    quantized_output = { 28, -14, -13, }
```

This will match the output from Generic Quantization and TINPU Quantization.

<hr>
Update history:
[30th May 2025]: Compatible with v1.0 of Tiny ML Modelmaker
