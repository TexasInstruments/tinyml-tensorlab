# Grid Stability: Understanding ModelMaker Outputs
### - Tushar Sharma, Adithya Thonse, Fasna S 
<hr>

## Overview

The Grid Stability dataset (from the UCI Machine Learning Repository) is a synthetic / simulation-derived dataset used to study the small-signal stability of a simple power network. The network model is a 4-node "star" system with one central generation node and three peripheral consumer nodes; each node is described by three parameters, giving 12 independent input variables in total. The grid is considered "stable" if, after a small perturbation (for example a step change in power or a short fault), the system variables return to an operating point without oscillating uncontrollably or diverging. The dataset contains two types of targets related to stability: a numerical stability index (useful for regression) and a categorical stable/unstable label (useful for classification). The samples are produced by running stability analyses across a range of parameter combinations (varying `tau`, `p`, and `g`) and recording whether the linearized system returns to an operating point after perturbations or diverges/oscillates. [Read More](https://arxiv.org/abs/1508.02217) 

1. tau1..tau4 **(reaction time)**: a time-constant representing how quickly each node (generator or consumer) adjusts its power in response to system changes. Larger values indicate slower dynamics and more inertia.
2. p1..p4 **(power)**: the active power level at each node (positive values typically indicate generation, negative values indicate load/consumption depending on sign convention used in the simulation).
3. g1..g4 **(price elasticity)**: a coefficient that models how strongly a node's consumption/generation responds to price signals or control incentives.

Why this dataset is useful:
- It isolates how node dynamics (reaction times), operating points (power injections/demands), and control sensitivity (price elasticity) affect the overall small-signal stability of a compact power network.
- It is ideal for exploring supervised learning tasks (classification and regression), feature engineering (e.g., forming aggregated statistics, temporal features, or physics-informed combinations), and model-interpretability experiments (which input combinations most strongly predict instability).
- Because the dataset is simulation-based, make sure your train/validation splits respect the range of operating conditions (avoid overfitting to narrow parameter ranges).

<p align="center">  
    <img src="assets/four_node_start.svg" width="480" alt="Image Alt">
</p>

## Downloading the dataset

Prepare the zipped dataset by running the grid_stability python file. The script will create zipped dataset as `grid_stability_dataset.zip`. 
```bash
cd examples/grid_stability
python grid_stability.py
```
The path of this zipped dataset file is already mentioned in [configuration](config.yaml) yaml, make sure it is same.

```yaml
dataset:
    input_data_path: 'examples/grid_stability/grid_stability_dataset.zip'
```

This zipped dataset is designed to work with TinyML ModelMaker. Run the modelmaker with the [configuration](config.yaml) yaml.

## How ModelMaker helps ?

TinyML ModelMaker is an end-to-end model development tool that provides dataset handling, model training and compilation. You can run the modelmaker for the given configuration yaml by the following command in terminal.

- "Feature extraction" refers to any preprocessing applied to raw numerical inputs before training (normalization, PCA, signal transforms). For time-series or dynamic systems, feature extraction often includes sliding-window statistics or spectral transforms.
- The `training/base` folder stores the float (non-quantized) model and feature data used for baseline experiments. The `training/quantization` folder contains int8 (quantized) model and analysis for the post-training quantized (int8) model.

Interpreting PCA plots:
- PCA (Principal Component Analysis) reduces high-dimensional feature vectors to 2D or 3D for visualization. PCA plots help you see whether classes separate well in feature space — well separated clusters suggest easier learning for simple models.

```bash
cd tinyml-modelmaker
run_tinyml_modelmaker.sh examples/grid_stability/config.yaml
```

ModelMaker will start by loading the dataset, train the model, test the model, compile the model. ModelMaker will create the output folder in `tinyml-modelmaker/data/projects/{dataset_name}`. For this example dataset_name is `grid_stability`. You can find the dataset name in the [configuration](config.yaml) yaml.

```yaml
dataset:
    dataset_name: grid_stability
```

The data folder structure of tinyml-modelmaker is as follows:

```
tinyml-modelmaker
|_ data
    |_ descriptions
        |_ description_timeseries.json
        |_ description_timeseries.yaml
    |_ projects
        |_ electrical_fault
        |_ grid_stability   # (dataset_name)
            |_ dataset
                |_ annotations
                |_ classes
                    |_ class0_stable
                    |_ class1_unstable
            |_ run
                |_ 20250226-132343  # (date-time)
                    |_ CLS_ResSlice_3k    # (model_name)
                        |_ compilation
                        |_ training
                            |_ base
                            |_ quantization
                |_ 20250227-111317
        |_ motor_fault_example_dsk
```
TinyML ModelMaker outputs extracted features, pca analysis of extracted features, model, compiled model, analysis of extracted features, test setup for testing model on device in `data/projects/grid_stability/run/date-time/model_name`. Let's look at this folder in depth.
```
|_ CLS_ResSlice_3k (model_name)
    |_ compilation
        |_ artifacts
            |_ mod.a
            |_ tvmgen_default.h
    |_ training
        |_ base
            |_ feat_ext_data
            |_ golden_vectors
                |_ test_vector.c
                |_ user_input_config.h
            |_ model.onnx
            |_ pca_on_feature_extracted_train_data.png
            |_ pca_on_feature_extracted_validation_data.png
        |_ quantization
            |_ feat_ext_data
            |_ golden_vectors
                |_ test_vector.c
                |_ user_input_config.h
            |_ post_training_analysis
            |_ model.onnx
            |_ pca_on_feature_extracted_train_data.png
            |_ pca_on_feature_extracted_validation_data.png
```
1. `compilation` (stores the compiled model which can be used in target device)
    1. **artifacts**
        1. **mod.a**: archive library (.a) model, it can be statically linked to your program while compiling
        2. **tvmgen_default.h** : header file to use the functions present in mod.a
2. `training` (related to extracted features, test setup for model and model)
    1. **base**
        1. feat_ext_data: stores the extracted features of dataset in NumPy (.npy)
        2. golden_vectors
            1. test_vector.c: contains test data and its output, this can be used to check if the model is working correctly in device
            2. user_input_config.h: configuration of feature extraction for ai library present in c2000ware
        3. model.onnx: onnx floating point model
        4. pca_on_feature_extracted_train_data.png: PCA analysis of extracted features. [Refer here](../../docs/how_good_is_your_feature_extraction/readme.md)
    2. **quantization**
        1. feat_ext_data: stores the extracted features of dataset in NumPy (.npy)
        2. **golden_vectors**
            1. **test_vector.c**: contains test data and its output, this can be used to check if the model is working correctly in device
            2. **user_input_config.h**: configuration of feature extraction for ai library present in c2000ware
        3. **model.onnx**: onnx fixed point model
        4. pca_on_feature_extracted_train_data.png: PCA analysis of extracted features. [Refer here](../../docs/how_good_is_your_feature_extraction/readme.md)
        5. **post_training_analysis**: Performing a quick insight on the model performance (not latency related). [Refer here](../../docs/post_training_analysis/Post_Training_Analysis.md)
3. run.json, run.yaml, status.json, status.yaml: complete list of configurations used to run the ModelMaker

## Output for Target Device

The target device (such as f28p55x) has four useful file outputs by ModelMaker.
- `mod.a`: The ONNX model is compiled by tvm to get C files, which are converted into a single mod.a that can run on device.
- `tvmgen_default.h`: Mod.a exposes few APIs to interact with model which are present here. You can use these APIs in your application to run model

- `test_vector.c`: ModelMaker gives a test dataset and the expected output. You can use the model to inference this test dataset and check if the output is matching. 
- `user_input_config.h`: This configuration file has preprocessing flag definitions for the parameters used for feature extraction.

These 4 files can be used in a CCS Project to perform AI on edge. [Refer here](../../docs/running_model_on_device/readme.md)

## Performance on device

We benchmarked the performance of the `CLS_ResCat_3k` model. The device used is F28P55x which comes with a HW accelearator (TINPU) to give low latency performance on ML models. Numbers are provided for running the model on CPU and NPU.

|              Configuration             | AI Model Cycles | Inference Time (us) | Flash Usage (B) | SRAM Usage (B) |
|----------------------------------------|-----------------|---------------------|-----------------|----------------|
|  **CPU (without Feature Extraction)**  |      955669     |      6371.13        |      2976       |     10112      |
|  **NPU (without Feature Extraction)**  |      239350     |      1595.67        |      3168       |     7686       |
|   **CPU (with Feature Extraction)**    |      500860     |      3339.07        |      2995       |     4992       |
|   **NPU (with Feature Extraction)**    |      125510     |       836.73        |      3175       |     3846       |

<hr>
Update history:

[29th Dec 2025]: Compatible with v1.2 of Tiny ML Modelmaker
[14th Mar 2025]: Compatible with v1.0 of Tiny ML Modelmaker
