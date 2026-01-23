# File-Level Classification Summary

### - Fasna Sharaf, Adithya Thonse, Tushar Sharma, Jaswanth Kumar, Abhijeet Pal

<hr>

## Overview

The File-level Classification Summary provides an overview of how samples from each input file are classified into different classes. It helps users quickly identify if any particular file contains misclassified samples.

While the confusion matrix shows overall counts of correct and incorrect classifications, it doesn't reveal which files contain those misclassified samples. For example, even if the total misclassification count is small, it might come entirely from one problematic file, and this feature helps pinpoint such cases instantly.

<hr> 

## Example output

Here we have a fan blade fault classification dataset (You can read more about it here: [Link](./../../examples/fan_blade_fault_classification/readme.md) ) consiting of four classes: **Normal**, **Blade Damage**, **Blade Imbalance** and **Blade Obstruction**. To demonstrate the usefulness of this feature, we have intentionally created a misclassified dataset using [this](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/fan_blade_fault_dsi_misclassified.zip) dataset.

We have created a `config.yaml` file where we define parameters related to dataset, feature extraction transforms, training, etc.

You can run this in modelmaker using this command:-

```bash
./run_tinyml_modelmaker.sh docs/file_level_classification_summary/config.yaml
```

Upon running, you will see a log file named `file_level_classification_summary.log` inside `tinyml-modelmaker/data/projects/{dataset_name}/run/{date-time}/{model_name}/training/base` directory.

In the log file, you will see tables for float train, quant train, and test data, depending on whether you have enabled them in the config file. Each table shows each file, its true class, and the number of samples from that file classified into each class. You can view the log file [here](./file_level_classification_summary.log).

The confusion matrix of float train best epoch is:-


| Ground Truth                  | Predicted as: BladeDamage | Predicted as: BladeImbalance | Predicted as: BladeObstruction | Predicted as: Normal |
|-------------------------------|---------------------------|-----------------------------|--------------------------------|-----------------------|
| **BladeDamage**               | 1159                     | 339                         | 0                              | 0                     |
| **BladeImbalance**            | 0                        | 1301                        | 0                              | 0                     |
| **BladeObstruction**          | 0                        | 0                           | 962                            | 0                     |
| **Normal**                    | 0                        | 0                           | 0                              | 2114                  |


From this confusion matrix, we can see that while all classes other than `BladeDamage` are correctly classified, some `BladeDamage` samples are incorrectly classified as `BladeImbalance`. However, from the confusion matrix alone, we cannot determine which specific files contain these misclassified samples.

When we inspect the File-Level Classification Summary of FloatTrain, we discover that in file numbers 0, 1, 2, 20, and 21, all samples were classified as `BladeImbalance`, even though their true class is `BladeDamage`. **A higher count of samples in the wrong class column for a specific file indicates potential data or labeling issues in that file.**


Similarly, here is the confusion matrix of test data:-


| Ground Truth                  | Predicted as: BladeDamage | Predicted as: BladeImbalance | Predicted as: BladeObstruction | Predicted as: Normal |
|-------------------------------|---------------------------|-----------------------------|--------------------------------|-----------------------|
| **BladeDamage**               | 481                     | 142                         | 0                              | 0                     |
| **BladeImbalance**            | 0                       | 426                         | 0                              | 0                     |
| **BladeObstruction**          | 0                       | 0                           | 378                            | 0                     |
| **Normal**                    | 0                       | 0                           | 0                              | 694                   |

Looking at the File-Level Classification Summary of TestData, we can see that in file numbers 7 and 8, all samples are misclassified. This demonstrates how this feature is ideal for identifying file-level data quality problems that might not be immediately obvious from the confusion matrix alone.

<hr>
Update history:
<br>
[8th October 2025]: Compatible with v1.3 of Tiny ML Modelmaker