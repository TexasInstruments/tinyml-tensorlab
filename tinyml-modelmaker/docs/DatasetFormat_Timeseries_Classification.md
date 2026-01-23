# Dataset & Datafile format: Timeseries Classification

- The dataset format is similar to that of the [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) dataset, but there are some changes as explained below.
 

##  1. Dataset format
The prepared dataset should have the following structure for it to be consumed by the Modelmaker. 

<pre>
dataset_name/
     |
     |--classes/
     |     |-- class1/                        # all the files corresponding to the class1 should be in this folder    
     |     |-- class2/                        # all the files corresponding to the class2 should be in this folder
     |     |-- and_so_on/
     |     |-- classN/                        # You can have as many classes as you want
     |
     |--annotations/
           |--file_list.txt                   # List of all the files in the dataset
           |--instances_train_list.txt        # List of all the files in the train set (subset of file_list.txt)
           |--instances_val_list.txt          # List of all the files in the validation set (subset of file_list.txt) 
           |--instances_test_list.txt         # List of all the files in the test set (subset of file_list.txt)
</pre>

- _annotations_ folder is **optional**. If the folder is not provided, the Modelmaker tool automatically generates one.
- _classes_ folder is mandatory.
- Look at the example dataset [Arc Fault Classification](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/arc_fault_classification_dsk.zip) to understand further.

### Notes

- Modelmaker accepts the dataset path through the `input_data_path` argument in the `config_*.yaml` file.
- The dataset path can be a **URL** or a **local path**.
- The dataset path can be a **zip file** or a **directory**.
  - If it is a zip file, it will be extracted and the path to the extracted folder will be used.
  - The zipped file should contain the `classes` directory immediately inside it. (Don't mess with the hierarchy by adding a level e.g `dataset_name`)
  - If it is a directory, you should give the path until `dataset_name/`. Modelmaker searches for the `classes` directory in this path.
- In the config file, provide the name of the dataset (dataset_name in this example) in the field `dataset_name` and provide the **local path** or **URL** in the field `input_data_path`.
- Then the ModelMaker tool can be invoked with the config file.

---

##  2. Datafile format

There are two accepted formats for how the file can look in the dataset:

* _**Headerless format**_: There is **no header row** and **no index column** in the file.
  * This is usually suitable for single column files (where only one variable is measured) like the [Arc Fault Classification](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/arc_fault_classification_dsk.zip) dataset.
    <pre>
    2078 
    2136 
    2117 
    2077 
    2029 
    1989 
    2056
    </pre>

* **_Headered format_**: There is a header row and an index column in the file.
  * This can be used for single variable of measurement (like current) or multiple variables (e.g x,y,z axes of vibration sensors)
  * Example for single variable of measurement
    <pre>
      Time(sec),I(amp)
    -0.7969984,7.84
    -0.7969952,7.76
    -0.796992,7.76
    -0.7969888,7.76
    -0.7969856,7.76
    -0.7969824,7.84
    -0.7969792,7.84
    </pre>
  * Example for multiple variables of measurement
    <pre>
    Time,Vibx,Viby,Vibz
    19393,-2753,-558,64376
    19394,-2551,-468,63910
    19395,-424,-427,64032
    19396,1429,-763,64132
    19397,1236,-974,64065
    19398,-903,-547,64242
    19399,-1512,-467,63919
    </pre>
  
* As you have seen in the headered format, there are columns named different variations of "time" (`Time(sec)`, `Time`)
* As far as a column has any instance of the text 'time' (case insensitive, E.g `Timestamp`, `TIME`, `TIME (microsec)`), **_this column is always dropped_**.
* So if a column is useful, header should not contain 'time' in the file.

---

## 3. Dataset Splitting

- The dataset can be split into train, validation and test sets.
- The default dataset splitting is 60% train, 30% validation and 10% test.
- The train set is used for training the model.
- The validation set is used for evaluating the model.
- The test set is used for testing the model

- The `annotation` directory provides the way dataset needs to be split
- If the user already provides this dataset, then Modelmaker consumes the provided dataset and splits it into train, validation and test sets as it is
- If the user does not provide this dataset, then Modelmaker will split the dataset into train, validation and test sets automatically
- This split can be done in two ways based on `dataset` section in the config file - split_type: `within_files` or split_type: `amongst_files` --> default
 
```yaml
dataset:
    # enable/disable dataset loading
    enable: True #False
    split_type: within_files  # amongst_files --> default
    split_factor: [0.6, 0.3, 0.1] # This is the default split, for train: val: test
```
### 3.1 amongst_files

- Say you have _10_ files (and each file has **100** lines)
  - This will be split into `6` train files: `3` val files: `1` test file by Modelmaker 
  - each file will still have **100** lines
- The `annotations` directory and all the files under it will be created automatically

### 3.2 within_files

- Say you have _10_ files (and each file has **100** lines)
- This will be split into the same `10` train files: `10` val files: `10` test file by Modelmaker
- However train files will have the first `60` lines, val files will have `30` lines and test files will have the last `10` lines
- The `annotations` directory and all the files under it will be created automatically

---