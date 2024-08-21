# Tiny ML TinyVerse

- This repository is a collection of all the base level scripts that are meant to be stitched together and run by the Tiny ML-Modelmaker
- Although the scripts itself can run standalone, it is not recommended to play with it unless you're a developer who understands what is going in here.

## Capabilities Supported

| Modality    | Task           | Models Supported                                                                        |
|-------------|----------------|-----------------------------------------------------------------------------------------|
| Time Series | Classification | ArcDet4x16, ArcDet6x16                                                                  |
| Audio       | Classification | - MatchboxNet - with MFCC <br/>- TCResNet - with STFT/MFCC <br/>- M5 - no preprocessing |


 # Project tree
 *   [README.md](./README.md)
 * ###  [datasets](./datasets) - Contains user dataset related info 
 * ###  [references](tinyml_tinyverse/references) - Contains training and compilation scripts for each \<modality>_\<task>
   * ###  [common](tinyml_tinyverse/references/common)
     * [compilation.py](tinyml_tinyverse/references/common/compilation.py)
   * ###  [timeseries_classification](tinyml_tinyverse/references/timeseries_classification)
     * ####  [train.py](tinyml_tinyverse/references/timeseries_classification/train.py)
 * ###  [requirements](./requirements)
   *   [requirements.txt](./requirements/requirements.txt)
   * [requirements_ti_packages.txt](./requirements/requirements_ti_packages.txt)
 * ###  [tinyml_tinyverse](./tinyml_tinyverse)
   * ####  [\_\_init__.py](./tinyml_tinyverse/\_\_init__.py)
   * ####  [common](./tinyml_tinyverse/common)
     * [\_\_init__.py](./tinyml_tinyverse/common/\_\_init__.py)
     * [compilation](./tinyml_tinyverse/common/compilation)
       * [\_\_init__.py](./tinyml_tinyverse/common/compilation/\_\_init__.py)
       * [tinyml_benchmark.py](./tinyml_tinyverse/common/compilation/tinyml_benchmark.py)
       * [tvm_input_config.py](./tinyml_tinyverse/common/compilation/tvm_input_config.py)
     * [datasets](./tinyml_tinyverse/common/datasets)
       * [\_\_init__.py](./tinyml_tinyverse/common/datasets/\_\_init__.py)
       * [generic_dataloaders.py](./tinyml_tinyverse/common/datasets/generic_dataloaders.py)
       * [timeseries_dataset.py](./tinyml_tinyverse/common/datasets/timeseries_dataset.py)
     * [models](./tinyml_tinyverse/common/models)
       * [\_\_init__.py](./tinyml_tinyverse/common/models/\_\_init__.py)
       * [generic_models.py](./tinyml_tinyverse/common/models/generic_models.py)
     * [transforms](./tinyml_tinyverse/common/transforms)
       * [\_\_init__.py](./tinyml_tinyverse/common/transforms/\_\_init__.py)
       * [basic_transforms.py](./tinyml_tinyverse/common/transforms/basic_transforms.py)
     * [utils](./tinyml_tinyverse/common/utils)
       * [\_\_init__.py](./tinyml_tinyverse/common/utils/\_\_init__.py)
       * [mdcl_utils.py](./tinyml_tinyverse/common/utils/mdcl_utils.py)
       * [misc_utils.py](./tinyml_tinyverse/common/utils/misc_utils.py)
       * [utils.py](./tinyml_tinyverse/common/utils/utils.py)
    
---
## Contributor and Maintainer
- [Adithya Thonse](https://github.com/Adithya-Thonse)
