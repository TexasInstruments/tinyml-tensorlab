# TinyML TinyVerse

- This repository is a collection of all the base level scripts that are meant to be stitched together and run by the TinyML-Modelmaker
- Although the scripts itself can run standalone, it is not recommended to play with it unless you're a developer who understands what is going in here.

## Capabilities Supported

| Modality    | Task           | Models Supported                                                                        |
|-------------|----------------|-----------------------------------------------------------------------------------------|
| Time Series | Classification | ArcDet4x16, ArcDet6x16                                                                  |
| Audio       | Classification | - MatchboxNet - with MFCC <br/>- TCResNet - with STFT/MFCC <br/>- M5 - no preprocessing |


 # Project tree
 *   [README.md](./README.md)
 * ###  [datasets](./datasets) - Contains user dataset related info 
 * ###  [references](./references) - Contains training and compilation scripts for each \<modality>_\<task>
   * ####  [audio_classification](./references/audio_classification)
     * [train.py](./references/audio_classification/train.py)
   * ####  [common](./references/common)
     * [compilation.py](./references/common/compilation.py)
   * ####  [timeseries_classification](./references/timeseries_classification)
   * ####  [train.py](./references/timeseries_classification/train.py)
 * ###  [requirements](./requirements)
   * ####  [requirements.txt](./requirements/requirements.txt)
 * ###  [setup.py](./setup.py)
 * ###  [tinyml_tinyverse](./tinyml_tinyverse)
   * ####  [\_\_init__.py](./tinyml_tinyverse/\_\_init__.py)
   * ####  [common](./tinyml_tinyverse/common)
     * [\_\_init__.py](./tinyml_tinyverse/common/\_\_init__.py)
     * [compilation](./tinyml_tinyverse/common/compilation)
       * [\_\_init__.py](./tinyml_tinyverse/common/compilation/\_\_init__.py)
       * [templates](./tinyml_tinyverse/common/compilation/templates)
         * [c_wrapper_template.txt](./tinyml_tinyverse/common/compilation/templates/c_wrapper_template.txt)
         * [c_wrapper_template_old.txt](./tinyml_tinyverse/common/compilation/templates/c_wrapper_template_old.txt)
       * [tinie-api](./tinyml_tinyverse/common/compilation/tinie-api)
         * [hard](./tinyml_tinyverse/common/compilation/tinie-api/hard)
           * [inc](./tinyml_tinyverse/common/compilation/tinie-api/hard/inc)
           * [ins](./tinyml_tinyverse/common/compilation/tinie-api/hard/ins)
           * [pyutils](./tinyml_tinyverse/common/compilation/tinie-api/hard/pyutils)
           * [src](./tinyml_tinyverse/common/compilation/tinie-api/hard/src)
           * [startup_theflash_ticlang.c](./tinyml_tinyverse/common/compilation/tinie-api/hard/src/startup_theflash_ticlang.c)
           * [tinie_m0.c](./tinyml_tinyverse/common/compilation/tinie-api/hard/src/tinie_m0.c)
         * [soft](./tinyml_tinyverse/common/compilation/tinie-api/soft)
         * [inc](./tinyml_tinyverse/common/compilation/tinie-api/soft/inc)
         * [src](./tinyml_tinyverse/common/compilation/tinie-api/soft/src)
         * [tinie.c](./tinyml_tinyverse/common/compilation/tinie-api/soft/src/tinie.c)
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
   * ####  [sensory](./tinyml_tinyverse/sensory)
     * [\_\_init__.py](./tinyml_tinyverse/sensory/\_\_init__.py)
     * [datasets](./tinyml_tinyverse/sensory/datasets)
       * [\_\_init__.py](./tinyml_tinyverse/sensory/datasets/\_\_init__.py)
     * [models](./tinyml_tinyverse/sensory/models)
       * [\_\_init__.py](./tinyml_tinyverse/sensory/models/\_\_init__.py)
     * [training](./tinyml_tinyverse/sensory/training)
   * ####  [vision](./tinyml_tinyverse/vision)
   * ####  [\_\_init__.py](./tinyml_tinyverse/vision/\_\_init__.py)
   * ####  [datasets](./tinyml_tinyverse/vision/datasets)
     * [\_\_init__.py](./tinyml_tinyverse/vision/datasets/\_\_init__.py)
   * ####  [models](./tinyml_tinyverse/vision/models)
   * ####  [\_\_init__.py](./tinyml_tinyverse/vision/models/\_\_init__.py)