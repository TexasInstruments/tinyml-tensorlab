# Guide to add models

### To add a model, the following guide will help you
- The model needs to be added to TinyVerse
- Say for example, to add a model for audio tasks
- Head over to `<root>/tinyml-tinyverse/tinyml_tinyverse/common/models`
- In the directory, add a `<model_name>.py` file
- and add the corresponding model name to `__init__.py` in the `model_dict`

---
#### At this stage, the TinyVerse package can function independently with the model

---
#### However, to make sure the Modelmaker also can run the model, we need to add a few more files

- Head over to `<root>/tinyml-modelmaker/tinyml_modelmaker/ai_modules/common/training/tinyml_tinyverse/`
- Note: In the above path `audio` needs to be replaced with the respective ai_module, i.e `timeseries` or `sensory` or `vision` etc
- In case the added model is for a classification task, then go to the script: Eg.: `timeseries_classification.py` 
- Add the model details under `model_urls`, `_model_descriptions` and `enabled_models_list`