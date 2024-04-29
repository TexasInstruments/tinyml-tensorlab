# Guide to add feature extractors

### To add a feature extractor, the following guide will help you
- The feature extractor needs to be added to TinyVerse
- Say for example, to add a feature extractor for audio tasks
- Head over to `<root>/tinyml-tinyverse/tinyml_tinyverse/common/transforms`
- In the directory, add a `<feature_extractor_name>.py` file
- and import the corresponding feature extractor name to `common/datasets/timeseries_dataset.py` and use it correspondingly in the data loader.

---
#### At this stage, the TinyVerse package can function independently with the feature extractor
