DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict( 
    Mnist_Default=dict(
        data_processing_feature_extraction=dict(image_height = 28, image_width = 28, image_num_channel= 1, image_mean= 0.1307, image_scale= 0.3081, variables=1),  
        common=dict(task_type=TASK_TYPE_IMAGE_CLASSIFICATION), ),
)