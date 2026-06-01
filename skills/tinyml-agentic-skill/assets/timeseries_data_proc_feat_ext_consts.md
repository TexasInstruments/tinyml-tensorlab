DATA_PREPROCESSING_DEFAULT = 'default'
DATA_PREPROCESSING_PRESET_DESCRIPTIONS = dict(
    default=dict(downsampling_factor=1), )
FEATURE_EXTRACTION_DEFAULT = 'default'
FEATURE_EXTRACTION_PRESET_DESCRIPTIONS = dict(
    Custom_ArcFault=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    Custom_ArcFault_MSPM0=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], q15_scale_factor=4, analysis_bandwidth=1, frame_skip=8, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, normalize_bin=True,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    Custom_MotorFault_MSPM0=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=3, q15_scale_factor=5 ),
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Custom_MotorFault=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=3, ),
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Custom_Default=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=[TASK_TYPE_GENERIC_TS_CLASSIFICATION, TASK_TYPE_GENERIC_TS_REGRESSION]), ),
    Custom_Default_MSPM0=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[ 'FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], normalize_bin=True, frame_skip=1, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, q15_scale_factor=5),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=[ 'FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, min_bin=1, frame_skip=1, offset=0, scale=1, variables=1, q15_scale_factor=5, normalize_bin=True, data_proc_transforms=[],),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_256Feature_1Frame_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_256Feature_1Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=122, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_64Feature_4Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=4, min_bin=1, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    FFT1024Input_32Feature_8Frame_Quarter_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=32, num_frame_concat=8, min_bin=1, analysis_bandwidth=4, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    ECG2500Input_Roundoff_1Frame = dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['ROUND_OFF'],frame_size=2500, variables=1,),
        common=dict(task_type=TASK_TYPE_ECG_CLASSIFICATION), ),
    ArcFault_1024Input_256Feature_1Frame_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_256Feature_1Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=256, num_frame_concat=1, min_bin=122, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_64Feature_4Frame_Half_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=4, min_bin=1, analysis_bandwidth=2, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1, ),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_32Feature_8Frame_Quarter_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=32, num_frame_concat=8, min_bin=1, analysis_bandwidth=4, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    # ArcFault_512Input_FFT=dict(
    #     data_processing_feature_extraction=dict(transform=['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=256, num_frame_concat=1, min_bin=1, analysis_bandwidth=1, frame_skip=1, log_mul=10, log_base='e', log_threshold=1e-12, data_proc_transforms=[], sampling_rate=1, new_sr=1, variables=1,),
    #     common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_1024Input_FE_RFFT_128Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=1024, feature_size_per_frame=128, num_frame_concat=8, min_bin=1, frame_skip=8, scale=1, offset=0, normalize_bin=True, variables=1, q15_scale_factor=4, data_proc_transforms=[],),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    ArcFault_512Input_FE_RFFT_32Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=512, feature_size_per_frame=32, num_frame_concat=8, min_bin=1, frame_skip=1, scale=1, offset=0, normalize_bin=True, variables=1, q15_scale_factor=5, data_proc_transforms=[],),
        common=dict(task_type=TASK_TYPE_ARC_FAULT), ),
    Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_1D=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='1D', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3,),  # ch=1,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    Input256_FFT_128Feature_1Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    # MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_1D=dict(
    #     data_processing_feature_extraction=dict(transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, ch=1, offset=0, scale=1, stacking='1D', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=3),
    #     common=dict(task_type=TASK_TYPE_MOTOR_FAULT),),
    Input128_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=3),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),
    MotorFault_256Input_FE_RFFT_16Feature_8Frame_3InputChannel_removeDC_2D1=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, min_bin=1, normalize_bin=True, offset=0, scale=1, frame_skip=1, variables=3, q15_scale_factor=5, data_proc_transforms=[], dc_remove=True, stacking='2D1', ),  # ch=3,
        common=dict(task_type=[TASK_TYPE_MOTOR_FAULT, TASK_TYPE_BLOWER_IMBALANCE]), ),

    Generic_1024Input_FFTBIN_64Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=64, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_FFTBIN_32Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=32, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_FFTBIN_16Feature_8Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=16, num_frame_concat=8, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_1024Input_FFT_512Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=1024, feature_size_per_frame=512, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_FFT_256Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=512, feature_size_per_frame=256, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_FFT_128Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['FFT_FE', 'FFT_POS_HALF', 'ABS', 'DC_REMOVE', 'LOG_DB', 'CONCAT'], frame_size=256, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, log_mul=20, log_base=10, log_threshold=1e-100, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_512Input_RAW_512Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=512, feature_size_per_frame=512, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_256Input_RAW_256Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=256, feature_size_per_frame=256, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    Generic_128Input_RAW_128Feature_1Frame=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['RAW_FE', 'CONCAT'], frame_size=128, feature_size_per_frame=128, num_frame_concat=1, normalize_bin=True, dc_remove=True, offset=0, scale=1, stacking='2D1', frame_skip=1, data_proc_transforms=[], sampling_rate=1, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_GENERIC_TS_CLASSIFICATION), ),
    PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['PIR_FE'], frame_size=125, window_count=25, chunk_size=8, stride_size=0.032, fft_size=64, sampling_rate=33, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_PIR_DETECTION), ),
    PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D_FixedPoint=dict(
        data_processing_feature_extraction=dict(feat_ext_transform=['PIR_FE_Q15'], frame_size=125, window_count=25, chunk_size=8, stride_size=0.032, fft_size=64, sampling_rate=31.25, variables=1),  # ch=3,
        common=dict(task_type=TASK_TYPE_PIR_DETECTION), ),  
)

