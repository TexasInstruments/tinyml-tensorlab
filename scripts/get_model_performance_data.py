import pandas as pd
from pprint import PrettyPrinter
from tinyml_modelmaker.ai_modules.timeseries import constants
from tinyml_modelmaker.ai_modules.vision import constants

valid_model_list = ['MotorFault_model_1_t', 'MotorFault_model_2_t', 'MotorFault_model_3_t',
                    'ArcFault_model_200_t', 'ArcFault_model_300_t', 'ArcFault_model_700_t', 'ArcFault_model_1400_t',
                    'TimeSeries_Generic_1k_t', 'TimeSeries_Generic_4k_t', 'TimeSeries_Generic_6k_t',
                    'TimeSeries_Generic_13k_t', 'Lenet5', 'PIRDetection_model_1_t']
soft_tinie_targets = ['c28_soft_int_in_int_out']
hard_tinie_targets = ['c28_hard_int_in_int_out']
valid_tinie_targets = soft_tinie_targets + hard_tinie_targets

url = 'https://jenkins-sdomc.dal.design.ti.com/job/build-tinie-tests/lastSuccessfulBuild/artifact/tinie-tests/testing/ti_tests.log'
tvm_results_df = pd.read_csv(url, sep="\s+|\:|\,|\+|\=", engine='python', skiprows=1,
                 header=None).dropna(axis=1).drop(columns=[3, 6, 13])
relevant_df = tvm_results_df.loc[tvm_results_df[0].isin(valid_model_list) & tvm_results_df[1].isin(valid_tinie_targets)]
relevant_df.columns = ['model', 'target_tinie_type', 'cycles', 'code', 'ro', 'rw', 'total']

# Now convert c28_soft_int_in_int_out to 2 columns having c28 and soft while removing int_in_int_out
relevant_df['target_tinie_type'] = relevant_df['target_tinie_type'].str.split('_').str[:2].str.join('_')
relevant_df[['target', 'tinie_type']] = relevant_df['target_tinie_type'].str.split('_', expand=True)

# Do a few calculations and add relevant columns
relevant_df['flash'] = (relevant_df['ro'] + relevant_df['code']).astype(object)
relevant_df['sram'] = relevant_df['rw'].astype(object)
relevant_df = relevant_df.drop(columns=['target_tinie_type', 'code', 'ro', 'rw', 'total'])
relevant_df = relevant_df.set_index(['model'])  # 'cycles',

device_list = [constants.TARGET_DEVICE_F280013, constants.TARGET_DEVICE_F280015, constants.TARGET_DEVICE_F28003, constants.TARGET_DEVICE_F28004,
               constants.TARGET_DEVICE_F2837, constants.TARGET_DEVICE_F28P65,  constants.TARGET_DEVICE_MSPM0G3507,
    constants.TARGET_DEVICE_MSPM0G5187,constants.TARGET_DEVICE_CC2755]
freq_MHz_dict  = {
    constants.TARGET_DEVICE_F280013: 120,
    constants.TARGET_DEVICE_F280015: 120,
    constants.TARGET_DEVICE_F28003: 120,
    constants.TARGET_DEVICE_F28004: 100,
    constants.TARGET_DEVICE_F2837: 120,
    constants.TARGET_DEVICE_F28P65: 200,
    constants.TARGET_DEVICE_F28P55: 150,
    constants.TARGET_DEVICE_MSPM0G3507: 111,
    constants.TARGET_DEVICE_MSPM0G5187: 111,
    constants.TARGET_DEVICE_CC2755: 96,
}

hard_tinie_df = relevant_df.loc[relevant_df.tinie_type=='hard']
hard_tinie_df['inference_time_us'] = (hard_tinie_df['cycles']/freq_MHz_dict[constants.TARGET_DEVICE_F28P55]).astype(int).astype(object)
hard_tinie_df['device'] = constants.TARGET_DEVICE_F28P55

soft_tinie_df = relevant_df.loc[relevant_df.tinie_type=='soft']
soft_tinie_df['device'] = [device_list for i in soft_tinie_df.index]
soft_tinie_df = soft_tinie_df.explode('device')
soft_tinie_df['inference_time_us'] = (soft_tinie_df['cycles'] / soft_tinie_df['device'].map(freq_MHz_dict)).astype(int).astype(object)

device_info_df = pd.concat((soft_tinie_df, hard_tinie_df)).drop(columns=['tinie_type', 'target'])  # .reset_index().set_index(['model', 'device'])
device_info_dict = {model: {device: {col: group[col].values[0] for col in ['inference_time_us', 'sram', 'flash']} for device, group in sub_group.groupby('device')} for model, sub_group in device_info_df.groupby('model')}
"""
The above dict comprehension is to get this structure

'motor_fault_3_t': {'F280015': {'flash': 2552,
                                'inference_time_us': 552,
                                'sram': 1158},
                     'F28003': {'flash': 2552,
                                'inference_time_us': 552,
                                'sram': 1158}, ...
"""
print("Please copy the below dictionary to the relevant py script (Example: device_run_info.py).")
print(PrettyPrinter(depth=3).pprint(device_info_dict))
'''
You may have use the regex: 
,\n\s*'i -> , 'i
,\n\s*'s -> , 's
 
'''
