import os
import pandas as pd
import requests as re

NUM_SPLITS = 2
DATASET_NAME = 'gas_sensor'
DATASET_FILE_NAME = 'gsalc.csv'

replace_ = ['ethanol', 'acetone', 'toluene', 'ethyl_acetate', 'isopropanol', 'hexane']
replace_ = [[name + '100ppb', name + '200ppb', name + '50ppb'] for name in replace_]
replace_ = [name for i in replace_ for name in i]
replace_ = {idx: name for idx, name in enumerate(replace_)}

DATASET_URL = "https://archive.ics.uci.edu/static/public/1081/gas+sensor+array+low-concentration.zip"

def download_data():
    print("Downloading dataset")
    data = re.get(DATASET_URL)
    with open(f"{DATASET_NAME}.zip", "wb") as f:
        f.write(data.content)
    return None

def extract_data():
    print("Extracting dataset")
    os.system(f"unzip {DATASET_NAME}.zip -d {DATASET_NAME} > /dev/null 2>&1")
    return None

def load_data():
    data = pd.read_csv(f"{DATASET_NAME}/{DATASET_FILE_NAME}", header=None)
    data['Target'] = data[0] + data[1]
    dfs = []
    unique_targets = data['Target'].unique()
    for target_idx, target in enumerate(unique_targets):
        current_target_df = data[data['Target'] == target]
        current_target_df = current_target_df.drop([0, 1, 'Target'], axis=1)
        data_ = {}
        for row in current_target_df.values:
            row = row.reshape((10, -1))
            for idx in range(10):
                data_['Sensor_'+ str(idx)] = (row[idx])
            
            current_target_df = pd.DataFrame(data_)
            current_target_df['Target'] = target_idx
            dfs.append(current_target_df)

    data = pd.concat(dfs, ignore_index=True)
    return data

def store_datafiles(df):
    df['Target'] = df['Target'].replace(replace_)
    unique_targets = df['Target'].unique()[9::3]
    if not os.path.exists("classes"):
        os.mkdir("classes")

    for idx, target in enumerate(unique_targets):
        if idx>8:
            continue
        if not os.path.exists(f"classes/class_{idx}_{target.lower()}"):
            os.mkdir(f"classes/class_{idx}_{target.lower()}")
        current_target_df = df[df['Target'] == target]
        num_splits = NUM_SPLITS
        chunk_size = len(current_target_df) // num_splits
        current_target_dfs = [current_target_df[i*chunk_size:(i+1)*chunk_size] for i in range(num_splits)]
        for df_idx, current_target_df in enumerate(current_target_dfs):
            current_target_df = current_target_df.drop(['Target'], axis=1)
            current_target_df.to_csv(f"classes/class_{idx}_{target.lower()}/{target.lower()}_{df_idx}.csv", index=False)
        print("Created classes/class_{}_{}".format(idx, target.lower()))

def zip_files():
    print(f"Zipping the classes into {DATASET_NAME}_dataset.zip")
    if os.path.exists(f"{DATASET_NAME}_dataset.zip"):
        os.remove(f"{DATASET_NAME}_dataset.zip")
    os.system(f"zip -r {DATASET_NAME}_dataset.zip classes > /dev/null 2>&1")
    os.system("rm -rf classes > /dev/null 2>&1")

def cleanup():
    os.system(f"rm -rf {DATASET_NAME} > /dev/null 2>&1")
    os.system(f"rm {DATASET_NAME}.zip > /dev/null 2>&1")
    
if __name__ == '__main__':

    download_data()
    extract_data()

    df = load_data()

    store_datafiles(df)
    zip_files()

    cleanup()