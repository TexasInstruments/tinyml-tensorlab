import os
import requests as re

import pandas as pd

# Project Definition
DATASET_NAME = "grid_stability"

# Dataset definition
DATASET_URL = "https://archive.ics.uci.edu/static/public/471/electrical+grid+stability+simulated+data.zip"
DATASET_FILE_NAME = "Data_for_UCI_named.csv"
independent_variables = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
dependent_variables = ["stabf"]

# Dataset Processing
NUM_SPLITS = 4
rename_target = {dependent_variables[0]: "Target"}
replace_target_values = {}

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
    data = pd.read_csv(f"{DATASET_NAME}/{DATASET_FILE_NAME}", index_col=False)
    columns = independent_variables + dependent_variables
    data = data[columns]
    data.rename(columns=rename_target, inplace=True)
    data['Target'] = data['Target'].replace(replace_target_values)
    return data


def store_datafiles(df):
    unique_targets = df['Target'].unique()
    if not os.path.exists("classes"):
        os.mkdir("classes")

    for idx, target in enumerate(unique_targets):
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

def cleanup():
    os.system(f"rm -rf {DATASET_NAME} > /dev/null 2>&1")
    os.system(f"rm {DATASET_NAME}.zip > /dev/null 2>&1")

def zip_files():
    print(f"Zipping the classes into {DATASET_NAME}_dataset.zip")
    os.system(f"zip -r {DATASET_NAME}_dataset.zip classes > /dev/null 2>&1")
    os.system("rm -rf classes > /dev/null 2>&1")

if __name__ == '__main__':

    download_data()
    extract_data()

    df = load_data()

    store_datafiles(df)
    zip_files()

    cleanup()
